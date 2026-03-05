"""
trainer.py
----------
Training loop for HybridSwinNet — Spec.md §4 & §5 Phase 3.

Features:
  - AdamW + cosine-annealing LR schedule
  - Mixed-precision training (torch.amp.autocast + GradScaler) for GPU speed
  - Gradient accumulation (simulate larger batches on low-VRAM GPUs)
  - WeightedRandomSampler for class-balanced mini-batches
  - AUC-first metric logging via torchmetrics.BinaryAUROC
  - Best-checkpoint saving by validation AUC
  - Optional Weights & Biases (wandb) logging
  - Resume-from-checkpoint support
  - Gradient clipping to prevent training instability
"""

import csv
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.datasets.base_dataset import DeepfakeDataset
from src.training.losses import BCEWithLogitsLoss

try:
    from torchmetrics.classification import BinaryAUROC
except ImportError:
    BinaryAUROC = None  # type: ignore[assignment,misc]

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# DataLoader factory helpers
# ---------------------------------------------------------------------------

def build_dataloaders(
    config: dict,
    train_transform,
    val_transform,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from config.

    Uses DeepfakeDataset.from_split() to create a stratified internal
    80/20 split from training_data_final/ (the official validation set
    has no labels and is reserved for submission inference).

    Args:
        config:          Full config dict (from configs/default.yaml).
        train_transform: Albumentations transform for training.
        val_transform:   Albumentations transform for validation.

    Returns:
        (train_loader, val_loader)
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    train_ds, val_ds = DeepfakeDataset.from_split(
        root=data_cfg["train_dir"],
        val_split=train_cfg.get("val_split", 0.2),
        seed=train_cfg.get("val_split_seed", 42),
        train_transform=train_transform,
        val_transform=val_transform,
    )

    # WeightedRandomSampler: ensure 1:1 real/fake ratio in each mini-batch
    sample_weights = train_ds.class_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)
    batch_size = train_cfg["batch_size"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"[Dataset] Train: {len(train_ds)} samples | "
        f"Val: {len(val_ds)} samples"
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the full training lifecycle for HybridSwinNet.

    Args:
        model:        Instantiated HybridSwinNet (on CPU; moved to device here).
        train_loader: DataLoader for training split.
        val_loader:   DataLoader for validation split.
        config:       Dict of hyperparameters (from configs/default.yaml).
        device:       torch.device to train on.
        output_dir:   Directory to save checkpoints and metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        output_dir: str = "checkpoints/",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config["training"]
        log_cfg = config.get("logging", {})

        # Apply CUDA memory allocator config (reduces fragmentation on small VRAM)
        cuda_alloc_conf = train_cfg.get("cuda_alloc_conf", "expandable_segments:True")
        if cuda_alloc_conf and torch.cuda.is_available():
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", cuda_alloc_conf)

        # Gradient accumulation: accumulate N micro-batches before stepping optimizer
        self.grad_accum_steps: int = train_cfg.get("gradient_accumulation_steps", 1)

        # ------------------------------------------------------------------
        # Optimizer & scheduler
        # ------------------------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.get("lr", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg.get("epochs", 50),
            eta_min=1e-6,
        )

        # ------------------------------------------------------------------
        # Loss
        # ------------------------------------------------------------------
        self.criterion = BCEWithLogitsLoss(pos_weight=1.0)

        # ------------------------------------------------------------------
        # AMP (mixed precision)
        # ------------------------------------------------------------------
        self.use_amp: bool = (
            train_cfg.get("amp", True) and device.type == "cuda"
        )
        self.scaler: Optional[GradScaler] = (
            GradScaler() if self.use_amp else None
        )
        self.grad_clip: float = train_cfg.get("grad_clip", 1.0)

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------
        if BinaryAUROC is not None:
            self.auroc = BinaryAUROC().to(device)
        else:
            self.auroc = None
            print("[WARN] torchmetrics not installed — AUC logging disabled.")

        # ------------------------------------------------------------------
        # Logging (wandb)
        # ------------------------------------------------------------------
        self.use_wandb = log_cfg.get("use_wandb", False) and wandb is not None
        if self.use_wandb:
            wandb.init(
                project=log_cfg.get("project", "ntire2026-deepfake"),
                name=log_cfg.get("run_name", "hybrid-swin"),
                config=config,
            )
            wandb.watch(self.model, log="gradients", log_freq=100)

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.best_auc: float = 0.0
        self.start_epoch: int = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "best") -> None:
        """Save model weights + training state."""
        path = self.output_dir / f"model_{tag}.pth"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_auc": self.best_auc,
                "epoch": self.start_epoch,
                "config": self.config,
            },
            path,
        )
        print(f"[Checkpoint] Saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        """Resume from a previous checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.best_auc = ckpt.get("best_auc", 0.0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Resume] Loaded checkpoint from '{path}' (epoch {self.start_epoch})")

    # ------------------------------------------------------------------
    # Epoch-level loops
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int, max_steps: Optional[int] = None) -> float:
        """
        Train for one epoch.

        Args:
            epoch:     Current epoch index (0-based).
            max_steps: If set, stop after this many steps (smoke test).

        Returns:
            Average BCE loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"[Train] Epoch {epoch + 1}",
            leave=False,
        )

        self.optimizer.zero_grad()  # reset at start of accumulation cycle

        for step, (images, labels) in enumerate(pbar):
            if max_steps is not None and step >= max_steps:
                break

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Check whether this is the last micro-step in the accumulation window
            is_last_accum = ((step + 1) % self.grad_accum_steps == 0) or \
                            (max_steps is not None and step + 1 >= max_steps)

            # --- Forward + Backward (with optional AMP) ---
            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    # Scale loss so gradients average across accumulation steps
                    loss = self.criterion(logits, labels) / self.grad_accum_steps
                self.scaler.scale(loss).backward()

                if is_last_accum:
                    self.scaler.unscale_(self.optimizer)
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels) / self.grad_accum_steps
                loss.backward()

                if is_last_accum:
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Recover unscaled loss for logging
            loss_val = loss.item() * self.grad_accum_steps
            total_loss += loss_val
            n_batches += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}")

            if self.use_wandb:
                wandb.log({"train/step_loss": loss_val})

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> float:
        """
        Validate over the full validation set.

        Returns:
            AUC-ROC score (float in [0, 1]).
        """
        self.model.eval()

        all_probs = []
        all_labels = []

        pbar = tqdm(
            self.val_loader,
            desc=f"[Val]   Epoch {epoch + 1}",
            leave=False,
        )
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
            else:
                logits = self.model(images)

            probs = torch.sigmoid(logits)
            all_probs.append(probs)
            all_labels.append(labels)

        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)

        if self.auroc is not None:
            auc = self.auroc(all_probs, all_labels).item()
            self.auroc.reset()
        else:
            # Fallback: scikit-learn (CPU)
            try:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(
                    all_labels.cpu().numpy(),
                    all_probs.cpu().numpy(),
                ))
            except Exception:
                auc = 0.0

        return auc

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        max_steps: Optional[int] = None,
        dry_run: bool = False,
    ) -> None:
        """
        Run training for config['training']['epochs'] epochs.

        Args:
            max_steps: Stop each epoch after this many batches (smoke test).
            dry_run:   Run a single forward pass and exit immediately.
        """
        epochs = self.config["training"].get("epochs", 50)

        if dry_run:
            print("[Dry Run] Running a single forward pass …")
            self.model.train()
            images, labels = next(iter(self.train_loader))
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            print(f"[Dry Run] logits shape: {logits.shape}, loss: {loss.item():.4f}")
            return

        print(
            f"[Train] Device: {self.device} | AMP: {self.use_amp} | "
            f"Epochs: {epochs} | Batches/epoch: {len(self.train_loader)}"
        )

        for epoch in range(self.start_epoch, epochs):
            train_loss = self.train_epoch(epoch, max_steps=max_steps)
            val_auc = self.validate_epoch(epoch)
            self.scheduler.step()

            lr = self.scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:>3}/{epochs} | "
                f"loss: {train_loss:.4f} | "
                f"val_auc: {val_auc:.4f} | "
                f"lr: {lr:.2e}"
            )

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/auc": val_auc,
                    "lr": lr,
                })

            # Track history
            self.start_epoch = epoch
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_auc": val_auc,
                "lr": lr,
            }
            self.history.append(row)
            self._save_history()

            # Save best checkpoint
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.save_checkpoint(tag="best")

            # Save periodic checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(tag=f"epoch{epoch + 1:03d}")

            if max_steps is not None:
                print(f"[Train] max_steps={max_steps} reached — exiting early.")
                break

        print(f"[Train] Finished. Best val AUC: {self.best_auc:.4f}")
        if self.use_wandb:
            wandb.finish()

    def _save_history(self) -> None:
        """Write training history to CSV."""
        path = self.output_dir / "history.csv"
        if not self.history:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)
