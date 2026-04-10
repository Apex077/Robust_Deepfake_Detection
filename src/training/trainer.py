"""
trainer.py
----------
Training loop for HybridSwinNet — Spec.md §4 & §5 Phase 3.

Features:
  - AdamW + cosine-annealing LR schedule
  - Layer-wise LR decay: earlier Swin blocks get lower LR (prevents
    overfitting the pretrained backbone on the tiny dataset)
  - Mixed-precision training (torch.amp.autocast + GradScaler) for GPU speed
  - Gradient accumulation (simulate larger batches on low-VRAM GPUs)
  - WeightedRandomSampler for class-balanced mini-batches
  - Mixup data augmentation (alpha=0.4) — interpolates pairs of samples
    and their labels to prevent memorisation
  - Label-smoothed BCEWithLogitsLoss — prevents over-confidence
  - AUC-first metric logging via torchmetrics.BinaryAUROC
  - Best-checkpoint saving by validation AUC
  - Early stopping (patience-based) to halt before overfitting deepens
  - Optional Weights & Biases (wandb) logging
  - Resume-from-checkpoint support
  - Gradient clipping to prevent training instability
"""

import csv
import os
from pathlib import Path
from typing import Optional

import numpy as np
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
    test_transform=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders from config.

    Uses DeepfakeDataset.from_split() to create a stratified internal
    split from the training_data_final directory.

    Args:
        config:          Full config dict (from configs/default.yaml).
        train_transform: Albumentations transform for training.
        val_transform:   Albumentations transform for validation.
        test_transform:  Albumentations transform for testing.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    train_ds, val_ds, test_ds = DeepfakeDataset.from_split(
        root=data_cfg["train_dir"],
        val_split=train_cfg.get("val_split", 0.125),
        test_split=train_cfg.get("test_split", 0.125),
        seed=train_cfg.get("val_split_seed", 42),
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform if test_transform is not None else val_transform,
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
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"[Dataset] Train: {len(train_ds)} samples | "
        f"Val: {len(val_ds)} samples | "
        f"Test: {len(test_ds)} samples"
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Layer-wise LR decay helpers
# ---------------------------------------------------------------------------

def _build_layer_wise_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_lr_decay: float,
) -> list[dict]:
    """
    Build parameter groups with exponentially decaying LR for Swin backbone.

    Swin V2 has 4 stages (depths 0..3). We assign:
        LR_stage = base_lr × decay^(num_stages - 1 - stage_idx)

    All non-backbone parameters (frequency stream, fusion, head) use base_lr.

    Args:
        model:           HybridSwinNet instance.
        base_lr:         Base learning rate (for the newest layers).
        weight_decay:    Weight decay for AdamW.
        layer_lr_decay:  Multiplicative factor per stage (e.g. 0.75).

    Returns:
        List of dicts suitable for torch.optim.AdamW(param_groups=...).
    """
    # Collect Swin stage parameters
    backbone = getattr(model, "stream_a", None)
    swin = getattr(backbone, "backbone", None) if backbone is not None else None

    if swin is None or not hasattr(swin, "layers"):
        # Fallback: single group with base_lr for everything
        print("[LR] Swin backbone not accessible via model.stream_a.backbone.layers — using flat LR.")
        return [{"params": model.parameters(), "lr": base_lr, "weight_decay": weight_decay}]

    num_stages = len(swin.layers)
    stage_params: list[list] = [[] for _ in range(num_stages)]
    stage_names: list[set[str]] = [set() for _ in range(num_stages)]

    for stage_idx, stage in enumerate(swin.layers):
        for name, param in stage.named_parameters():
            if param.requires_grad:
                stage_params[stage_idx].append(param)
                stage_names[stage_idx].add(name)

    # Collect backbone params that don't belong to any stage (e.g. patch_embed, norm)
    stage_param_ids = {id(p) for group in stage_params for p in group}
    backbone_other_params = [
        p for p in swin.parameters()
        if id(p) not in stage_param_ids and p.requires_grad
    ]

    # Collect all non-backbone params (freq stream, fusion, head)
    backbone_all_ids = {id(p) for p in swin.parameters()}
    other_params = [
        p for p in model.parameters()
        if id(p) not in backbone_all_ids and p.requires_grad
    ]

    param_groups: list[dict] = []

    # Stage groups: stage 0 (earliest) gets lowest LR
    for stage_idx in range(num_stages):
        # Deeper stages (higher idx) get higher LR
        lr_multiplier = layer_lr_decay ** (num_stages - 1 - stage_idx)
        stage_lr = base_lr * lr_multiplier
        if stage_params[stage_idx]:
            param_groups.append({
                "params": stage_params[stage_idx],
                "lr": stage_lr,
                "weight_decay": weight_decay,
                "name": f"swin_stage_{stage_idx}",
            })

    # Backbone non-stage params (patch embedding etc.) — use lowest LR
    if backbone_other_params:
        param_groups.append({
            "params": backbone_other_params,
            "lr": base_lr * (layer_lr_decay ** num_stages),
            "weight_decay": weight_decay,
            "name": "swin_other",
        })

    # Non-backbone params use full base_lr
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "heads_and_freq",
        })

    print("[LR] Layer-wise LR groups:")
    for g in param_groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"  {g.get('name', '?'):25s} | lr={g['lr']:.2e} | params={n_params:,}")

    return param_groups


# ---------------------------------------------------------------------------
# Mixup helper
# ---------------------------------------------------------------------------

def _mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Mixup augmentation (Zhang et al., 2018) to a single mini-batch.

    Randomly draws a mixing coefficient λ ~ Beta(alpha, alpha) and produces:
        images_mix = λ * images + (1-λ) * images[perm]
        labels_mix = λ * labels + (1-λ) * labels[perm]

    The mixed labels are soft floats, compatible with BCEWithLogitsLoss.

    Args:
        images: (B, C, H, W) batch
        labels: (B,) integer labels
        alpha:  Beta distribution concentration parameter (>0). Higher α
                → stronger mixing. Typical: 0.2–0.4.

    Returns:
        (mixed_images, mixed_labels_float)
    """
    lam = float(np.random.beta(alpha, alpha))
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1.0 - lam) * images[perm]
    mixed_labels = lam * labels.float() + (1.0 - lam) * labels[perm].float()
    return mixed_images, mixed_labels


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
        test_loader: Optional[DataLoader] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
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

        # Mixup
        self.mixup_alpha: float = train_cfg.get("mixup_alpha", 0.0)

        # Early stopping
        self.early_stopping_patience: int = train_cfg.get("early_stopping_patience", 0)
        self._patience_counter: int = 0

        # ------------------------------------------------------------------
        # Optimizer with layer-wise LR decay
        # ------------------------------------------------------------------
        base_lr: float = train_cfg.get("lr", 1e-4)
        weight_decay: float = train_cfg.get("weight_decay", 5e-3)
        layer_lr_decay: float = train_cfg.get("layer_lr_decay", 1.0)

        param_groups = _build_layer_wise_param_groups(
            self.model, base_lr, weight_decay, layer_lr_decay
        )
        self.optimizer = torch.optim.AdamW(param_groups)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg.get("epochs", 60),
            eta_min=1e-6,
        )

        # ------------------------------------------------------------------
        # Loss — with label smoothing
        # ------------------------------------------------------------------
        label_smoothing: float = train_cfg.get("label_smoothing", 0.0)
        self.criterion = BCEWithLogitsLoss(
            pos_weight=1.0,
            label_smoothing=label_smoothing,
        )

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
        Train for one epoch.  Applies Mixup when mixup_alpha > 0.

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

            # Apply Mixup augmentation (only during training, when alpha > 0)
            if self.mixup_alpha > 0.0:
                images, labels = _mixup_batch(images, labels, self.mixup_alpha)

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
    def validate_epoch(self, epoch: int, loader: Optional[DataLoader] = None) -> float:
        """
        Validate over the full validation set.

        Returns:
            AUC-ROC score (float in [0, 1]).
        """
        self.model.eval()

        all_probs = []
        all_labels = []

        eval_loader = loader if loader is not None else self.val_loader
        desc = f"[Val]   Epoch {epoch + 1}" if loader is None else "[Test]  Evaluation"

        pbar = tqdm(
            eval_loader,
            desc=desc,
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
        Run training for config['training']['epochs'] epochs, with early
        stopping when val AUC hasn't improved for `early_stopping_patience`
        consecutive epochs.

        Args:
            max_steps: Stop each epoch after this many batches (smoke test).
            dry_run:   Run a single forward pass and exit immediately.
        """
        epochs = self.config["training"].get("epochs", 60)

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
            f"Epochs: {epochs} | Batches/epoch: {len(self.train_loader)} | "
            f"Mixup α: {self.mixup_alpha} | "
            f"Early-stop patience: {self.early_stopping_patience}"
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
            improved = val_auc > self.best_auc
            if improved:
                self.best_auc = val_auc
                self.save_checkpoint(tag="best")
                self._patience_counter = 0
            else:
                self._patience_counter += 1

            # Save periodic checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(tag=f"epoch{epoch + 1:03d}")

            # Early stopping check
            if (
                self.early_stopping_patience > 0
                and self._patience_counter >= self.early_stopping_patience
            ):
                print(
                    f"[EarlyStopping] Val AUC has not improved for "
                    f"{self._patience_counter} epochs. "
                    f"Best AUC: {self.best_auc:.4f}. Stopping."
                )
                break

            if max_steps is not None:
                print(f"[Train] max_steps={max_steps} reached — exiting early.")
                break

        print(f"[Train] Finished. Best val AUC: {self.best_auc:.4f}")
        
        if getattr(self, "test_loader", None) is not None:
            print("[Test] Evaluating best model on held-out test set...")
            best_checkpoint = self.output_dir / "model_best.pth"
            if best_checkpoint.exists():
                self.load_checkpoint(str(best_checkpoint))
            test_auc = self.validate_epoch(-1, loader=self.test_loader)
            print(f"[Test] Final Test AUC: {test_auc:.4f}")
            if self.use_wandb:
                wandb.log({"test/auc": test_auc})
                
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
