"""
trainer.py
----------
Training loop for HybridSwinNet — Spec.md §4 & §5 Phase 3.

Features:
  - AdamW optimiser + cosine annealing LR schedule
  - FMSI applied during training forward pass on frequency maps
  - AUC-first evaluation via torchmetrics
  - Best-checkpoint saving by validation AUC
  - Optional wandb logging

TODO (Phase 3 implementation):
  - Implement full train_epoch() loop
  - Implement validate_epoch() loop
  - Wire up wandb logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Lazy imports (installed separately)
try:
    from torchmetrics.classification import BinaryAUROC
except ImportError:
    BinaryAUROC = None

try:
    import wandb
except ImportError:
    wandb = None


class Trainer:
    """
    Manages the training lifecycle for HybridSwinNet.

    Args:
        model:          Instantiated HybridSwinNet.
        train_loader:   DataLoader for training split.
        val_loader:     DataLoader for validation split.
        config:         Dict of hyperparameters (from configs/default.yaml).
        device:         torch.device to train on.
        output_dir:     Directory to save checkpoints.
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

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("epochs", 50),
        )
        self.criterion = nn.BCELoss()

        if BinaryAUROC is not None:
            self.auroc = BinaryAUROC().to(device)
        else:
            self.auroc = None
            print("[WARN] torchmetrics not found — AUC logging disabled.")

        self.best_auc: float = 0.0

    def fit(self) -> None:
        """Run training for config['epochs'] epochs. TODO: implement."""
        # TODO (Phase 3): implement epoch loop calling train_epoch + validate_epoch
        raise NotImplementedError("Training loop — implement in Phase 3.")

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch. Returns average loss. TODO: implement."""
        raise NotImplementedError

    def validate_epoch(self, epoch: int) -> float:
        """Validate and return AUC score. TODO: implement."""
        raise NotImplementedError

    def save_checkpoint(self, tag: str = "best") -> None:
        path = self.output_dir / f"model_{tag}.pth"
        torch.save(self.model.state_dict(), path)
        print(f"[Checkpoint] Saved → {path}")
