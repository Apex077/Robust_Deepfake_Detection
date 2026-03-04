"""
losses.py
---------
Loss functions for binary deepfake classification.

Currently implements:
  - BinaryCrossEntropyLoss (default)
  - Placeholder for AUC-maximisation loss (LibAUC integration point)
"""

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """Wrapper around BCELoss for clean API."""

    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        weight = torch.tensor([pos_weight])
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets.float())
