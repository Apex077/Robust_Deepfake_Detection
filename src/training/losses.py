"""
losses.py
---------
Loss functions for binary deepfake classification.

Implements:
  - BCEWithLogitsLoss (default, numerically stable)
  - FocalLoss (for class-imbalanced datasets)

Note: The model outputs raw logits (no sigmoid). Always use these
loss functions directly with logits, never with probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    """
    Numerically stable BCE loss for binary classification.

    Args:
        pos_weight: Weight for positive (fake) class.
                    Set > 1.0 if real/fake ratio is imbalanced.
    """

    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self._pos_weight_val: float = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B,) — raw model outputs (no sigmoid)
            targets: (B,) — integer labels: 0=real, 1=fake

        Returns:
            Scalar loss tensor.
        """
        pw = torch.tensor([self._pos_weight_val], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pw)


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification (Lin et al., 2017).
    Downweights easy negatives so the model focuses on hard examples.

    Args:
        alpha:  Weighting factor for the positive class [0, 1].
        gamma:  Focusing exponent (0 = standard BCE, 2 is typical).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B,) — raw model outputs
            targets: (B,) — integer labels

        Returns:
            Scalar focal loss.
        """
        targets_f = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets_f * probs + (1 - targets_f) * (1 - probs)   # p_t
        alpha_t = targets_f * self.alpha + (1 - targets_f) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()
