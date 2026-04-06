"""
losses.py
---------
Loss functions for binary deepfake classification.

Implements:
  - BCEWithLogitsLoss (default, numerically stable) — with optional label smoothing
  - FocalLoss (for class-imbalanced datasets)

Note: The model outputs raw logits (no sigmoid). Always use these
loss functions directly with logits, never with probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    """
    Numerically stable BCE loss for binary classification with optional
    label smoothing to prevent the model from becoming over-confident on
    the small training set.

    Label smoothing converts hard 0/1 targets to:
        y_smooth = y * (1 - ε) + 0.5 * ε
    so the model is never penalised for assigning a small probability to
    the wrong class, acting as an implicit regulariser.

    Args:
        pos_weight:      Weight for positive (fake) class. Set > 1.0 if
                         real/fake ratio is imbalanced.
        label_smoothing: Smoothing factor ε ∈ [0, 1). 0.0 disables smoothing.
                         Typical: 0.05–0.15.
    """

    def __init__(self, pos_weight: float = 1.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self._pos_weight_val: float = pos_weight
        self.label_smoothing: float = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B,) — raw model outputs (no sigmoid)
            targets: (B,) — integer or float labels: 0=real, 1=fake

        Returns:
            Scalar loss tensor.
        """
        targets_f = targets.float()

        # Apply label smoothing: pull hard 0/1 targets toward 0.5
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            targets_f = targets_f * (1.0 - eps) + 0.5 * eps

        pw = torch.tensor([self._pos_weight_val], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets_f, pos_weight=pw)


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
