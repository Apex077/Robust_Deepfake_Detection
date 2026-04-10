"""
stream_frequency.py
-------------------
Stream B: Frequency Domain — F3-Net style backbone (Spec.md §3.2).

Pipeline:
  1. Convert input RGB tensor to DCT-II frequency map via dct_utils.rgb_to_dct.
  2. Apply FMSI (Frequency Masking) during training to prevent fingerprint overfit.
  3. Split DCT map into Low-Frequency (LF) and High-Frequency (HF) sub-bands.
  4. Pass each sub-band through a ResNet-style branch with residual connections.
  5. Concatenate + project → final frequency embeddings of shape (B, embed_dim).

Reference:
    "Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware
    Clues" (Li et al., ECCV 2020) — F3-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.dct_utils import rgb_to_dct
from src.utils.fmsi import apply_fmsi


# ---------------------------------------------------------------------------
# Building block: Residual Conv Block
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """
    Two-layer residual block:  Conv → BN → ReLU → Conv → BN → (+skip) → ReLU
    If in_channels != out_channels, a 1×1 projection is added for the skip.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x), inplace=True)


# ---------------------------------------------------------------------------
# Single frequency branch (LF or HF)
# ---------------------------------------------------------------------------

class _FreqBranch(nn.Module):
    """
    ResNet-style branch for a single frequency sub-band.

    Architecture: 3×3 Conv → ResBlock (64ch) → ResBlock (128ch) → GAP → Linear
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _ResBlock(64, 64)
        self.layer2 = _ResBlock(64, 128, stride=2)
        self.layer3 = _ResBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)  # (B, 256)
        return self.proj(x)           # (B, out_dim)


# ---------------------------------------------------------------------------
# Main frequency stream
# ---------------------------------------------------------------------------

class StreamFrequency(nn.Module):
    """
    DCT → dual-branch ResNet frequency feature extractor (F3-Net style).

    Input:  (B, 3, 224, 224) normalised RGB tensor
    Output: (B, embed_dim) frequency embedding
    """

    def __init__(
        self,
        branch_dim: int = 256,
        embed_dim: int = 512,
        fmsi_mask_ratio: float = 0.15,
    ) -> None:
        """
        Args:
            branch_dim:       Output dimension of each LF/HF branch.
            embed_dim:        Final projected embedding dimension.
            fmsi_mask_ratio:  Fraction of DCT coefficients to mask during
                              training (FMSI, Spec.md §4.2). Set to 0 to disable.
        """
        super().__init__()
        self.fmsi_mask_ratio = fmsi_mask_ratio
        self.embed_dim = embed_dim

        self.lf_branch = _FreqBranch(in_channels=3, out_dim=branch_dim)
        self.hf_branch = _FreqBranch(in_channels=3, out_dim=branch_dim)

        self.proj = nn.Sequential(
            nn.Linear(branch_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------
    # Frequency band splitting
    # ------------------------------------------------------------------
    @staticmethod
    def _split_bands(
        freq_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split DCT map into Low-Frequency (top-left 1/16th) and
        High-Frequency (bottom-right 9/16th) sub-bands.

        Args:
            freq_map: (B, C, H, W)

        Returns:
            (lf, hf)
        """
        h, w = freq_map.shape[-2:]
        lf_h, lf_w = h // 4, w // 4  # 1/16 area
        lf = freq_map[..., :lf_h, :lf_w]
        hf = freq_map[..., lf_h:, lf_w:]
        return lf, hf

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised RGB tensor
        Returns:
            embeddings: (B, embed_dim) frequency feature vector
        """
        freq_map = rgb_to_dct(x)                        # (B, C, H, W)

        # FMSI: apply only during training to prevent fingerprint overfit
        if self.training and self.fmsi_mask_ratio > 0.0:
            freq_map = apply_fmsi(freq_map, self.fmsi_mask_ratio)

        lf, hf = self._split_bands(freq_map)
        feat_lf = self.lf_branch(lf)                    # (B, branch_dim)
        feat_hf = self.hf_branch(hf)                    # (B, branch_dim)
        feat = torch.cat([feat_lf, feat_hf], dim=1)    # (B, branch_dim*2)
        return self.proj(feat)                          # (B, embed_dim)
