"""
stream_frequency.py
-------------------
Stream B: Frequency Domain — F3-Net style backbone.

Pipeline:
  1. Convert input RGB tensor to DCT frequency map (via dct_utils.rgb_to_dct).
  2. Pass DCT map through a lightweight CNN (High-Frequency & Low-Frequency branches).
  3. Return fused frequency embeddings of shape (B, D).

Reference: "Thinking in Frequency: Face Forgery Detection by Mining
           Frequency-aware Clues" (Li et al., ECCV 2020).
"""

import torch
import torch.nn as nn
from src.utils.dct_utils import rgb_to_dct


class _FreqBranch(nn.Module):
    """Single-branch CNN for processing a frequency sub-band."""

    def __init__(self, in_channels: int = 3, out_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)  # (B, out_channels)


class StreamFrequency(nn.Module):
    """
    DCT → dual-branch CNN frequency feature extractor.

    Input:  (B, 3, 224, 224) normalised RGB tensor
    Output: (B, embed_dim) frequency embedding
    """

    def __init__(self, branch_dim: int = 256, embed_dim: int = 512) -> None:
        super().__init__()
        # Two branches capture different frequency characteristics
        self.lf_branch = _FreqBranch(in_channels=3, out_channels=branch_dim // 2)
        self.hf_branch = _FreqBranch(in_channels=3, out_channels=branch_dim // 2)
        self.proj = nn.Linear(branch_dim, embed_dim)
        self.embed_dim = embed_dim

    def _split_bands(self, freq_map: torch.Tensor):
        """Naive LF/HF split: top-left quadrant vs rest (in frequency space)."""
        h, w = freq_map.shape[-2], freq_map.shape[-1]
        lf = freq_map[..., : h // 2, : w // 2]
        lf = nn.functional.interpolate(lf, size=(h, w), mode="bilinear", align_corners=False)
        hf = freq_map - nn.functional.interpolate(
            freq_map[..., : h // 2, : w // 2], size=(h, w), mode="bilinear", align_corners=False
        )
        return lf, hf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised RGB
        Returns:
            embeddings: (B, embed_dim)
        """
        freq_map = rgb_to_dct(x)             # (B, C, H, W)
        lf, hf = self._split_bands(freq_map)
        feat_lf = self.lf_branch(lf)         # (B, branch_dim//2)
        feat_hf = self.hf_branch(hf)         # (B, branch_dim//2)
        feat = torch.cat([feat_lf, feat_hf], dim=1)  # (B, branch_dim)
        return self.proj(feat)               # (B, embed_dim)
