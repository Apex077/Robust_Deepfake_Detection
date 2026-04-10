"""
hybrid_net.py
-------------
Top-level HybridSwinNet model — Spec.md §2 (System Architecture).

Composes:
  StreamSpatial   (Swin V2 — RGB spatial domain)
  StreamFrequency (F3-Net / DCT + ResNet CNN — frequency domain)
  CrossAttentionFusion

Single .forward(x) call returns raw logits (un-sigmoided).
Use .predict(x) to get calibrated [0, 1] fake probabilities.
"""

import torch
import torch.nn as nn

from src.models.stream_spatial import StreamSpatial
from src.models.stream_frequency import StreamFrequency
from src.models.fusion import CrossAttentionFusion


class HybridSwinNet(nn.Module):
    """
    Multi-modal deepfake detector (Spec.md §2).

    Input:  (B, 3, 224, 224) normalised RGB tensor
    Output: (B,) raw logits  ← use BCEWithLogitsLoss during training
            or call .predict(x) for [0, 1] probabilities at inference.
    """

    def __init__(
        self,
        swinv2_variant: str = "swinv2_tiny_window8_256.ms_in1k",
        pretrained: bool = True,
        freq_branch_dim: int = 256,
        freq_embed_dim: int = 512,
        fmsi_mask_ratio: float = 0.15,
        fusion_d_model: int = 512,
        fusion_heads: int = 8,
        fusion_dropout: float = 0.1,
    ) -> None:
        """
        Args:
            swinv2_variant:   timm model name for Swin V2.
            pretrained:       Load ImageNet-22k pretrained weights.
            img_size:         Input spatial resolution (224).
            freq_branch_dim:  Output dim of each LF/HF frequency branch.
            freq_embed_dim:   Final frequency embedding dimension.
            fmsi_mask_ratio:  DCT coefficient mask ratio during training.
            fusion_d_model:   Cross-attention hidden size.
            fusion_heads:     Number of attention heads.
            fusion_dropout:   Dropout in fusion block and classification head.
        """
        super().__init__()

        # --- Stream A: Spatial (Swin V2, native 256×256 input) ---
        self.stream_a = StreamSpatial(
            model_name=swinv2_variant,
            pretrained=pretrained,
        )

        # --- Stream B: Frequency (F3-Net / DCT + ResNet) ---
        self.stream_b = StreamFrequency(
            branch_dim=freq_branch_dim,
            embed_dim=freq_embed_dim,
            fmsi_mask_ratio=fmsi_mask_ratio,
        )

        # --- Fusion + Classification Head ---
        self.fusion = CrossAttentionFusion(
            d_a=self.stream_a.embed_dim,    # auto-detected from Swin backbone
            d_b=self.stream_b.embed_dim,    # = freq_embed_dim
            d_model=fusion_d_model,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parallel dual-stream forward pass.

        Args:
            x: (B, 3, H, W) normalised RGB tensor

        Returns:
            logits: (B,) — raw logits (no sigmoid)
        """
        emb_a = self.stream_a(x)       # (B, swin_embed_dim)
        emb_b = self.stream_b(x)       # (B, freq_embed_dim)
        return self.fusion(emb_a, emb_b)  # (B,) logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference helper: returns sigmoid probabilities in [0, 1].
        Calls eval mode; does not affect training mode state.

        Args:
            x: (B, 3, H, W) normalised RGB tensor

        Returns:
            probs: (B,) — probability of image being fake
        """
        was_training = self.training
        self.eval()
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        if was_training:
            self.train()
        return probs
