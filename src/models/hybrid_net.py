"""
hybrid_net.py
-------------
Top-level HybridSwinNet model — Spec.md §2 (System Architecture).

Composes:
  StreamSpatial  (Swin V2)
  StreamFrequency (F3-Net / DCT + CNN)
  CrossAttentionFusion

Single .forward(x) call returns a per-sample fake probability in [0, 1].
"""

import torch
import torch.nn as nn

from src.models.stream_spatial import StreamSpatial
from src.models.stream_frequency import StreamFrequency
from src.models.fusion import CrossAttentionFusion


class HybridSwinNet(nn.Module):
    """
    Multi-modal deepfake detector.

    Input:  (B, 3, 224, 224) normalised RGB tensor
    Output: (B,) probability of being fake
    """

    def __init__(
        self,
        swinv2_variant: str = "swinv2_base_window12_192_22k",
        pretrained: bool = True,
        freq_embed_dim: int = 512,
        fusion_d_model: int = 512,
        fusion_heads: int = 8,
    ) -> None:
        super().__init__()
        self.stream_a = StreamSpatial(
            model_name=swinv2_variant,
            pretrained=pretrained,
        )
        self.stream_b = StreamFrequency(embed_dim=freq_embed_dim)
        self.fusion = CrossAttentionFusion(
            d_a=self.stream_a.embed_dim,
            d_b=freq_embed_dim,
            d_model=fusion_d_model,
            num_heads=fusion_heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            prob_fake: (B,)
        """
        emb_a = self.stream_a(x)
        emb_b = self.stream_b(x)
        return self.fusion(emb_a, emb_b)
