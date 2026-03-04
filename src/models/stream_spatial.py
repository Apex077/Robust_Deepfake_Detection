"""
stream_spatial.py
-----------------
Stream A: Spatial Domain — Swin Transformer V2.

Wraps a pretrained Swin V2 backbone (via timm) and exposes
spatial feature embeddings of shape (B, D).

Requires: timm >= 0.9
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for StreamSpatial. Install with: pip install timm"
    ) from e


class StreamSpatial(nn.Module):
    """
    Swin Transformer V2 spatial feature extractor.

    Input:  (B, 3, 224, 224) normalised RGB tensor
    Output: (B, embed_dim) spatial embedding
    """

    def __init__(
        self,
        model_name: str = "swinv2_base_window12_192_22k",
        pretrained: bool = True,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,   # remove classification head → raw features
            global_pool="avg",
        )
        self.embed_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            embeddings: (B, embed_dim)
        """
        return self.backbone(x)
