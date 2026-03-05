"""
stream_spatial.py
-----------------
Stream A: Spatial Domain — Swin Transformer V2 (Spec.md §3.1).

Wraps a pretrained Swin V2 backbone via timm and exposes spatial feature
embeddings of shape (B, embed_dim).

Default model: swinv2_base_window8_256.ms_in22k_ft_in1k
  - Pretrained: ImageNet-22k → fine-tuned on ImageNet-1k
  - Window size: 8  (fits 224×224 input cleanly: 224/8=28 windows per side)
  - Setting img_size=224 tells timm to adapt the positional embeddings.

Requires: timm >= 1.0
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "timm is required. Install with: pip install timm>=1.0.12"
    ) from exc


class StreamSpatial(nn.Module):
    """
    Swin Transformer V2 spatial feature extractor.

    Input:  (B, 3, 224, 224) normalised RGB tensor
    Output: (B, embed_dim) spatial embedding
    """

    def __init__(
        self,
        model_name: str = "swinv2_base_window8_256.ms_in1k",
        pretrained: bool = True,
        grad_checkpointing: bool = True,
    ) -> None:
        """
        Args:
            model_name:        timm model identifier.
                               - swinv2_base_window8_256.ms_in1k (IN-1k, native 256px)
                               - swinv2_base_window12to16_192to256.ms_in22k_ft_in1k (22k)
            pretrained:        Download pretrained weights.
            grad_checkpointing: Recompute activations during backward to save VRAM.
                               ~30% slower but cuts activation memory by ~50%.
                               Essential for Swin-B on ≤6 GB VRAM.

        Note:
            Native input resolution is 256×256 (window8 divides evenly into 256/8=32).
            The augmentation pipeline must resize images to 256 accordingly.
        """
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,    # remove classification head → raw features
            global_pool="avg",
        )
        # timm exposes num_features as the backbone's output channels
        self.embed_dim: int = self.backbone.num_features

        # Enable gradient checkpointing to reduce activation memory footprint
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(enable=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised RGB tensor
        Returns:
            embeddings: (B, embed_dim)
        """
        return self.backbone(x)
