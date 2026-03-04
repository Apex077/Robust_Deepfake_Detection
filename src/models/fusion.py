"""
fusion.py
---------
Cross-Attention Fusion Block — Spec.md §3.3.

Dynamically weights the reliability of Stream A (spatial) vs. Stream B (frequency)
based on image quality, then produces a fused classification output.

Architecture:
  - Project both streams to a common d_model
  - Spatial features attend to frequency features (MultiheadAttention)
  - Quality-aware gating: softmax gate over both streams
  - Linear head → sigmoid probability of fake
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion of spatial and frequency embeddings.

    Input:
        emb_a: (B, D_a)  — from StreamSpatial
        emb_b: (B, D_b)  — from StreamFrequency
    Output:
        prob_fake: (B,)  — probability in [0, 1]
    """

    def __init__(
        self,
        d_a: int = 1024,
        d_b: int = 512,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Project both streams into shared space
        self.proj_a = nn.Linear(d_a, d_model)
        self.proj_b = nn.Linear(d_b, d_model)

        # Cross-attention: spatial (query) attends to frequency (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Quality-aware gating: produces scalar weights for each stream
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, 2),
            nn.Softmax(dim=-1),
        )

        # Final classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb_a: (B, D_a)
            emb_b: (B, D_b)
        Returns:
            prob_fake: (B,) — sigmoid probabilities
        """
        # Project to d_model
        a = self.proj_a(emb_a)   # (B, d_model)
        b = self.proj_b(emb_b)   # (B, d_model)

        # Cross-attention: reshape to (B, 1, d_model) for MHA
        a_seq = a.unsqueeze(1)
        b_seq = b.unsqueeze(1)
        attended, _ = self.cross_attn(query=a_seq, key=b_seq, value=b_seq)
        attended = attended.squeeze(1)   # (B, d_model)

        # Quality-aware gating
        gate_weights = self.gate(torch.cat([a, b], dim=-1))  # (B, 2)
        fused = gate_weights[:, 0:1] * a + gate_weights[:, 1:2] * attended  # (B, d_model)

        # Classification
        logits = self.head(fused).squeeze(-1)    # (B,)
        return torch.sigmoid(logits)
