"""
fusion.py
---------
Cross-Attention Fusion Block — Spec.md §3.3.

Dynamically weights the reliability of Stream A (spatial) vs. Stream B
(frequency) based on image quality, then produces logits for binary
classification.

Architecture:
  1. Project both streams into shared d_model space.
  2. Cross-attention: spatial (query) attends to frequency (key/value).
  3. Quality-aware gating: softmax gate over spatial + attended features.
  4. LayerNorm + Dropout + Linear head → raw logit (no sigmoid).

The output is a raw logit, not a probability.
Use sigmoid at inference; use BCEWithLogitsLoss during training for
numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion of spatial and frequency embeddings.

    Inputs:
        emb_a: (B, D_a)  — from StreamSpatial  (Swin V2)
        emb_b: (B, D_b)  — from StreamFrequency (F3-Net / DCT)

    Output:
        logit: (B,) — raw (un-sigmoided) fake probability logit
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

        # Project both streams into shared d_model space
        self.proj_a = nn.Linear(d_a, d_model)
        self.proj_b = nn.Linear(d_b, d_model)

        # Cross-attention: spatial (query) attends to frequency (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Quality-aware gate: learns when to trust spatial vs. attended freq.
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1),
        )

        # Final classification head — outputs raw logit
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            emb_a: (B, D_a) — spatial embeddings
            emb_b: (B, D_b) — frequency embeddings

        Returns:
            logit: (B,) — raw fake-probability logit (apply sigmoid for prob)
        """
        # Project to d_model
        a = self.proj_a(emb_a)   # (B, d_model)
        b = self.proj_b(emb_b)   # (B, d_model)

        # Cross-attention: reshape to (B, 1, d_model) for MultiheadAttention
        a_seq = a.unsqueeze(1)
        b_seq = b.unsqueeze(1)
        attended, _ = self.cross_attn(query=a_seq, key=b_seq, value=b_seq)
        attended = attended.squeeze(1)   # (B, d_model)

        # Quality-aware gating: [stream_a, attended_b] → 2 soft weights
        gate_weights = self.gate(torch.cat([a, attended], dim=-1))  # (B, 2)
        fused = gate_weights[:, 0:1] * a + gate_weights[:, 1:2] * attended  # (B, d_model)

        # Classification: raw logit
        logit = self.head(fused).squeeze(-1)   # (B,)
        return logit
