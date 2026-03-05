"""
fmsi.py
-------
Frequency Masking Spectrum Inversion (FMSI) — Spec.md §4.2.

During training, randomly zeroes out a fraction of DCT frequency
coefficients to prevent the model overfitting to specific generator
fingerprints from known datasets (FaceForensics++, Celeb-DF, etc.).

Fully vectorised — no Python loops — for GPU efficiency.
"""

import torch


def apply_fmsi(freq_map: torch.Tensor, mask_ratio: float = 0.15) -> torch.Tensor:
    """
    Randomly mask (zero-out) ``mask_ratio`` fraction of DCT coefficients.

    Operates without any Python-level loops: builds a per-sample boolean
    mask in a single vectorised operation on-device.

    Args:
        freq_map:   Float tensor of shape (B, C, H, W) — DCT frequency maps.
        mask_ratio: Fraction of spatial positions (H*W) to zero out per sample.
                    Default 0.15 (15 %).

    Returns:
        Masked frequency map of the same shape.
        The input tensor is NOT modified in-place; a new tensor is returned.

    Note:
        Should only be called when ``model.training`` is True.
    """
    if mask_ratio <= 0.0:
        return freq_map

    b, c, h, w = freq_map.shape
    n_coeffs = h * w
    n_mask = int(n_coeffs * mask_ratio)

    if n_mask == 0:
        return freq_map

    # ---------------------------------------------------------------
    # Vectorised mask: shape (B, n_coeffs)
    # torch.rand-based approach: take the n_mask smallest values as mask
    # This is equivalent to drawing without replacement for each sample.
    # ---------------------------------------------------------------
    noise = torch.rand(b, n_coeffs, device=freq_map.device, dtype=freq_map.dtype)
    # Positions to mask = the n_mask smallest random values
    threshold = noise.kthvalue(n_mask, dim=1).values.unsqueeze(1)  # (B, 1)
    mask = (noise > threshold).float()   # 1 = keep, 0 = masked

    # Reshape mask to (B, 1, H, W) → broadcast over channels
    mask = mask.view(b, 1, h, w)

    return freq_map * mask
