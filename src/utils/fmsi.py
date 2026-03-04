"""
fmsi.py
-------
Frequency Masking Spectrum Inversion (FMSI) — Spec.md §4.2.

During training, randomly zeroes out a fraction of frequency coefficients
in the DCT map to prevent overfitting to specific generator fingerprints.
"""

import torch


def apply_fmsi(freq_map: torch.Tensor, mask_ratio: float = 0.15) -> torch.Tensor:
    """
    Randomly mask (zero-out) `mask_ratio` fraction of DCT coefficients.

    Args:
        freq_map:   Float tensor of shape (B, C, H, W) — DCT frequency maps.
        mask_ratio: Fraction of coefficients to zero out, default 0.15.

    Returns:
        Masked frequency map of the same shape (in-place modification).

    Note:
        Should only be called during the training forward pass, not at eval time.
    """
    if mask_ratio <= 0.0:
        return freq_map

    b, c, h, w = freq_map.shape
    n_coeffs = h * w
    n_mask = int(n_coeffs * mask_ratio)

    # Build a random mask per sample (same mask across channels for efficiency)
    # Shape: (B, 1, H*W) — broadcast over C
    flat = freq_map.view(b, c, n_coeffs)
    for i in range(b):
        indices = torch.randperm(n_coeffs, device=freq_map.device)[:n_mask]
        flat[i, :, indices] = 0.0

    return flat.view(b, c, h, w)
