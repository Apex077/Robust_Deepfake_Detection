"""
dct_utils.py
------------
GPU-compatible Discrete Cosine Transform (DCT) utilities.

Implements a 2D DCT-II on image tensors using torch.fft,
keeping all computation on the same device as the input tensor.

Usage:
    freq_map = rgb_to_dct(image_tensor)   # (B, C, H, W) → (B, C, H, W)
"""

import torch


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D DCT-II of a real-valued tensor via torch.fft.

    Args:
        x: Tensor of shape (..., H, W)

    Returns:
        DCT coefficients of the same shape as x.
    """
    # Mirror trick: DCT-II via FFT
    # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html
    h, w = x.shape[-2], x.shape[-1]

    # Row-wise DCT
    x_row = torch.cat([x, x.flip(-1)], dim=-1)
    X_row = torch.fft.rfft(x_row, dim=-1)[..., :w]
    k_row = torch.arange(w, device=x.device, dtype=x.dtype)
    phase_row = torch.exp(-1j * torch.pi * k_row / (2 * w)).real  # approximation
    X_row = (X_row * torch.complex(phase_row, torch.zeros_like(phase_row))).real

    # Column-wise DCT
    x_col = torch.cat([X_row, X_row.flip(-2)], dim=-2)
    X_col = torch.fft.rfft(x_col, dim=-2)[..., :h, :]
    k_col = torch.arange(h, device=x.device, dtype=x.dtype).unsqueeze(-1)
    phase_col = torch.exp(-1j * torch.pi * k_col / (2 * h)).real
    X = (X_col * torch.complex(phase_col, torch.zeros_like(phase_col))).real

    return X


def rgb_to_dct(images: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of RGB images to their DCT frequency maps.

    Args:
        images: Float tensor of shape (B, C, H, W), values in [0, 1] or normalised.

    Returns:
        DCT frequency map of shape (B, C, H, W) on the same device.
    """
    return dct_2d(images)
