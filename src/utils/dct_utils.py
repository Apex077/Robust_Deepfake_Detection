"""
dct_utils.py
------------
GPU-compatible Discrete Cosine Transform (DCT-II) utilities.

Implements a correct 2D DCT-II on image tensors using torch.fft,
keeping all computation on the same device as the input tensor.

The DCT-II is computed via the FFT mirror trick (ref: Makhoul 1980):
  - Extend x by mirroring: [x, flip(x)]
  - Take FFT, keep first N coefficients
  - Multiply by complex phase: exp(-j * π * k / 2N)
  - Take the real part

This is equivalent to scipy.fft.dct(x, type=2, norm=None).

Usage:
    freq_map = rgb_to_dct(image_tensor)   # (B, C, H, W) → (B, C, H, W)
"""

import math
import torch


def dct_1d(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the 1D DCT-II of a real-valued tensor along `dim`.

    Uses the FFT mirror trick:
        DCT-II[k] = Re( FFT([x, flip(x)])[k] * exp(-j * π * k / (2*N)) )

    Args:
        x:   Real tensor of arbitrary shape.
        dim: Dimension along which to compute the DCT (default: last dim).

    Returns:
        DCT-II coefficients of the same shape as x.
    """
    n = x.shape[dim]

    # Mirror: [x, flip(x)] along `dim` → length 2N
    x_mirror = torch.cat([x, x.flip(dim)], dim=dim)

    # Real FFT → shape has size N+1 along `dim`, complex dtype
    X = torch.fft.rfft(x_mirror, dim=dim)

    # Keep only first N components
    # rfft output has length (2N//2 + 1) = N+1, slice to N
    X = torch.narrow(X, dim, 0, n)

    # Phase factor: exp(-j * π * k / (2N)), broadcast over all other dims
    k = torch.arange(n, dtype=x.dtype, device=x.device)
    # Build shape for broadcasting
    shape = [1] * x.dim()
    shape[dim] = n
    k = k.view(shape)

    phase = torch.exp(
        torch.complex(
            torch.zeros_like(k),
            -torch.pi * k / (2 * n),        # imaginary part: -π*k/(2N)
        )
    )

    # Scale appropriately to prevent FP16 overflow during AMP training.
    # Without this, the 2D DC component of a 256x256 bright image reaches 65536,
    # which overflows the float16 maximum value (65504).
    scale = math.sqrt(2 * n)
    return (X * phase).real / scale


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D DCT-II of a real-valued tensor.
    Applies separable 1D DCT along the last two dimensions (H, W).

    Args:
        x: Tensor of shape (..., H, W).

    Returns:
        DCT-II coefficients of the same shape as x.
    """
    return dct_1d(dct_1d(x, dim=-1), dim=-2)


def rgb_to_dct(images: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of RGB images to their 2D DCT-II frequency maps.

    Args:
        images: Float tensor of shape (B, C, H, W) on any device.
                Values may be normalised (e.g. ImageNet-normalised) or in [0,1].

    Returns:
        DCT frequency map of the same shape (B, C, H, W) on the same device.
    """
    return dct_2d(images)
