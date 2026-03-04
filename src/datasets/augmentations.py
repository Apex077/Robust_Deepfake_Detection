"""
augmentations.py
----------------
The Degradation Pipeline as specified in Spec.md §4.1.

Applies (in order, with random probability):
  1. JPEG compression  — quality factor uniform in [30, 60]
  2. Gaussian blur     — kernel size odd in {3, 5, 7}
  3. Severe downscale  — 224→64→224 (simulates reconstruction artefacts)
  4. ImageNet normalise

Requires: albumentations, torchvision
"""

from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet statistics
_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

IMAGE_SIZE: int = 224
DOWNSCALE_TARGET: int = 64


def build_train_transform() -> A.Compose:
    """
    Returns the hostile augmentation pipeline for training.
    All operations are applied to numpy HWC uint8 images.
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        # 1. JPEG compression
        A.ImageCompression(quality_lower=30, quality_upper=60, p=0.8),
        # 2. Gaussian blur
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        # 3. Severe downscale: 224 → 64 → 224
        A.Downscale(
            scale_min=DOWNSCALE_TARGET / IMAGE_SIZE,
            scale_max=DOWNSCALE_TARGET / IMAGE_SIZE,
            interpolation=0,   # cv2.INTER_NEAREST for downscale
            p=0.5,
        ),
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),   # back to network size
        # 4. Normalise + to tensor
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def build_val_transform() -> A.Compose:
    """
    Minimal transform for validation / inference (no hostile augmentation).
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])
