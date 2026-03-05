"""
augmentations.py
----------------
The Degradation Pipeline as specified in Spec.md §4.1.

Applies (stochastically) for training:
  1. JPEG compression   — quality factor uniform in [30, 60]
  2. Gaussian blur      — kernel size odd in {3..7}
  3. Severe downscale   — 224 → 64 → 224 (simulates reconstruction artefacts)
  4. Horizontal flip    — standard geometric augmentation
  5. Color jitter       — brightness / contrast / saturation variation
  6. ImageNet normalise + to-tensor

For validation / inference — only resize + normalise.

Requires: albumentations >= 2.0  (API changed in 2.0: Downscale now uses
  `scale_range` and `interpolation_pair` instead of `scale_min/max/interpolation`)
"""

from typing import Tuple
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet statistics
_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

IMAGE_SIZE: int = 256       # Swin V2 window8_256 native input resolution
DOWNSCALE_TARGET: int = 64
_SCALE: float = DOWNSCALE_TARGET / IMAGE_SIZE  # 64/256 = 0.25


def build_train_transform() -> A.Compose:
    """
    Returns the hostile augmentation pipeline for training (Spec.md §4.1).
    Expects numpy HWC uint8 images as input.
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),

        # ---- Degradation pipeline (§4.1) --------------------------------
        # 1. JPEG compression: QF in [30, 60]
        A.ImageCompression(
            quality_range=(30, 60),
            compression_type="jpeg",
            p=0.8,
        ),

        # 2. Gaussian blur (kernel sizes 3, 5, 7)
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),

        # 3. Severe downscale: 224 → 64 → 224
        #    albumentations ≥ 2.0: use scale_range + interpolation_pair
        A.Downscale(
            scale_range=(_SCALE, _SCALE),
            interpolation_pair={
                "downscale": cv2.INTER_NEAREST,
                "upscale": cv2.INTER_LINEAR,
            },
            p=0.5,
        ),
        # Restore to network input size after possible Downscale resize
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),

        # ---- Standard augmentations for generalization ------------------
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=0.3,
        ),

        # ---- Normalise + to tensor --------------------------------------
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def build_val_transform() -> A.Compose:
    """
    Minimal transform for validation/inference — no hostile augmentation.
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def build_degraded_val_transform(jpeg_qf: int = 50) -> A.Compose:
    """
    Validation transform with fixed-QF JPEG degradation for robustness
    evaluation (Spec.md §5 Phase 4 / evaluation.jpeg_robustness_qf).

    Args:
        jpeg_qf: JPEG quality factor to apply (e.g. 50).
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.ImageCompression(
            quality_range=(jpeg_qf, jpeg_qf),
            compression_type="jpeg",
            p=1.0,
        ),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])
