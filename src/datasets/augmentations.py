"""
augmentations.py
----------------
The Degradation Pipeline as specified in Spec.md §4.1, with additional
augmentations to combat overfitting on the small (~800 sample) training set.

Applies (stochastically) for training:
  1. JPEG compression   — quality factor uniform in [30, 60]
  2. Gaussian blur      — kernel size odd in {3..7}
  3. Severe downscale   — 224 → 64 → 224 (simulates reconstruction artefacts)
  4. Horizontal flip    — standard geometric augmentation
  5. Random 90° rotate  — adds rotational variety
  6. Color jitter       — brightness / contrast / saturation variation (stronger)
  7. Gaussian noise     — low-level intensity noise
  8. Grid distortion    — mild geometric warp (breaks spatial fingerprints)
  9. Coarse dropout     — random patch erasure (CutOut-style)
  10. ImageNet normalise + to-tensor

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
    Returns the hostile augmentation pipeline for training (Spec.md §4.1)
    extended with extra regularisation augmentations to reduce overfitting.
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

        # 90° random rotations — increases rotational variety cheaply
        A.RandomRotate90(p=0.3),

        # Stronger colour jitter (p and magnitudes both bumped up)
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.08,
            p=0.6,
        ),

        # ---- Extra anti-overfitting augmentations -----------------------
        # Additive Gaussian noise — std_range is in [0,1] relative to pixel range
        # std_range=(0.015, 0.04) ≈ var_limit=(10, 50) on uint8 images / 255
        A.GaussNoise(std_range=(0.015, 0.04), p=0.3),


        # Mild grid distortion — breaks spatial fingerprints without destroying structure
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),

        # Random erasing (CutOut) — drops up to 8 patches of ~12%×12% image size
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.05, 0.12),   # fraction of image height
            hole_width_range=(0.05, 0.12),    # fraction of image width
            fill=0,
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
