"""
base_dataset.py
---------------
PyTorch Datasets for the NTIRE 2026 Deepfake Detection challenge.

Two dataset classes:

1. DeepfakeDataset
   Reads a flat directory whose filenames contain '_real' or '_fake'.
   Supports an internal stratified 80/20 train/val split via
   ``from_split()`` so we can use the labelled training set for
   both training and validation (the official validation_data_final/
   folder has no labels — it is the challenge blind test set).

   Label encoding:  0 = Real,  1 = Fake

2. UnlabeledDataset
   Reads a flat directory of images regardless of filename.
   Returns (tensor, filename_stem) — used for blind test-set inference
   and submission CSV generation.

Directory layout expected for DeepfakeDataset:
    <root>/
        0000_real.png
        0001_fake.png
        ...

Directory layout expected for UnlabeledDataset:
    <root>/
        0000.png
        0001.png
        ...
"""

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to HWC numpy uint8 for Albumentations."""
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Labelled dataset
# ---------------------------------------------------------------------------

class DeepfakeDataset(Dataset):
    """
    Reads a flat directory of images whose filenames contain '_real' or '_fake'.
    Applies an optional Albumentations transform pipeline.

    Args:
        samples:   Pre-built list of (Path, label) pairs (use from_split()).
        transform: Albumentations Compose pipeline (or any callable on numpy HWC).
    """

    _EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if not samples:
            raise RuntimeError("DeepfakeDataset received an empty samples list.")
        self.samples = samples
        self.transform = transform

    # ------------------------------------------------------------------
    # Factory: build from directory (full set)
    # ------------------------------------------------------------------
    @classmethod
    def from_dir(
        cls,
        root: str,
        transform: Optional[Callable] = None,
    ) -> "DeepfakeDataset":
        """
        Load all labelled images from *root*.

        Args:
            root:      Directory containing labelled images.
            transform: Albumentations transform.
        """
        root_path = Path(root)
        samples: List[Tuple[Path, int]] = []
        for fp in sorted(root_path.iterdir()):
            if fp.suffix.lower() not in cls._EXTENSIONS:
                continue
            stem = fp.stem.lower()
            if "_real" in stem:
                label = 0
            elif "_fake" in stem:
                label = 1
            else:
                continue
            samples.append((fp, label))

        if not samples:
            raise RuntimeError(
                f"No labelled images found in '{root}'. "
                "Filenames must contain '_real' or '_fake'."
            )
        return cls(samples, transform)

    # ------------------------------------------------------------------
    # Factory: stratified train/val split
    # ------------------------------------------------------------------
    @classmethod
    def from_split(
        cls,
        root: str,
        val_split: float = 0.2,
        seed: int = 42,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
    ) -> Tuple["DeepfakeDataset", "DeepfakeDataset"]:
        """
        Load all labelled images from *root* and split into stratified
        train / val subsets.

        Args:
            root:            Directory containing labelled images.
            val_split:       Fraction of images to use for validation [0, 1).
            seed:            Random seed for reproducibility.
            train_transform: Transform applied to training samples.
            val_transform:   Transform applied to validation samples.

        Returns:
            (train_dataset, val_dataset)
        """
        full = cls.from_dir(root, transform=None)
        rng = random.Random(seed)

        # Separate indices by class for stratified split
        real_idx = [i for i, (_, l) in enumerate(full.samples) if l == 0]
        fake_idx = [i for i, (_, l) in enumerate(full.samples) if l == 1]
        rng.shuffle(real_idx)
        rng.shuffle(fake_idx)

        n_real_val = max(1, int(len(real_idx) * val_split))
        n_fake_val = max(1, int(len(fake_idx) * val_split))

        val_idx = set(real_idx[:n_real_val] + fake_idx[:n_fake_val])
        train_idx = [i for i in range(len(full.samples)) if i not in val_idx]
        val_idx_list = list(val_idx)

        train_samples = [full.samples[i] for i in train_idx]
        val_samples = [full.samples[i] for i in val_idx_list]

        return (
            cls(train_samples, transform=train_transform),
            cls(val_samples, transform=val_transform),
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image_np = _pil_to_numpy(Image.open(path).convert("RGB"))
        if self.transform is not None:
            transformed = self.transform(image=image_np)
            image = transformed["image"]   # tensor (C, H, W)
        else:
            # Fallback: convert to tensor without normalisation
            image = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
        return image, label

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def class_weights(self) -> torch.Tensor:
        """
        Returns per-sample weights for WeightedRandomSampler so every
        class is sampled equally regardless of dataset imbalance.
        """
        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        total = n_real + n_fake
        w_real = total / (2 * n_real) if n_real > 0 else 1.0
        w_fake = total / (2 * n_fake) if n_fake > 0 else 1.0
        return torch.tensor([w_real if l == 0 else w_fake for _, l in self.samples])

    def __repr__(self) -> str:
        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        return (
            f"DeepfakeDataset(total={len(self.samples)}, "
            f"real={n_real}, fake={n_fake})"
        )


# ---------------------------------------------------------------------------
# Unlabelled dataset (challenge blind test set)
# ---------------------------------------------------------------------------

class UnlabeledDataset(Dataset):
    """
    Reads all images from a flat directory regardless of filename convention.
    Returns (tensor, filename_stem) for submission generation.

    Args:
        root:      Directory of images.
        transform: Albumentations transform (val-style, no augmentation).
    """

    _EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        root_path = Path(root)
        self.samples: List[Path] = sorted(
            fp for fp in root_path.iterdir()
            if fp.suffix.lower() in self._EXTENSIONS
        )
        if not self.samples:
            raise RuntimeError(f"No images found in '{root}'.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.samples[idx]
        image_np = _pil_to_numpy(Image.open(path).convert("RGB"))
        if self.transform is not None:
            transformed = self.transform(image=image_np)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
        return image, path.stem

    def __repr__(self) -> str:
        return f"UnlabeledDataset(total={len(self.samples)})"
