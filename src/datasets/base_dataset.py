"""
base_dataset.py
---------------
PyTorch Dataset for the NTIRE 2026 Deepfake Detection challenge.

Dataset directory layout expected (flat, label-in-filename convention):
  <root>/
    0000_real.png
    0001_fake.png
    ...

Label encoding:
  0 = Real
  1 = Fake
"""

import os
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):
    """
    Reads a flat directory of images whose filenames contain '_real' or '_fake'.
    Optionally applies a transform pipeline (e.g. the Degradation Pipeline).
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        """
        Args:
            root:       Path to directory containing labelled image files.
            transform:  Callable applied to a PIL image → returns tensor.
            extensions: Image file extensions to include.
        """
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for fp in sorted(self.root.iterdir()):
            if fp.suffix.lower() not in extensions:
                continue
            name = fp.stem.lower()
            if "_real" in name:
                label = 0
            elif "_fake" in name:
                label = 1
            else:
                # Skip files that carry no label tag
                continue
            self.samples.append((fp, label))

        if not self.samples:
            raise RuntimeError(
                f"No labelled images found in '{root}'. "
                "Filenames must contain '_real' or '_fake'."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __repr__(self) -> str:
        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        return (
            f"DeepfakeDataset(root='{self.root}', "
            f"total={len(self.samples)}, real={n_real}, fake={n_fake})"
        )
