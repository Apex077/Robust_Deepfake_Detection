"""
infer.py
--------
Entry point: single-image inference.

Usage:
    python scripts/infer.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
"""

import argparse
import torch
from PIL import Image
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with HybridSwinNet")
    p.add_argument("--checkpoint", required=True, help="Path to saved model checkpoint")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--threshold", type=float, default=0.5, help="Fake/real threshold")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO (Phase 4): Load model, load checkpoint, preprocess image, run forward pass
    print(f"[Infer] Analysing: {args.image}")
    raise NotImplementedError("Inference entry-point — implement in Phase 4.")


if __name__ == "__main__":
    main()
