"""
train.py
--------
Entry point: training.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --max_steps 50 --dry_run
"""

import argparse
import yaml
import torch
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HybridSwinNet for deepfake detection")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--max_steps", type=int, default=None, help="Stop after N steps (smoke test)")
    p.add_argument("--dry_run", action="store_true", help="Single forward pass only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # TODO (Phase 3): Instantiate model, dataloaders, trainer and call trainer.fit()
    raise NotImplementedError(
        "Training entry-point — wire up Trainer in Phase 3. "
        "See src/training/trainer.py for the Trainer class."
    )


if __name__ == "__main__":
    main()
