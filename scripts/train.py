"""
train.py
--------
Training entry point for HybridSwinNet — Spec.md §5 Phase 1–3.

Usage:
    # Full training run
    python scripts/train.py --config configs/default.yaml

    # Smoke test (one batch, no checkpoint save)
    python scripts/train.py --config configs/default.yaml --dry_run

    # Limit to N batches per epoch (debugging)
    python scripts/train.py --config configs/default.yaml --max_steps 20

    # Resume from checkpoint
    python scripts/train.py --config configs/default.yaml --resume checkpoints/model_best.pth
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Ensure project root is on PYTHONPATH when run as `python scripts/train.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.augmentations import build_train_transform, build_val_transform
from src.models.hybrid_net import HybridSwinNet
from src.training.trainer import Trainer, build_dataloaders


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HybridSwinNet for deepfake detection")
    p.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file",
    )
    p.add_argument(
        "--max_steps", type=int, default=None,
        help="Stop each epoch after N batches (smoke / debug mode)",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Run a single forward pass and exit (no training)",
    )
    p.add_argument(
        "--resume", default=None,
        help="Path to a checkpoint (.pth) to resume training from",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Train] GPU: {gpu_name}  | CUDA {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("[Train] WARNING: CUDA not available — training on CPU (very slow).")

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    train_transform = build_train_transform()
    val_transform = build_val_transform()

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    test_transform = build_val_transform()
    train_loader, val_loader, test_loader = build_dataloaders(
        config, train_transform, val_transform, test_transform
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = config["model"]
    model = HybridSwinNet(
        swinv2_variant=model_cfg["swinv2_variant"],
        pretrained=model_cfg.get("pretrained", True),
        freq_embed_dim=model_cfg.get("freq_embed_dim", 512),
        freq_branch_dim=model_cfg.get("freq_branch_dim", 256),
        fmsi_mask_ratio=config["training"].get("fmsi_mask_ratio", 0.30),
        fusion_d_model=model_cfg.get("fusion_d_model", 512),
        fusion_heads=model_cfg.get("fusion_heads", 8),
        fusion_dropout=model_cfg.get("fusion_dropout", 0.3),
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] HybridSwinNet — {n_params:.1f} M parameters")
    print(f"[Model] StreamA embed_dim: {model.stream_a.embed_dim}")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    output_dir = config["training"].get("checkpoint_dir", "checkpoints/")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=output_dir,
        test_loader=test_loader,
    )

    # Resume?
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    trainer.fit(max_steps=args.max_steps, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
