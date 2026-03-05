"""
evaluate.py
-----------
CLI entry point for evaluation — wraps src/evaluation/evaluate.py.

Usage:
    # Standard evaluation
    python scripts/evaluate.py \
        --checkpoint checkpoints/model_best.pth \
        --data_dir   data/trainval_data_final/training_data_final \
        --output_dir results/

    # JPEG robustness evaluation (QF=50 as in Spec.md §5 Phase 4)
    python scripts/evaluate.py \
        --checkpoint checkpoints/model_best.pth \
        --data_dir   data/trainval_data_final/training_data_final \
        --output_dir results/ \
        --jpeg_qf    50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluate import run_evaluation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NTIRE 2026 Deepfake Evaluation — AUC + ROC Curves")
    p.add_argument("--checkpoint", required=True, help="Trained model checkpoint (.pth)")
    p.add_argument("--data_dir", required=True, help="Flat image directory with _real/_fake filenames")
    p.add_argument("--output_dir", default="results/", help="Where to save plots and metrics JSON")
    p.add_argument("--jpeg_qf", type=int, default=None,
                   help="JPEG quality factor for robustness evaluation (e.g. 50)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        jpeg_qf=args.jpeg_qf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        config_path=args.config,
    )
