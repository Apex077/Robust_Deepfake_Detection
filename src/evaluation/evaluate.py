"""
evaluate.py
-----------
Cross-generator evaluation script — Spec.md §5 Phase 4.

Loads a trained HybridSwinNet checkpoint and measures:
  - Per-generator AUC
  - Overall ROC curve (saved as PNG)
  - Optional JPEG QF=50 robustness evaluation

TODO (Phase 4 implementation):
  - Implement run_evaluation() with actual model loading
  - Implement per-generator partitioning
  - Implement ROC curve plotting via matplotlib
"""

import argparse
from pathlib import Path

import torch

# Lazy imports
try:
    from torchmetrics.classification import BinaryAUROC
except ImportError:
    BinaryAUROC = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def run_evaluation(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str = "results/",
    jpeg_qf: int = None,
) -> None:
    """
    Evaluate a trained model checkpoint on a test dataset.

    Args:
        checkpoint_path: Path to saved .pth checkpoint.
        data_dir:        Root of test images (flat, label-in-filename).
        output_dir:      Where to save ROC curves and metrics.
        jpeg_qf:         If set, apply JPEG compression at this quality before inference.

    TODO: Implement in Phase 4.
    """
    raise NotImplementedError("Evaluation — implement in Phase 4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NTIRE 2026 Deepfake Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_dir", required=True, help="Test dataset directory")
    parser.add_argument("--output_dir", default="results/", help="Output directory for plots")
    parser.add_argument("--jpeg_qf", type=int, default=None, help="JPEG quality factor for robustness test")
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.data_dir, args.output_dir, args.jpeg_qf)
