"""
evaluate.py
-----------
Cross-generator evaluation script — Spec.md §5 Phase 4.

Loads a trained HybridSwinNet checkpoint and measures:
  - Overall AUC-ROC on the evaluation set
  - Per-generator AUC (if generator tag is embedded in filename, e.g. '_midjourney_fake')
  - Robustness at JPEG QF=50 (configurable)
  - ROC curve plot saved as PNG + metrics saved as JSON

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/model_best.pth \
        --data_dir   data/trainval_data_final/training_data_final \
        --output_dir results/ \
        --jpeg_qf    50
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.augmentations import build_val_transform, build_degraded_val_transform
from src.datasets.base_dataset import DeepfakeDataset
from src.models.hybrid_net import HybridSwinNet

try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str = "results/",
    jpeg_qf: int | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
    config_path: str = "configs/default.yaml",
) -> dict:
    """
    Evaluate a trained HybridSwinNet on a labelled test directory.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        data_dir:        Flat image directory with '_real'/'_fake' filenames.
        output_dir:      Where to save ROC plot + metrics JSON.
        jpeg_qf:         If set, apply JPEG compression at this QF before inference.
        batch_size:      Inference batch size.
        num_workers:     DataLoader workers.
        config_path:     YAML config (used to reconstruct model architecture).

    Returns:
        Dict with overall AUC and per-generator AUC scores.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    if jpeg_qf is not None:
        print(f"[Eval] JPEG robustness mode: QF={jpeg_qf}")

    # ------------------------------------------------------------------
    # Load config for model architecture
    # ------------------------------------------------------------------
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]

    # ------------------------------------------------------------------
    # Build transform
    # ------------------------------------------------------------------
    if jpeg_qf is not None:
        transform = build_degraded_val_transform(jpeg_qf=jpeg_qf)
    else:
        transform = build_val_transform()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = DeepfakeDataset.from_dir(data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[Eval] {repr(dataset)}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = HybridSwinNet(
        swinv2_variant=model_cfg["swinv2_variant"],
        pretrained=False,           # weights come from checkpoint
        freq_embed_dim=model_cfg.get("freq_embed_dim", 512),
        freq_branch_dim=model_cfg.get("freq_branch_dim", 256),
        fmsi_mask_ratio=0.0,        # no FMSI at evaluation
        fusion_d_model=model_cfg.get("fusion_d_model", 512),
        fusion_heads=model_cfg.get("fusion_heads", 8),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)  # support both wrapped and raw
    model.load_state_dict(state)
    model.eval()
    print(f"[Eval] Loaded checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    all_probs: list[float] = []
    all_labels: list[int] = []
    all_paths: list[Path] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().tolist())

    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels)

    # ------------------------------------------------------------------
    # Overall AUC
    # ------------------------------------------------------------------
    overall_auc = float(roc_auc_score(all_labels_np, all_probs_np))
    print(f"\n{'='*50}")
    print(f"  Overall AUC-ROC: {overall_auc:.4f}")
    print(f"{'='*50}")

    # ------------------------------------------------------------------
    # Per-generator AUC (parses generator name from filename)
    # E.g. "0001_midjourney_fake.png" → generator = "midjourney"
    # Falls back gracefully if no generator tag is found.
    # ------------------------------------------------------------------
    per_gen: dict[str, dict[str, list]] = defaultdict(lambda: {"probs": [], "labels": []})
    for sample_path, prob, label in zip(
        [p for p, _ in dataset.samples], all_probs, all_labels
    ):
        stem = sample_path.stem.lower()
        parts = stem.split("_")
        # Heuristic: generator tag is any part that is NOT a number,
        # 'real', or 'fake'.
        gen_tags = [
            p for p in parts
            if p not in ("real", "fake") and not p.isdigit()
        ]
        gen = "_".join(gen_tags) if gen_tags else "unknown"
        per_gen[gen]["probs"].append(prob)
        per_gen[gen]["labels"].append(label)

    per_gen_auc: dict[str, float] = {}
    for gen, data in per_gen.items():
        if len(set(data["labels"])) < 2:
            # Need both classes for AUC
            continue
        auc = float(roc_auc_score(data["labels"], data["probs"]))
        per_gen_auc[gen] = auc
        print(f"  Generator [{gen:>20s}]: AUC = {auc:.4f}  (n={len(data['labels'])})")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    tag = f"_qf{jpeg_qf}" if jpeg_qf else ""
    metrics = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "jpeg_qf": jpeg_qf,
        "overall_auc": overall_auc,
        "per_generator_auc": per_gen_auc,
        "n_samples": len(all_labels),
    }
    metrics_path = output_path / f"metrics{tag}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Eval] Metrics saved → {metrics_path}")

    # ------------------------------------------------------------------
    # ROC curve plot
    # ------------------------------------------------------------------
    if _PLOT_AVAILABLE:
        fpr, tpr, _ = roc_curve(all_labels_np, all_probs_np)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, lw=2, color="#4C72B0",
                label=f"HybridSwinNet (AUC = {overall_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        title = f"ROC Curve — NTIRE 2026 Deepfake Detection"
        if jpeg_qf:
            title += f"  [JPEG QF={jpeg_qf}]"
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        roc_path = output_path / f"roc_curve{tag}.png"
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        print(f"[Eval] ROC curve saved  → {roc_path}")
    else:
        print("[WARN] matplotlib/sklearn not available — skipping ROC plot.")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NTIRE 2026 Deepfake Evaluation — Cross-Generator AUC + ROC",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_dir", required=True, help="Labelled test image directory")
    parser.add_argument("--output_dir", default="results/", help="Output directory for plots/metrics")
    parser.add_argument("--jpeg_qf", type=int, default=None,
                        help="Apply JPEG at this QF for robustness evaluation (e.g. 50)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--config", default="configs/default.yaml",
                        help="YAML config to reconstruct model architecture")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        jpeg_qf=args.jpeg_qf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        config_path=args.config,
    )
