"""
evaluate_overfitting.py
-----------------------
Evaluates on the training and validation splits to compare performance
and diagnose underfitting or overfitting.

Usage:
    python scripts/evaluate_overfitting.py \
        --checkpoint checkpoints/model_best.pth \
        --data_dir data/trainval_data_final/training_data_final \
        --output_dir results/overfitting/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.augmentations import build_val_transform
from src.datasets.base_dataset import DeepfakeDataset
from src.models.hybrid_net import HybridSwinNet

try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        roc_auc_score, roc_curve, accuracy_score, precision_score,
        recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
    )
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False


def _evaluate_split(
    split_name: str,
    dataset: DeepfakeDataset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    output_path: Path,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    all_probs: list[float] = []
    all_labels: list[int] = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating {split_name}"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().tolist())

    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels)

    # Calculate ROC curve to find optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    overall_auc = float(roc_auc_score(all_labels_np, all_probs_np))
    all_preds_np = (all_probs_np >= optimal_threshold).astype(int)

    accuracy = float(accuracy_score(all_labels_np, all_preds_np))
    precision = float(precision_score(all_labels_np, all_preds_np, zero_division=0))
    recall = float(recall_score(all_labels_np, all_preds_np, zero_division=0))
    f1 = float(f1_score(all_labels_np, all_preds_np, zero_division=0))

    metrics = {
        "split": split_name,
        "optimal_threshold": float(optimal_threshold),
        "overall_auc": overall_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "n_samples": len(all_labels),
    }

    if _PLOT_AVAILABLE:
        output_path.mkdir(parents=True, exist_ok=True)
        # Confusion Matrix
        cm = confusion_matrix(all_labels_np, all_preds_np)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax_cm, cmap="Blues", values_format="d", colorbar=False)
        ax_cm.set_title(f"Confusion Matrix - {split_name}")
        fig_cm.tight_layout()
        cm_path = output_path / f"confusion_matrix_{split_name.lower()}.png"
        fig_cm.savefig(cm_path, dpi=150)
        plt.close(fig_cm)

        # ROC Curve
        fpr, tpr, _ = roc_curve(all_labels_np, all_probs_np)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {overall_auc:.4f})")
        ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"ROC Curve - {split_name}")
        ax_roc.legend(loc="lower right")
        fig_roc.tight_layout()
        roc_path = output_path / f"roc_curve_{split_name.lower()}.png"
        fig_roc.savefig(roc_path, dpi=150)
        plt.close(fig_roc)

    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Underfitting/Overfitting using dataset splits")
    p.add_argument("--checkpoint", required=True, help="Trained model checkpoint (.pth)")
    p.add_argument("--data_dir", required=True, help="Training data directory containing labelled data")
    p.add_argument("--output_dir", default="results/overfitting/", help="Output directory")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    train_cfg = config["training"]
    transform = build_val_transform()  # same deterministic pipeline for both
    
    train_ds, val_ds, test_ds = DeepfakeDataset.from_split(
        root=args.data_dir,
        val_split=train_cfg.get("val_split", 0.125),
        test_split=train_cfg.get("test_split", 0.125),
        seed=train_cfg.get("val_split_seed", 42),
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
    )
    
    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_cfg = config["model"]
    model = HybridSwinNet(
        swinv2_variant=model_cfg["swinv2_variant"],
        pretrained=False,
        freq_embed_dim=model_cfg.get("freq_embed_dim", 512),
        freq_branch_dim=model_cfg.get("freq_branch_dim", 256),
        fmsi_mask_ratio=0.0,
        fusion_d_model=model_cfg.get("fusion_d_model", 512),
        fusion_heads=model_cfg.get("fusion_heads", 8),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    
    print("\n" + "="*60)
    print("  EVALUATING ON TRAINING SPLIT")
    print("="*60)
    train_metrics = _evaluate_split(
        "Train", train_ds, model, device, args.batch_size, args.num_workers, out_dir
    )

    print("\n" + "="*60)
    print("  EVALUATING ON VALIDATION SPLIT")
    print("="*60)
    val_metrics = _evaluate_split(
        "Validation", val_ds, model, device, args.batch_size, args.num_workers, out_dir
    )

    print("\n" + "="*60)
    print("  EVALUATING ON TEST SPLIT")
    print("="*60)
    test_metrics = _evaluate_split(
        "Test", test_ds, model, device, args.batch_size, args.num_workers, out_dir
    )
    
    print("\n" + "="*60)
    print("  OVERFITTING/UNDERFITTING ANALYSIS")
    print("="*60)
    print(f"{'Metric':<15} | {'Training':<10} | {'Validation':<10} | {'Test':<10} | {'Diff (Train - Val)':<15}")
    print("-" * 60)
    
    keys_to_compare = ["overall_auc", "accuracy", "precision", "recall", "f1_score"]
    
    for key in keys_to_compare:
        train_val = train_metrics.get(key, 0.0)
        val_val = val_metrics.get(key, 0.0)
        test_val = test_metrics.get(key, 0.0)
        diff = train_val - val_val
        print(f"{key.capitalize():<15} | {train_val:<10.4f} | {val_val:<10.4f} | {test_val:<10.4f} | {diff:<15.4f}")
        
    # Save combined json
    out_json = out_dir / "overfitting_analysis.json"
    with open(out_json, "w") as f:
        json.dump({"train": train_metrics, "validation": val_metrics, "test": test_metrics}, f, indent=2)

    print(f"\n[Done] Analysis saved in {args.output_dir}")
