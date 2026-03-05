"""
infer.py
--------
Inference entry point — single image or batch / blind-test submission.

Usage (single image):
    python scripts/infer.py \
        --checkpoint checkpoints/model_best.pth \
        --image      path/to/image.jpg

Usage (batch / submission CSV for blind test set):
    python scripts/infer.py \
        --checkpoint  checkpoints/model_best.pth \
        --image_dir   data/trainval_data_final/validation_data_final \
        --output_csv  submissions/submission.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.augmentations import build_val_transform
from src.datasets.base_dataset import UnlabeledDataset
from src.models.hybrid_net import HybridSwinNet
from torch.utils.data import DataLoader
import yaml


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> HybridSwinNet:
    """Load a HybridSwinNet from a checkpoint."""
    model_cfg = config["model"]
    model = HybridSwinNet(
        swinv2_variant=model_cfg["swinv2_variant"],
        pretrained=False,
        freq_embed_dim=model_cfg.get("freq_embed_dim", 512),
        freq_branch_dim=model_cfg.get("freq_branch_dim", 256),
        fmsi_mask_ratio=0.0,           # always off at inference
        fusion_d_model=model_cfg.get("fusion_d_model", 512),
        fusion_heads=model_cfg.get("fusion_heads", 8),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HybridSwinNet Inference")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability threshold for FAKE prediction (default 0.5)")

    # Single image mode
    p.add_argument("--image", default=None, help="Single image to classify")

    # Batch mode
    p.add_argument("--image_dir", default=None, help="Directory of images for batch inference")
    p.add_argument("--output_csv", default="submissions/submission.csv",
                   help="Output CSV path for batch results")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def infer_single(
    image_path: str,
    model: HybridSwinNet,
    transform,
    device: torch.device,
    threshold: float = 0.5,
) -> None:
    """Run inference on a single image and print result."""
    img_np = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    label = "FAKE" if prob >= threshold else "REAL"
    print(f"\nImage : {image_path}")
    print(f"Prob  : {prob:.4f}")
    print(f"Label : {label}  (threshold={threshold})\n")


def infer_batch(
    image_dir: str,
    model: HybridSwinNet,
    transform,
    device: torch.device,
    output_csv: str,
    batch_size: int = 16,
    num_workers: int = 4,
    threshold: float = 0.5,
) -> None:
    """
    Run batch inference over all images in image_dir and write a submission CSV.

    CSV format:
        filename,prob_fake,label
        0000,0.8732,fake
        0001,0.1234,real
        ...
    """
    from tqdm import tqdm

    dataset = UnlabeledDataset(image_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[Infer] {repr(dataset)}")

    rows: list[dict] = []
    with torch.no_grad():
        for images, stems in tqdm(loader, desc="Inferring"):
            images = images.to(device, non_blocking=True)
            probs = torch.sigmoid(model(images)).cpu().numpy().tolist()
            for stem, prob in zip(stems, probs):
                rows.append({
                    "filename": stem,
                    "prob_fake": f"{prob:.6f}",
                    "label": "fake" if prob >= threshold else "real",
                })

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "prob_fake", "label"])
        writer.writeheader()
        writer.writerows(rows)

    n_fake = sum(1 for r in rows if r["label"] == "fake")
    print(f"[Infer] Processed {len(rows)} images — {n_fake} FAKE, {len(rows)-n_fake} REAL")
    print(f"[Infer] Submission CSV → {out_path}")


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = load_model(args.checkpoint, config, device)
    transform = build_val_transform()

    if args.image is not None:
        infer_single(args.image, model, transform, device, args.threshold)
    elif args.image_dir is not None:
        infer_batch(
            args.image_dir, model, transform, device,
            args.output_csv, args.batch_size, args.num_workers, args.threshold,
        )
    else:
        print("[ERROR] Provide either --image or --image_dir.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
