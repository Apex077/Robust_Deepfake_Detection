# Robust Deepfake Detection System — NTIRE 2026

A multi-modal deep learning system for detecting AI-generated images "in-the-wild",
targeting the **NTIRE 2026 Deepfake Detection Challenge**.

**Primary Metric:** AUC (Area Under ROC Curve) — Target ≥ 85–90% on unseen generators.

---

## Architecture Overview: Hybrid-Swin Network

The system processes each image through **two parallel streams** then fuses them:

```
Input Image (3×224×224)
       │
       ├──── Stream A: Spatial (Swin V2) ──── Spatial Embeddings (B, 1024)
       │                                                                    │
       │                                                   Cross-Attention Fusion → prob_fake
       │                                                                    │
       └──── Stream B: Frequency (DCT + F3-Net) ── Freq Embeddings (B, 512)
```

| Stream | Backbone | Input | Purpose |
|--------|----------|-------|---------|
| **A — Spatial** | Swin Transformer V2 | RGB `3×224×224` | Visual artifacts, structural dependencies |
| **B — Frequency** | F3-Net style CNN | DCT map `3×224×224` | Checkerboard patterns, upsampling fingerprints |
| **Fusion** | Cross-Attention | Both embeddings | Quality-aware dynamic weighting → binary output |

---

## Project Structure

```
IVP_Project/
├── src/
│   ├── datasets/
│   │   ├── base_dataset.py      # DeepfakeDataset (reads _real/_fake filenames)
│   │   └── augmentations.py     # Degradation Pipeline (JPEG, Blur, Downscale)
│   ├── models/
│   │   ├── stream_spatial.py    # Stream A: Swin V2 wrapper
│   │   ├── stream_frequency.py  # Stream B: DCT + dual-branch CNN
│   │   ├── fusion.py            # Cross-Attention Fusion Block
│   │   └── hybrid_net.py        # Top-level HybridSwinNet
│   ├── training/
│   │   ├── trainer.py           # Training loop (AdamW, cosine LR, AUC logging)
│   │   └── losses.py            # BCE loss (LibAUC integration point)
│   ├── evaluation/
│   │   └── evaluate.py          # ROC curves, per-generator AUC, JPEG robustness
│   └── utils/
│       ├── dct_utils.py         # GPU DCT-II via torch.fft
│       └── fmsi.py              # Frequency Masking Spectrum Inversion
├── scripts/
│   ├── train.py                 # Entry point: training
│   └── infer.py                 # Entry point: inference
├── configs/
│   └── default.yaml             # All hyperparameters
├── trainval_data_final/         # Challenge dataset (images excluded from git)
│   ├── training_data_final/
│   └── validation_data_final/
├── data/test/                   # Test split (populate when released)
├── checkpoints/                 # Saved model weights (excluded from git)
├── results/                     # ROC plots, metric tables (excluded from git)
├── dummy_submission/
│   └── submission.txt           # Example submission format
├── requirements.txt
├── environment.yml
├── Spec.md                      # Original PRD
└── .gitignore
```

---

## Dataset

### Format
Images follow a **flat directory, label-in-filename** convention:
```
training_data_final/
  0000_real.png
  0001_fake.png
  0002_real.png
  ...
```
- `_real` → label `0` (authentic)
- `_fake` → label `1` (AI-generated)

### Layout
| Split | Directory | Count |
|-------|-----------|-------|
| Train | `trainval_data_final/training_data_final/` | ~1000 images |
| Val   | `trainval_data_final/validation_data_final/` | ~100 images |
| Test  | `data/test/` | TBD (released by challenge) |

> **Note:** Dataset images are excluded from git via `.gitignore`.
> Download them from the competition and place them in the respective directories.
> The `.gitkeep` files in each folder preserve the directory structure.

### Submission Format
Predictions are submitted as a plain-text file with one probability per line (see `dummy_submission/submission.txt`):
```
0.1    ← Real (low fake probability)
0.9    ← Fake (high fake probability)
...
```

---

## Implementation Plan

### Phase 1 — Pipeline Setup *(Status: Scaffolded)*
- [x] Project structure & package layout
- [x] `DeepfakeDataset` — reads flat `_real`/`_fake` directories
- [x] Degradation Pipeline — JPEG QF 30–60, Gaussian blur, 224→64→224 downscale
- [ ] Wire DataLoaders in `scripts/train.py`

### Phase 2 — Stream Implementation *(Status: Scaffolded)*
- [x] `dct_utils.py` — on-GPU DCT-II via `torch.fft`
- [x] `StreamSpatial` — Swin V2 via `timm`, pretrained, features only
- [x] `StreamFrequency` — DCT → dual-branch CNN (LF + HF)
- [ ] Validate output shapes with unit tests

### Phase 3 — Fusion & Training Loop *(Status: Completed)*Do 
- [x] `CrossAttentionFusion` — quality-gated cross-attention + classification head
- [x] `HybridSwinNet` — top-level model composition
- [x] `Trainer` implementation with AdamW, cosine LR, and Mixup augmentation
- [x] Implement layer-wise LR decay and early stopping
- [x] Apply FMSI in training forward pass
- [ ] Integrate `wandb` logging

### Phase 4 — Evaluation *(Status: Completed)*
- [x] `evaluate.py` CLI
- [x] Optimal threshold calibration via Youden's J statistic
- [x] ROC curve and Confusion Matrix plots
- [ ] Per-generator AUC table
- [ ] Inference entry-point (`scripts/infer.py`)

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Core framework | PyTorch ≥ 2.2 |
| Spatial backbone | Swin V2 via `timm` ≥ 0.9 |
| Frequency transform | `torch.fft` (on-GPU, no extra deps) |
| Augmentation | `albumentations` |
| Metrics | `torchmetrics` (AUROC) |
| Experiment tracking | `wandb` (optional) |
| Config management | `PyYAML` |

**GPU Target:** `sm_86` minimum (RTX 3090, 24 GB VRAM) for local development.
Cloud: Azure `NC A100 v4` (`sm_80`, A100 80 GB).

---

## Quick Start

### 1. Set up environment

```bash
# Option A: conda (recommended)
conda env create -f environment.yml
conda activate ivp-deepfake

# Option B: pip
pip install -r requirements.txt
```

### 2. Place dataset

```bash
# Training images → trainval_data_final/training_data_final/
# Validation images → trainval_data_final/validation_data_final/
ls trainval_data_final/training_data_final/ | head
# 0000_real.png  0001_fake.png  ...
```

### 3. Verify dataset loads

```bash
python -c "
from src.datasets.base_dataset import DeepfakeDataset
ds = DeepfakeDataset('trainval_data_final/training_data_final')
print(ds)
img, label = ds[0]
print('Image type:', type(img), '| Label:', label)
"
```

### 4. Train (Phase 3 — to be implemented)

```bash
python scripts/train.py --config configs/default.yaml
```

### 5. Evaluate (Phase 4 — to be implemented)

```bash
python src/evaluation/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --data_dir data/test \
  --output_dir results/
```

---

## Key Hyperparameters

See [`configs/default.yaml`](configs/default.yaml) for all parameters. Critical ones:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `weight_decay` | 5e-3 | Strong L2 regularisation for small dataset |
| `fmsi_mask_ratio` | 0.30 | 30% of DCT coefficients masked per step |
| `fusion_dropout` | 0.3 | Dropout applied inside Fusion gate and head |
| `mixup_alpha` | 0.4 | Mixup data augmentation parameter |
| `label_smoothing` | 0.1 | Converts hard labels to 0.05/0.95 |
| `layer_lr_decay` | 0.75 | 25% LR reduction per deeper Swin stage |

---

## Success Metrics

| Metric | Target | Current Results |
|--------|--------|-----------------|
| AUC on val set | ≥ 0.85 | **0.84** (peak prior to early stopping) |
| Accuracy | — | **0.81** (calibrated via ROC Youden's J) |
| Precision | — | **0.92** |
| AUC at JPEG QF=50 | < 10 pt drop vs. clean | *Pending evaluation* |

---

## Recent Improvements

To combat severe overfitting on the small dataset, the pipeline was recently overhauled with the following regularisation strategies:
1. **Stronger Degradations & Augmentations**: Added `Mixup (α=0.4)`, `GridDistortion`, `CoarseDropout`, `GaussNoise`, and `RandomRotate90`. Increased `ColorJitter`.
2. **Layer-wise LR Decay**: The Swin-V2 backbone is trained with progressively lower learning rates in earlier stages (`decay=0.75`) to avoid destroying the ImageNet-1K pretrained weights.
3. **Enhanced Regularisation**: Increased weight decay to `5e-3` and added label smoothing (`ε=0.1`) to the `BCEWithLogitsLoss`.
4. **Optimal Threshold Calibration**: The evaluation script dynamically calculates the optimum decision boundary using Youden's J statistic from the ROC curve, yielding an immediate **+6.5% Validation Accuracy** improvement over a hard `0.5` threshold cutoff.
