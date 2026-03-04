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

### Phase 3 — Fusion & Training Loop *(Status: Stub)*
- [x] `CrossAttentionFusion` — quality-gated cross-attention + classification head
- [x] `HybridSwinNet` — top-level model composition
- [x] `Trainer` scaffold — AdamW, cosine LR, checkpoint saving
- [ ] Implement `train_epoch()` and `validate_epoch()`
- [ ] Apply FMSI in training forward pass
- [ ] Integrate `wandb` logging

### Phase 4 — Evaluation *(Status: Stub)*
- [x] `evaluate.py` CLI scaffold
- [ ] Per-generator AUC table
- [ ] ROC curve plots (`matplotlib`)
- [ ] JPEG QF=50 robustness run
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
| `batch_size` | 32 | Reduce to 16 + grad accumulation on 16 GB VRAM |
| `jpeg_qf_range` | [30, 60] | Hostile augmentation range |
| `fmsi_mask_ratio` | 0.15 | 15% of DCT coefficients masked per step |
| `fusion_heads` | 8 | Multi-head attention heads in fusion block |
| `lr` | 1e-4 | AdamW, cosine annealing |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| AUC on val set | ≥ 0.85 |
| AUC at JPEG QF=50 | < 10 pt drop vs. clean |
| AUC on unseen generators | ≥ 0.85 |
