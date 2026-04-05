# PM2.5 Pollution Forecasting — ANRF-AISE Hackathon Phase 2 (Theme 2)
### IIIT Hyderabad Submission | IIT Delhi Competition

---

## Overview

This repository contains the solution for **Phase 2 of the ANRF-AISE Hackathon — Theme 2: Pollution Forecasting**, hosted by IIT Delhi. The task is to forecast **PM2.5 concentrations** across a 140×124 spatial grid covering India for the **next 16 hours**, given 10 hours of historical atmospheric and emission data.

The model achieved a **competition score of 0.924** on the validation set.

---

## Problem Statement

- **Input**: 10 timesteps of 15 atmospheric/emission variables across a 140×124 grid → **170 channels**
- **Output**: PM2.5 concentration at every grid cell for the **next 16 hours** → shape `(16, 140, 124)`
- **Challenge**: Accurately forecast rare but dangerous **pollution episodes** (sudden PM2.5 spikes, ~1% of data) that are critical for health emergency response

---

## Dataset

Data is sourced from 4 months of 2016 representing different seasons:

| Month | Season |
|---|---|
| `APRIL_16` | Spring |
| `JULY_16` | Monsoon |
| `OCT_16` | Post-Monsoon |
| `DEC_16` | Winter |

### Input Features (15 variables × 10 timesteps = 150 channels + 10 sin/cos hour encodings = 170 total)

| Group | Features |
|---|---|
| **Target (PM2.5)** | `cpm25` — current PM2.5 concentration |
| **Meteorology** | `u10`, `v10` (winds), `t2` (temperature), `q2` (humidity), `psfc` (pressure), `pblh` (boundary layer height), `rain`, `swdown` (solar radiation) |
| **Emissions** | `PM25`, `NH3`, `SO2`, `NOx`, `NMVOC_e`, `NMVOC_finn`, `bio` |

### Data Split

- **Train**: April, July, December (full) + October (first 539 hours) → **1316 samples**
- **Validation**: October (last 200 hours) → **87 samples**
- **Test**: Provided separately via `test_in/` → **218 samples**

---

## Model Architecture — `ImprovedSkipUNet` (v9)

A hybrid **ConvLSTM + U-Net** architecture with spatial self-attention.

```
Input x: (B, 170, H, W)
        │
        ├─── Trend Delta: x[t=9] - x[t=7]  →  (B, 1, H, W)
        │
        ├─── StackedConvLSTM (pm25 + u10 + v10 + pblh over 10 timesteps)
        │         Cell1: 4ch → 32ch hidden
        │         Cell2: 32ch → 64ch hidden
        │         Output: (B, 64, H, W)  [temporal context map]
        │
        └─── Concatenate: [x (170) | trend (1) | ctx (64)] = (B, 235, H, W)
                    │
                    ▼
            U-Net Encoder
              enc1: (B, 96,  140, 124)
              enc2: (B, 192,  70,  62)
              enc3: (B, 384,  35,  31)
              bottleneck: (B, 768, 17, 15)  ← CBAM + SpatialSelfAttention
                    │
                    ▼
            U-Net Decoder (skip connections)
              dec3: (B, 384, 35, 31)  ← CBAM
              dec2: (B, 192, 70, 62)  ← CBAM
              dec1: (B, 96, 140, 124)
                    │
                    ▼
            Output: (B, 16, H, W) + last_pm25  [residual prediction]
```

### Key Components

| Component | Purpose |
|---|---|
| `ConvLSTMCell` | Spatially-aware LSTM cell — maintains 2D memory maps, not flat vectors |
| `StackedConvLSTM` | 2-layer temporal encoder on wind + PM2.5 — learns how pollution drifts |
| `DoubleConv` | Basic U-Net block: 2× (Conv → GroupNorm → GELU) |
| `CBAM` | Channel + Spatial Attention — focuses on relevant features and locations |
| `SpatialSelfAttention` | Multi-head self-attention at bottleneck — connects distant grid cells (upwind → downwind) |
| `EMA` | Exponential Moving Average of weights — smoother, more stable inference weights |
| **Residual Output** | Predicts delta from current PM2.5 — physically grounded, easier optimization |
| **Trend Delta** | Explicit momentum signal: rate of change of PM2.5 over last 2 hours |

**Total Parameters**: 20,345,822 (StackedConvLSTM: 267,520)

---

## Loss Function

A **dynamic composite loss** that shifts focus from smooth learning to episode accuracy over training:

```python
Total Loss = huberw × L_huber(all)
           + smapew × L_smape(all)
           + epw    × [L_huber(episodes, δ=0.5) + L_smape(episodes)]
```

| Epoch | `huberw` | `smapew` | `epw` |
|---|---|---|---|
| 1 | 0.50 | 0.30 | 0.20 |
| 40 | 0.475 | 0.25 | 0.275 |
| 80 | 0.45 | 0.20 | 0.35 |

- **Huber Loss** (`δ=1.0`): Stable training, robust to outliers
- **SMAPE**: Relative error — aligns with competition metric
- **Episode Loss** (`δ=0.5`): Targets the critical ~1% pollution spike pixels with extra pressure

---

## Pollution Episode Detection

Episodes are detected using **STL decomposition**:
1. Remove 25-hour moving average trend from PM2.5
2. Remove hourly diurnal cycle
3. Flag residual > `3 × spatial sigma` as an episode

```
April:    episode = 0.988%,  σ_mean = 5.23
July:     episode = 0.931%,  σ_mean = 3.60
October:  episode = 0.842%,  σ_mean = 7.29
December: episode = 1.020%,  σ_mean = 10.36
```

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 80 |
| Batch Size | 16 |
| Learning Rate | 5e-4 (AdamW) |
| Weight Decay | 1e-4 |
| Gradient Clip | 1.0 |
| Warmup Epochs | 5 |
| EMA Decay | 0.999 |
| Dropout | 0.10 |
| Patience (early stop) | 15 |
| Hardware | 2× Tesla T4 GPUs (DataParallel) |
| Mixed Precision | ✅ `torch.amp.autocast` |

### Learning Rate Schedule
- Linear warmup: epochs 1–5 (0 → 5e-4)
- Cosine annealing: epochs 6–80 (5e-4 → 1e-5)

---

## Data Preprocessing

- **Log-transform** applied to skewed features before normalization: `cpm25, rain, pblh, PM25, NH3, SO2, NOx, NMVOCe, bio`
- **Z-score normalization**: `(x - mean) / std` computed from 5% random sample of training data
- **Sin/Cos hour encoding**: `sin(2π·h/24)` and `cos(2π·h/24)` appended as 10 timestep channels each

---

## Data Augmentation

Applied **only during training**:
- Random horizontal flip (50% probability)
- Random vertical flip (70% probability)

---

## Test-Time Augmentation (TTA)

At inference, **4 flip combinations** are averaged:
1. No flip
2. Horizontal flip
3. Vertical flip
4. Horizontal + Vertical flip

Each flip is applied, predicted, then un-flipped before averaging — reduces prediction variance.

---

## Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| `gSMAPE` | Mean SMAPE over all pixels & times | Global relative accuracy |
| `eCorr` | Pearson correlation on episode pixels | Episode spatial pattern accuracy |
| `eSMAPE` | SMAPE on episode pixels only | Episode magnitude accuracy |
| **Score** | `((1 - gSMAPE/2) + (eCorr+1)/2 + (1 - eSMAPE/2)) / 3` | **Final competition score** |

---

## Results

| Metric | Value |
|---|---|
| Validation Score | **0.9239** |
| gSMAPE | 0.1843 |
| eCorr | 0.9231 |
| eSMAPE | 0.1954 |
| Training Time | ~3 hours (80 epochs, dual T4) |

Score progression: **0.813** (epoch 1) → **0.924** (epoch 80)

---

## Output

Predictions are saved as:
```
preds.npy  — shape: (218, 140, 124, 16)
             218 test windows × 140×124 grid × 16 forecast hours
             Values in original PM2.5 units (µg/m³, denormalized)
```

---

## File Structure

```
iiith-hack1-3.ipynb        ← Main training + inference notebook
```

### Key Input Paths (Kaggle)
```
/kaggle/input/.../raw/APRIL_16/     ← Training data (April 2016)
/kaggle/input/.../raw/JULY_16/      ← Training data (July 2016)
/kaggle/input/.../raw/OCT_16/       ← Train + Val data (October 2016)
/kaggle/input/.../raw/DEC_16/       ← Training data (December 2016)
/kaggle/input/.../test_in/          ← Test input features
/kaggle/input/.../stats/feat_min_max.mat  ← Feature statistics
```

---

## Dependencies

```python
torch >= 2.0
numpy
scipy
pandas
pathlib
datetime
copy (deepcopy)
```

---

## Competition

**ANRF-AISE Hackathon Phase 2 — Theme 2: Pollution Forecasting**
Organized by **IIT Delhi**
Competition: `anrf-aise-hack-phase-2-theme-2-pollution-forecasting-iitd`

