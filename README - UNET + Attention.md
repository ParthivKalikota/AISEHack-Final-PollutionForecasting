# ANRF AISE Hack — Phase 2, Theme 2: PM2.5 Pollution Forecasting (IIT Delhi)
## Notebook: `notebook52859f54b8.ipynb` — ImprovedSkipUNet v3

---

## Competition Overview

| Field | Detail |
|---|---|
| Competition | ANRF AISE Hack Phase 2 — Theme 2 |
| Host | IIT Delhi |
| Task | Spatiotemporal PM2.5 forecasting |
| Input | 10h of PM2.5 history + meteorology + emissions |
| Output | Next 16h PM2.5 concentrations |
| Spatial Grid | 140 x 124 |
| Test Samples | 218 samples |
| Training Data | April, July, October, December 2016 |
| Target Score | 0.9300 |

---

## Evaluation Metric

The official competition score is a weighted combination of three sub-metrics:

    Score = (1 - gSMAPE)/2  +  eCorr/2  +  (1 - eSMAPE)/2 * 3

| Metric | Description |
|---|---|
| gSMAPE | Global SMAPE across all timesteps and grid cells |
| eCorr | Pearson correlation on pollution episode pixels (r > 3*sigma) |
| eSMAPE | SMAPE restricted to pollution episode pixels |

Episodes are identified using the official STL decomposition: a pixel at time t is an episode if its
residual (after removing 25h MA trend + diurnal cycle) exceeds 3x its spatial standard deviation.

---

## Input Features (170 channels total)

| Group | Features | Timesteps | Channels |
|---|---|---|---|
| PM2.5 History | cpm25 | 10 | 10 |
| Meteorology | u10, v10, t2, q2, psfc, pblh, rain, swdown | 10 | 80 |
| Emissions | PM25, NH3, SO2, NOx, NMVOCe, bio | 10 | 60 |
| Diurnal Encoding | sin(hour), cos(hour) | 10 | 20 |
| TOTAL | | | 170 |

Note: An internal trend channel (+1) is also computed inside the model forward pass
      as: x[:, T_IN-1:T_IN] - x[:, T_IN-3:T_IN-2]

---

## Model Architecture — ImprovedSkipUNet v3

Input: (B, 170+1, 140, 124)   <- 170 features + 1 trend channel

    [Encoder]
    DoubleConv (B, 96, 140, 124)          <- enc1
    MaxPool -> DoubleConv (B, 192, 70, 62) <- enc2
    MaxPool -> DoubleConv (B, 384, 35, 31) <- enc3

    [Bottleneck]
    MaxPool -> DoubleConv (B, 768, 17, 15)
    CBAM(768)                             <- Channel + Spatial Attention
    SpatialSelfAttention(768, nheads=8)   <- Multi-head self-attention

    [Decoder with skip connections]
    ConvTranspose + cat(enc3) -> DoubleConv + CBAM (384)  <- dec3
    ConvTranspose + cat(enc2) -> DoubleConv + CBAM (192)  <- dec2
    ConvTranspose + cat(enc1) -> DoubleConv        (96)   <- dec1

    Conv2d(96 -> 16) + last_pm25 residual skip
Output: (B, 16, 140, 124)

Parameter Count: ~20,023,006 (~20M params)

---

## Training Configuration

    epochs:         120   # was 80 in v2
    batch_size:     16
    lr:             5e-4
    weight_decay:   1e-4
    grad_clip:      1.0
    patience:       20    # was 15 in v2
    warmup_epochs:  5
    ema_decay:      0.999
    Hardware:       2x Tesla T4 (nn.DataParallel)

### Loss Function — Composite (Dynamic Weights)

    progress = epoch / total_epochs   # 0 -> 1

    ep_w    = 0.20 + 0.25 * progress  # 0.20 -> 0.45  (ceiling was 0.35 in v2)
    huber_w = 0.50 - 0.10 * progress  # 0.50 -> 0.40
    smape_w = 1.0 - ep_w - huber_w    # 0.30 -> 0.15

    loss = huber_w * Huber(pred, target, delta=1.0)
         + smape_w * SMAPE(pred, target)
         + ep_w    * [Huber(ep_pixels, delta=0.5) + SMAPE(ep_pixels)]

### LR Schedule — Cosine Annealing with Linear Warmup + Floor

    if epoch < warmup_epochs:
        lr_scale = (epoch + 1) / warmup_epochs
    else:
        p = (epoch - warmup) / (total - warmup)
        lr_scale = 0.5 * (1 + cos(pi*p)) * (1 - 0.005) + 0.005
        # Floor = 0.005 (was 0.02 in v2 — lower floor allows finer late convergence)

---

## Validation Results (October, last 200h as val set)

| Epoch | Train Loss | Val Loss | gSMAPE | eCorr | eSMAPE | Score |
|---|---|---|---|---|---|---|
| 1 | 0.1969 | 0.2364 | 0.3831 | 0.7830 | 0.6041 | 0.7993 |
| 10 | 0.1242 | 0.1257 | 0.2225 | 0.9047 | 0.3213 | 0.8935 |
| 30 | 0.1011 | 0.0969 | 0.1948 | 0.9157 | 0.2083 | 0.9188 |
| 50 | 0.0894 | 0.0929 | 0.1900 | 0.9227 | 0.1871 | 0.9243 |
| 80 | 0.0786 | 0.0957 | 0.1868 | 0.9296 | 0.1832 | 0.9266 |
| 100 | 0.0745 | 0.0985 | 0.1860 | 0.9323 | 0.1833 | 0.9272 |
| 120 | 0.0728 | 0.1007 | 0.1868 | 0.9337 | 0.1815 | 0.9276 |

Best EMA Score: 0.9276 (target: 0.9300)

---

## Data Augmentation (Training Only)

- Horizontal flip: p=0.5  ->  arr[:, :, ::-1]
- Vertical flip:   p=0.7  ->  arr[:, ::-1, :]

Applied symmetrically to both input (x) and target (y).

---

## Inference — 8-aug TTA + Checkpoint Ensemble

### 8 Test-Time Augmentations

    AUG_CONFIGS = [
        (flip_h, flip_v, rot180)
        for flip_h in [False, True]
        for flip_v in [False, True]
        for rot180 in [False, True]
    ]
    # 2 x 2 x 2 = 8 augmentations (doubled from v2's 4 flip-only)

Each augmentation: apply to input -> forward pass -> inverse transform output -> average all 8.

### 4-Checkpoint Ensemble

    UNet_ep100.pt  -> TTA predictions
    UNet_ep110.pt  -> TTA predictions
    UNet_ep120.pt  -> TTA predictions
    UNet_best.pt   -> TTA predictions
    ─────────────────────────────────
            Mean of all 4

### Final Prediction Stats

    Shape:  (218, 140, 124, 16)
    Mean:   37.20  |  Median: 18.16  |  P95: 131.3  |  Max: 2611.3
    NaNs:   0
    Output: /kaggle/working/preds.npy

---

## What Changed: v1 -> v2 -> v3

### v1 (Phase 1 — trainingv8.ipynb) vs v2/v3

| Aspect | Phase 1 (v1) | Phase 2 (v3) |
|---|---|---|
| Architecture | DeepFNO (82M) + SkipUNet (7.9M) | Single ImprovedSkipUNet (20M) |
| Input channels | 270 (no emissions; T_MET=26) | 170 (with emissions; T_MET=10) |
| Emission features | None | PM25, NH3, SO2, NOx, NMVOCe, bio |
| Loss | Huber only | Composite (Huber + SMAPE + Episode) |
| Metric optimised | Domain RMSE | Competition Score |
| Episode masking | None | STL (r > 3*sigma) |
| EMA | No | Yes (decay=0.999) |
| Augmentation | No | H/V flips (train) |
| TTA | No | 8-aug TTA (inference) |
| Checkpoint ensemble | FNO + UNet | Last 3 + best (4 total) |
| Multi-GPU | No | DataParallel (2x T4) |
| Max epochs | 60 (patience=12) | 120 (patience=20) |
| Attention | CBAM (bottleneck) | CBAM + SpatialSelfAttention |

### v2 -> v3 (this notebook)

| Aspect | v2 | v3 |
|---|---|---|
| Model base width | 64 (~9M params) | 96 (~20M params) |
| Episode weight ceiling (ep_w) | 0.35 | 0.45 |
| LR cosine floor | 0.02 | 0.005 |
| Epochs | 80 | 120 |
| Patience | 15 | 20 |
| TTA augmentations | 4 (flips only) | 8 (flips + 180deg rotation) |
| Checkpoint ensemble | Last 2 + best | Last 3 + best (4 total) |
| Best val score | 0.9247 | 0.9276 |

---

## Potential Improvements to reach 0.9300+

1. Reduce dataset stride to 1 (currently stride=2) for more training samples
2. Add CBAM at enc1 decoder level for finer spatial attention
3. Replace MaxPool2d with strided Conv2d for less information loss
4. Add month-of-year sinusoidal embedding as extra input channels (seasonal context)
5. Increase PM2.5 history window: T_IN=12 or 16 instead of 10
6. Add spatial gradient loss term to encourage smooth predictions
7. Encode lat/lon grid as additional learnable positional embedding channels
8. Stochastic depth / DropPath regularization for larger model capacity

---

## Dependencies

    pip install timm einops
    # Standard: torch, numpy, scipy, pathlib, tqdm

Hardware: 2x Tesla T4 GPUs (nn.DataParallel)
Run time: ~2.9 hours for 120 epochs

---

## Output Files

| File | Description |
|---|---|
| UNet_best.pt | Best EMA model checkpoint |
| UNet_ep{N}.pt | Epoch checkpoints saved every 10 epochs |
| preds.npy | Final predictions array: shape (218, 140, 124, 16) |

