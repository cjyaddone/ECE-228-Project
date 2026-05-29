# Bird Trajectory Prediction: Comprehensive Experiment Report

## Task

Given the past **k** consecutive daily movement records of a bird, predict its **next-day GPS location**. Two prediction paradigms are explored:

1. **Direct Regression** (One-shot-Transformer): Predict delta lat/lon directly from the last transformer token
2. **Decomposed Prediction** (Triline-Transformer): Predict fly/no-fly, distance, and heading separately, then reconstruct GPS

**Dataset**: `dataset2_daily_movement.csv` — 34,421 daily records from 92 birds. Fly rate (>50 km/day) is 6.6%.

---

## Method 1: Direct Regression (One-shot Transformer)

```
┌──────────────────────────────────────────────────────┐
│  Input: past k days × 16 features                     │
│  [day₁ │ day₂ │ ... │ dayₖ]                           │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Feature Projection + Bird Embedding                   │
│  Linear(16→128) + Embedding(92 birds→32)               │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Transformer Encoder (2 layers, 4 heads, d_model=128) │
│  + Learned Positional Encoding                         │
│  + Additive Bird Embedding                             │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Take Last Token → MLP → [Δlat, Δlon]                  │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Reconstruct: pred_loc = curr_loc + [Δlat, Δlon]       │
│  Loss: MSE on normalized delta                         │
└──────────────────────────────────────────────────────┘
```

## Method 2: Triline Decomposed Prediction (Triline Transformer)

```
┌──────────────────────────────────────────────────────┐
│  Input: past k days × 16 features                     │
│  [day₁ │ day₂ │ ... │ dayₖ]                           │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Feature Projection + Bird Embedding                   │
│  Linear(16→128) + Embedding(92 birds→32)               │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Transformer Encoder (2 layers, 4 heads, d_model=128) │
│  + Learned Positional Encoding                         │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Mean Pooling → Shared Representation (128-dim)        │
└───────┬──────────────┬──────────────┬────────────────┘
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Fly Head    │ │  Dist Head   │ │  Dir Head    │
│  128→64→1    │ │  128→64→1    │ │  128→64→2    │
│  BCE loss    │ │  MSE loss    │ │  MSE loss    │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       ▼                ▼                ▼
  fly_logit       log(1+dist)      [sin θ, cos θ]
       │                │                │
       └────────────────┴────────────────┘
                        ▼
┌──────────────────────────────────────────────────────┐
│  Reconstruct:                                         │
│  dist = exp(pred_log_dist) - 1                         │
│  Δlat = dist × cos(θ) / 111                            │
│  Δlon = dist × sin(θ) / (111 × cos(lat))               │
│  pred_loc = curr_loc + [Δlat, Δlon]                    │
│                                                        │
│  Multi-task Loss:                                      │
│  L = γ·BCE(fly) + α·MSE(dist) + β·MSE(dir)            │
└──────────────────────────────────────────────────────┘
```

## Method 3: Triline LSTM (Autoregressive Baseline)

```
┌──────────────────────────────────────────────────────┐
│  Input: past k days × 16 features                     │
│  [day₁ │ day₂ │ ... │ dayₖ]                           │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Feature Projection + Bird Embedding                   │
│  Linear(16→128) + Embedding(92 birds→32)               │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  LSTM (2 layers, hidden=128, batch_first)              │
│  Processes sequence step-by-step (autoregressive)      │
│  h₁ → h₂ → ... → hₖ                                   │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Take Last Hidden State → Same 3 Heads as Triline      │
│  (Fly head, Dist head, Dir head)                       │
└──────────────────────────────────────────────────────┘
```

## Method 4: Triline LinearAR (Non-sequential Baseline)

```
┌──────────────────────────────────────────────────────┐
│  Input: past k days × n_features                       │
│  [day₁ │ day₂ │ ... │ dayₖ]                           │
│  n_features = 2 (delta only) or 16 (full)             │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Flatten: (k × n_features)-dim vector                  │
│  + Bird Embedding (32-dim)                              │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Linear Projection: (k×n_feat + 32) → 128             │
│  + LayerNorm + ReLU + Dropout                          │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Same 3 Heads as Triline                               │
│  (Fly head, Dist head, Dir head)                       │
└──────────────────────────────────────────────────────┘
```

---

## Results

### A. Direct Regression vs. Triline Decomposed

| Model | k | Mean (km) | Median (km) | Migration Mean (km) | Fly Recall |
|-------|--:|----------:|------------:|---------------------:|-----------:|
| **Baselines** | | | | | |
| Persistence | — | 13.69 | 0.30 | — | — |
| Const-Velocity | — | 32.16 | 9.98 | — | — |
| **Direct Regression** (One-shot) | | | | | |
| MLP last-day | 1 | 9.09 | 1.17 | 6.79 | — |
| MLP sequential | 30 | 8.97 | 1.17 | 7.29 | — |
| Transformer | 7 | 8.90 | 1.69 | 4.90 | — |
| Transformer | 14 | 8.98 | 0.87 | 4.93 | — |
| Transformer | 30 | **8.37** | **0.81** | **4.75** | — |
| Transformer | 60 | 8.66 | 1.44 | 4.99 | — |
| **Triline Decomposed** | | | | | |
| Transformer | 7 | 13.28 | 0.33 | 9.18 | 0.833 |
| Transformer | 14 | 13.14 | 0.32 | 9.95 | 0.804 |
| Transformer | 30 | 12.46 | 0.31 | 8.45 | 0.816 |
| Transformer | 60 | **10.43** | **0.31** | **8.69** | 0.946 |

**Key insight**: Direct regression achieves lower mean error (8.37 vs 12.46 km at k=30) but Triline has much lower median error (0.31 vs 0.81 km). Triline handles stationary days more precisely; direct regression handles migration days better.

### B. Context Length Sweep (Triline Transformer 2-layer)

| k | Train Samples | Mean (km) | Median (km) | Migration Mean (km) |
|--:|--------------:|----------:|------------:|--------------------:|
| 7 | 28,321 | 13.28 | 0.33 | 9.18 |
| 14 | 26,423 | 13.14 | 0.32 | 9.95 |
| 30 | 24,301 | 12.46 | 0.31 | 8.45 |
| **60** | **20,476** | **10.43** | **0.31** | **8.69** |
| 90 | 17,659 | 11.53 | 0.37 | 12.66 |
| 120 | 15,516 | 13.02 | 0.46 | 12.20 |

**Key insight**: k=60 is the optimal context length. Beyond k=60, reduced sample count outweighs benefits of longer history.

### C. Model Architecture Comparison (Triline paradigm, k=30)

| Model | Params | Mean (km) | Median (km) | Fly Recall |
|-------|-------:|----------:|------------:|-----------:|
| Transformer 2L | 309K | 12.46 | 0.31 | 0.816 |
| Transformer 3L | 442K | 12.60 | 0.31 | 0.837 |
| Transformer 4L | 575K | 12.65 | 0.31 | 0.776 |
| LSTM 2L | 304K | 12.57 | 0.32 | 0.857 |
| LinearAR (full) | 94K | **12.30** | 0.33 | 0.633 |
| LinearAR (delta) | 16K | 13.25 | 0.32 | 0.714 |

### D. Model Architecture Comparison (Triline paradigm, k=60)

| Model | Params | Mean (km) | Median (km) | Fly Recall |
|-------|-------:|----------:|------------:|-----------:|
| Transformer 2L | 314K | **10.43** | 0.31 | 0.946 |
| Transformer 4L | 579K | 10.69 | 0.31 | 0.811 |
| LSTM 2L | 304K | 10.59 | 0.30 | 0.892 |
| LinearAR (full) | 156K | 10.41 | 0.34 | 0.838 |

### E. Full Architecture × Context Matrix

| Architecture | k=7 | k=30 | k=60 | k=90 | k=120 |
|-------------:|----:|-----:|-----:|-----:|------:|
| Transformer 2L | 13.28 | 12.46 | **10.43** | 11.53 | 13.02 |
| Transformer 3L | — | 12.60 | — | — | — |
| Transformer 4L | — | 12.65 | 10.69 | — | — |
| LSTM 2L | 13.61 | 12.57 | 10.59 | — | — |
| LinearAR-full | 13.87 | 12.30 | 10.41 | — | — |
| LinearAR-delta | — | 13.25 | — | — | — |

### F. Baselines on All Splits

| Baseline | k=7 split | k=14 split | k=30 split | k=60 split | k=90 split | k=120 split |
|---------:|----------:|-----------:|-----------:|-----------:|-----------:|------------:|
| Persistence | 14.45 | 14.03 | 13.69 | 11.61 | 12.83 | 14.39 |
| Const-Velocity | 33.75 | 33.51 | 32.16 | 28.89 | 29.57 | 31.64 |

---

## Analysis

### 1. Context Length: k=60 is the Sweet Spot

Mean error decreases from k=7 (13.28 km) to k=60 (10.43 km) — a **21% improvement**. However, k=90 (11.53 km) and k=120 (13.02 km) degrade performance. The number of valid training samples drops sharply: 28,321 → 20,476 → 15,516. At k=120, the model simply doesn't have enough data (only 15.5K train samples) to learn effectively, and the persistence baseline itself degrades due to the different test split composition.

### 2. Architecture Matters Little — Context Matters Much

The most striking result: at k=60, a simple LinearAR (156K params, no sequence modeling) achieves **10.41 km** — essentially identical to the Transformer (10.43 km, 314K params). Even at k=30, LinearAR (12.30 km) slightly outperforms Transformer (12.46 km). This suggests that for this task, the temporal structure captured by attention or recurrence is secondary to simply having a longer lookback window.

### 3. More Layers Don't Help

At k=30: 2L=12.46, 3L=12.60, 4L=12.65 km — monotonic degradation. At k=60: 2L=10.43, 4L=10.69 km. With ~20K training samples, deeper models overfit. The 2-layer architecture with ~300K parameters is well-matched to the dataset size.

### 4. Direct Regression vs. Decomposed: Different Strengths

| Metric | Direct Regression (k=30) | Triline (k=30) |
|--------|------------------------:|---------------:|
| Mean Error | **8.37 km** | 12.46 km |
| Median Error | 0.81 km | **0.31 km** |
| Migration Error | **4.75 km** | 8.45 km |
| Stationary Error | 8.66 km | **12.78 km** |

Direct regression optimizes MSE on delta lat/lon directly — well-suited for the continuous coordinate space. Triline's distance+heading reconstruction introduces approximation error, especially on stationary days where tiny heading errors produce spurious displacements. However, Triline provides **interpretable** intermediate outputs (fly probability, distance, heading) and much lower median error (0.31 km).

### 5. Triline's Fly Detection is Strong

Fly recall is consistently high: 81-95% for the Transformer across k values. Precision is lower (29-41%), meaning the model is conservative — it correctly identifies most migration events but also has false positives. k=60 achieves the best recall (94.6%) at the cost of lowest precision (29.2%).

### 6. LSTM is Competitive but Not Superior

LSTM (k=60: 10.59 km) nearly matches Transformer (10.43 km) and LinearAR (10.41 km). The LSTM's sequential hidden state provides no clear advantage over parallel attention or even simple flattening. This reinforces finding #2: the temporal inductive bias of recurrent networks doesn't help for this particular sequence-to-point task.

### 7. Delta-Only LinearAR: Surprisingly Strong Baseline

With just 16K parameters and only delta_lat/delta_lon as input (2 features × 30 days = 60 inputs), LinearAR-delta achieves 13.25 km — close to the full transformer (12.46 km) and beating persistence (13.69 km). This confirms that raw movement deltas carry most of the predictive signal.

---

## Training Configuration (All Triline Models)

| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=4) |
| Early stopping | patience=10 epochs |
| Max epochs | 50 |
| Loss weights | γ=0.5 (fly), α=1.0 (dist), β=1.0 (dir) |
| BCE pos_weight | ~13 (auto-computed from train fly rate) |
| Seed | 42 |

---

## Files

| File | Description |
|------|-------------|
| `Triline-Transformer/model.py` | Transformer, LSTM, LinearAR models + factory |
| `Triline-Transformer/ar_models.py` | TrilineLSTM and TrilineLinearAR classes |
| `Triline-Transformer/train.py` | Multi-task training, baselines, evaluation |
| `Triline-Transformer/dataset.py` | PyTorch dataset with decomposed targets |
| `Triline-Transformer/data_preprocessing.py` | Feature engineering, window building, metrics |
| `One-shot-Transformer/model.py` | Direct regression Transformer + MLP baselines |
| `One-shot-Transformer/train.py` | Single-task training for direct GPS prediction |

---

## Conclusions

1. **k=60 is optimal** for the Triline paradigm — 21% better than k=7, but longer windows (k≥90) hurt due to reduced sample count
2. **Architecture choice is secondary** — LinearAR matches Transformer at k=60 despite having no attention mechanism, suggesting the key factor is the window length, not how it's processed
3. **Deeper models overfit** — 3L and 4L transformers consistently underperform 2L at this dataset size
4. **Triline excels at stationary days** (median 0.31 km) while **direct regression excels at migration days** (4.75 km mean)
5. **Fly detection works well** (>80% recall) — the model successfully identifies when a bird will migrate
6. **LinearAR (156K params)** is the most parameter-efficient model — matching the Transformer at half the size
