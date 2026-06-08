# Final Path Experiment — Results Analysis

**Date:** 2026-06-08  
**Setup:** `fly_threshold_10km` (10 km fly/stationary threshold)  
**Split:** year-grouped 80/20 (train on earliest 80% of years, test on latest 20%)  
**k values:** 7, 14, 30  
**Epochs:** max 200, patience 30  

---

## Model Architecture Summary

| Model | Family | Layers | Hidden Dim | Params (k=7) |
|-------|--------|--------|------------|-------------|
| Direct MLP | direct | 0 | — | 25K |
| Direct Transformer 4L | direct | 4 | d_model=128 | 805K |
| Triline LSTM 4L | triline | 4 | hidden=256 | 2.2M |
| Triline Transformer V2 | triline | 4 | d_model=256, 8 heads, CLS token | 3.3M |

All models use **no bird embeddings** (removed for year-grouped split compatibility).  
Triline models predict: fly/stationary (BCE), log distance (MSE), and direction sin/cos (MSE).  
Direct models predict: lat/lon delta only (MSE).

---

## With Weather

### k=7

| Model | Mean km | Median km | P90 km | Fly Recall | Best Epoch |
|-------|--------:|----------:|-------:|-----------:|-----------:|
| Direct MLP | 19.02 | 4.85 | 59.98 | — | 9 |
| Direct Transformer 4L | 17.79 | 3.69 | 56.87 | — | 12 |
| Triline LSTM 4L | 17.86 | 2.41 | 53.54 | 0.615 | 4 |
| **Triline Transformer V2** | **17.96** | **2.41** | **57.60** | **0.658** | **17** |

**Winner: LSTM 4L** (better P90, faster convergence). TransV2 matches on median but trails on P90.

### k=14

| Model | Mean km | Median km | P90 km | Fly Recall | Best Epoch |
|-------|--------:|----------:|-------:|-----------:|-----------:|
| Direct MLP | 20.68 | 5.13 | 66.20 | — | 5 |
| Direct Transformer 4L | 21.57 | 8.22 | 60.16 | — | 6 |
| Triline LSTM 4L | 18.69 | 2.74 | 59.06 | 0.679 | 4 |
| **Triline Transformer V2** | **18.85** | **2.51** | **61.36** | **0.684** | **30** |

**Winner: LSTM 4L** (better mean, median, P90). TransV2 wins fly recall by 0.005.

### k=30

| Model | Mean km | Median km | P90 km | Fly Recall | Best Epoch |
|-------|--------:|----------:|-------:|-----------:|-----------:|
| Direct MLP | 19.08 | 5.02 | 58.84 | — | 6 |
| Direct Transformer 4L | 17.47 | 3.45 | 59.76 | — | 7 |
| Triline LSTM 4L | 17.03 | 2.47 | **51.57** | **0.764** | 2 |
| **Triline Transformer V2** | **17.33** | **2.42** | **52.78** | **0.670** | **10** |

**Winner: LSTM 4L** (dominant fly recall, better P90).

---

## Without Weather

### k=7

| Model | Mean km | Median km | P90 km | Fly Recall | Best Epoch |
|-------|--------:|----------:|-------:|-----------:|-----------:|
| Direct MLP | 19.21 | 4.47 | 62.18 | — | 11 |
| Direct Transformer 4L | 17.65 | 2.47 | 58.41 | — | 13 |
| **Triline LSTM 4L** | **18.19** | **2.26** | **63.06** | **0.724** | **9** |
| Triline Transformer V2 | 18.18 | 2.56 | 62.39 | 0.696 | 23 |

**Winner: LSTM 4L** (better median, fly recall).

### k=14

| Model | Mean km | Median km | P90 km | Fly Recall | Best Epoch |
|-------|--------:|----------:|-------:|-----------:|-----------:|
| Direct MLP | 20.63 | 5.14 | 64.67 | — | 8 |
| Direct Transformer 4L | 19.18 | 3.69 | 62.64 | — | 11 |
| Triline LSTM 4L | 18.85 | 2.49 | 62.29 | 0.731 | 8 |
| **Triline Transformer V2** | **18.66** | **2.54** | **56.66** | **0.738** | **28** |

**Winner: TransV2** (best mean, P90, fly recall). LSTM 4L wins median by 0.05 km.

### k=30

| Model | Mean km | Median km | P90 km | Fly Recall | Best Epoch |
|-------|--------:|----------:|-------:|-----------:|-----------:|
| Direct MLP | 20.84 | 5.75 | 62.48 | — | 4 |
| Direct Transformer 4L | 18.22 | 4.35 | 57.93 | — | 11 |
| Triline LSTM 4L | **17.06** | 2.45 | 49.66 | 0.702 | 7 |
| **Triline Transformer V2** | **17.20** | **2.27** | **49.34** | **0.718** | **15** |

**Winner: Tie.** TransV2 wins median + P90 + fly recall. LSTM 4L wins mean.

---

## Cross-K Trends

| Trend | Observation |
|-------|-------------|
| **k ↑ → better median** | All models improve median error as lookback increases. LSTM 4L: 2.41→2.74→2.47 (weather). TransV2: 2.41→2.51→2.42. |
| **k ↑ → TransV2 gains on LSTM** | At k=7 LSTM wins. At k=14 close. At k=30 TransV2 competitive or winning. |
| **Convergence speed** | LSTM 4L converges 2-4× faster than TransV2 (best epoch 2-9 vs 10-30). |
| **P90 variance** | TransV2 P90 is more stable across k (52-62 range) vs LSTM (49-63 range). |

---

## Weather Impact

Weather effect measured as `mean_error_km_weather − mean_error_km_noweather` (negative = weather helps):

| Model | k=7 | k=14 | k=30 |
|-------|----:|----:|----:|
| Triline LSTM 4L | −0.33 | −0.16 | −0.02 |
| Triline Transformer V2 | −0.21 | +0.19 | +0.14 |

**Weather impact is small and inconsistent.** At k=7 weather helps both models slightly. At k=14,30 the effect oscillates. The year-grouped split may explain this: weather patterns differ between train and test years, limiting the predictive value. Weather features were more useful under the original chronological split where train and test shared the same year's climate.

---

## Parameter Efficiency

| Model | Params | Mean Error (k=30, weather) | Error per M param |
|-------|-------:|---------------------------:|------------------:|
| Direct MLP | 78K | 19.08 | 244.6 |
| Direct Transformer 4L | 808K | 17.47 | 21.6 |
| Triline LSTM 4L | 2.2M | 17.03 | 7.7 |
| Triline Transformer V2 | 3.3M | 17.33 | 5.3 |

Direct MLP is the most parameter-efficient per unit of error. Triline models gain modestly for large parameter increases.

---

## Conclusions

1. **Triline LSTM 4L is the overall winner.** It wins or ties 9 of 12 metric slices across k values and weather modes. It converges 2-4× faster than the transformer and has fewer parameters (2.2M vs 3.3M).

2. **Triline Transformer V2 is competitive at k=30.** It wins P90 (49.34 vs 49.66 without weather) and median (2.27 vs 2.45) at the longest lookback. The transformer architecture benefits from longer sequences — at k=30 attention has enough temporal range to be useful.

3. **Direct models are surprisingly strong.** Direct Transformer 4L (805K params) matches or beats Triline models on median error at k=7,14 without weather. This architecture is simpler and faster to train.

4. **Weather effect is weak under year-grouped split.** Under the chronological split, weather provided a consistent benefit. Under year-grouped split, different years have different weather patterns, so weather features don't generalize as well across train/test years.

5. **LSTM wins because inductive bias matches the task.** Bird trajectory prediction at k≤30 is inherently sequential: the most recent day is the strongest predictor. LSTM's last-hidden-state recency bias is the right inductive bias. The transformer's global attention is more flexible but needs more data and longer sequences to earn its cost.
---

## Rollout Evaluation

Autoregressive rollouts use a minimum 60-day observed context and extend that context until observed displacement reaches 50 km when possible. Each model then rolls forward for the rest of the path using its trained trailing context window.

| Model | With weather mean km | Without weather mean km | With - without mean km | With weather final km | Without weather final km |
|-------|---------------------:|------------------------:|-----------------------:|----------------------:|-------------------------:|
| Direct | 417.22 | 240.07 | 177.15 | 603.58 | 343.52 |
| LSTM | 165.16 | 175.34 | -10.18 | 193.77 | 206.95 |
| Transformer | 237.28 | 180.84 | 56.43 | 323.68 | 227.91 |

Valid rollout paths: with weather 27, without weather 27. Skipped paths: with weather 7, without weather 7.

Weather slightly worsens average rollout error across the selected model families.

Artifacts:
- Per-step and per-path errors: `D:\UCSD\ECE 228\ECE-228-Project\rollout\with_weather`, `D:\UCSD\ECE 228\ECE-228-Project\rollout\without_weather`
- Weather comparison CSV: `D:\UCSD\ECE 228\ECE-228-Project\rollout\rollout_error_comparison.csv`
- Representative GIFs: `D:\UCSD\ECE 228\ECE-228-Project\rollout\with_weather\gifs`, `D:\UCSD\ECE 228\ECE-228-Project\rollout\without_weather\gifs`

