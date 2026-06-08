# Final Path Experiment - Results Analysis

**Date:** 2026-06-08
**Setup:** `fly_threshold_10km` (10 km fly/stationary threshold)
**Split:** year-grouped 80/20 (train on earliest 80% of years, test on latest 20%)
**k values:** 7, 14, 30
**Epochs:** max 200, patience 30

Triline LSTM and Transformer GPS errors are now reported with fly/no-fly gating as the primary metric. Ungated diagnostics remain in the CSVs with `ungated_` prefixes.

---

## With Weather

### k=7

| Model | Mean km | Median km | P90 km | Fly Recall | Best Epoch |
|-------|--------:|----------:|-------:|-----------:|-----------:|
| Direct MLP | 19.02 | 4.85 | 59.98 | -- | 9 |
| Direct Transformer 4L | 17.79 | 3.69 | 56.87 | -- | 12 |
| Triline LSTM 4L | 17.33 | 1.41 | 53.57 | 0.615 | 4 |
| Triline Transformer V2 | 17.44 | 1.51 | 57.60 | 0.658 | 17 |

**Best mean-error model:** Triline LSTM 4L (17.33 km).

### k=14

| Model | Mean km | Median km | P90 km | Fly Recall | Best Epoch |
|-------|--------:|----------:|-------:|-----------:|-----------:|
| Direct MLP | 20.68 | 5.13 | 66.20 | -- | 5 |
| Direct Transformer 4L | 21.57 | 8.22 | 60.16 | -- | 6 |
| Triline LSTM 4L | 18.02 | 1.57 | 59.06 | 0.679 | 4 |
| Triline Transformer V2 | 18.31 | 1.62 | 62.05 | 0.684 | 30 |

**Best mean-error model:** Triline LSTM 4L (18.02 km).

### k=30

| Model | Mean km | Median km | P90 km | Fly Recall | Best Epoch |
|-------|--------:|----------:|-------:|-----------:|-----------:|
| Direct MLP | 19.08 | 5.02 | 58.84 | -- | 6 |
| Direct Transformer 4L | 17.47 | 3.45 | 59.76 | -- | 7 |
| Triline LSTM 4L | 16.57 | 1.96 | 51.81 | 0.764 | 2 |
| Triline Transformer V2 | 16.83 | 1.70 | 53.04 | 0.670 | 10 |

**Best mean-error model:** Triline LSTM 4L (16.57 km).

---

## Without Weather

### k=7

| Model | Mean km | Median km | P90 km | Fly Recall | Best Epoch |
|-------|--------:|----------:|-------:|-----------:|-----------:|
| Direct MLP | 19.21 | 4.47 | 62.18 | -- | 11 |
| Direct Transformer 4L | 17.65 | 2.47 | 58.41 | -- | 13 |
| Triline LSTM 4L | 17.77 | 1.59 | 63.24 | 0.724 | 9 |
| Triline Transformer V2 | 17.58 | 1.52 | 62.75 | 0.696 | 23 |

**Best mean-error model:** Triline Transformer V2 (17.58 km).

### k=14

| Model | Mean km | Median km | P90 km | Fly Recall | Best Epoch |
|-------|--------:|----------:|-------:|-----------:|-----------:|
| Direct MLP | 20.63 | 5.14 | 64.67 | -- | 8 |
| Direct Transformer 4L | 19.18 | 3.69 | 62.64 | -- | 11 |
| Triline LSTM 4L | 18.31 | 1.65 | 62.29 | 0.731 | 8 |
| Triline Transformer V2 | 18.11 | 1.64 | 56.66 | 0.738 | 28 |

**Best mean-error model:** Triline Transformer V2 (18.11 km).

### k=30

| Model | Mean km | Median km | P90 km | Fly Recall | Best Epoch |
|-------|--------:|----------:|-------:|-----------:|-----------:|
| Direct MLP | 20.84 | 5.75 | 62.48 | -- | 4 |
| Direct Transformer 4L | 18.22 | 4.35 | 57.93 | -- | 11 |
| Triline LSTM 4L | 16.51 | 1.73 | 50.50 | 0.702 | 7 |
| Triline Transformer V2 | 16.76 | 1.89 | 49.49 | 0.718 | 15 |

**Best mean-error model:** Triline LSTM 4L (16.51 km).

---

## Weather Impact

Weather effect measured as `mean_error_km_weather - mean_error_km_noweather` (negative = weather helps):

| Model | k=7 | k=14 | k=30 |
|-------|----:|----:|----:|
| Triline LSTM 4L | -0.44 | -0.28 | +0.05 |
| Triline Transformer V2 | -0.14 | +0.20 | +0.07 |

Weather impact remains small and inconsistent. It helps both triline models at k=7, helps the LSTM at k=14, and is slightly worse at k=30.

---

## Parameter Efficiency

| Model | Params | Mean Error (k=30, weather) | Error per M param |
|-------|-------:|---------------------------:|------------------:|
| Direct MLP | 78K | 19.08 | 244.9 |
| Direct Transformer 4L | 808K | 17.47 | 21.6 |
| Triline LSTM 4L | 2161K | 16.57 | 7.7 |
| Triline Transformer V2 | 3297K | 16.83 | 5.1 |

---

## Conclusions

1. **Triline LSTM 4L remains the strongest overall mean-error model.** It has the best k=30 mean error in both weather and no-weather settings after gating.
2. **Gating materially improves typical triline error.** Median triline errors are now around 1.4-2.0 km because predicted no-fly days hold position.
3. **Transformer V2 is competitive in tail metrics.** It wins no-weather k=30 P90 and has close mean error, but does not consistently beat the LSTM.
4. **Weather effect is weak under year-grouped split.** The most reliable improvement is at shorter context; at k=30, no-weather triline models are slightly better.

---

## Rollout Evaluation

Autoregressive rollouts use a minimum 60-day observed context and extend that context until observed displacement reaches 50 km when possible. Triline rollouts now gate predicted movement using each model selected fly threshold.

| Model | With weather mean km | Without weather mean km | With - without mean km | With weather final km | Without weather final km |
|-------|---------------------:|------------------------:|-----------------------:|----------------------:|-------------------------:|
| Direct | 417.22 | 240.07 | +177.15 | 603.58 | 343.52 |
| LSTM | 199.94 | 208.86 | -8.93 | 248.82 | 266.35 |
| Transformer | 226.78 | 200.89 | +25.89 | 316.39 | 257.22 |

Valid rollout paths: with weather 27, without weather 27. Skipped paths: with weather 7, without weather 7.

Weather improves the LSTM rollout but worsens Direct and Transformer rollouts in this evaluation.

Artifacts:
- Per-step and per-path errors: `rollout/with_weather`, `rollout/without_weather`
- Weather comparison CSV: `rollout/rollout_error_comparison.csv`
- Representative GIFs: `rollout/with_weather/gifs`, `rollout/without_weather/gifs`
