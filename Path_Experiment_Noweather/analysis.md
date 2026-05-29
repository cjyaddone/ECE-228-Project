# Path Experiment No Weather

## Executive Summary

This experiment evaluates next-day GPS prediction on extracted southbound migration paths without using weather features. Each model receives a compact movement/location/history window and predicts the next daily location. The best one-step GPS model in both fly-label setups is a triline LSTM with 7 days of context:

| Setup | Best one-step GPS model | Family | k | Mean km | Median km | P90 km | Migration50 mean km |
|---|---|---|---:|---:|---:|---:|---:|
| setup_30km | triline_lstm_2l_k7 | triline | 7 | 13.0181 | 1.6979 | 35.9378 | 104.2883 |
| setup_10km | triline_lstm_2l_k7 | triline | 7 | 12.7311 | 1.4469 | 37.5566 | 97.5540 |

The strongest overall pattern is that the triline models improve mean GPS error over persistence and direct regression, while direct regression remains more competitive on the largest migration moves. The best triline model also benefits from fly/no-fly gating: in `setup_30km`, the triline LSTM k=7 improves from 13.0181 km ungated mean error to 12.4526 km gated mean error; in `setup_10km`, it improves from 12.7311 km to 12.3069 km.

## Experiment Description

The task is one-step trajectory prediction. Given the previous `k` consecutive daily records for a bird path, the model predicts the next day's latitude and longitude. The experiment is intentionally no-weather: it uses only compact trajectory features, so the result measures how much can be learned from recent movement, location, seasonality, and bird identity alone.

Two label setups are evaluated:

| Setup | Fly label definition | Purpose |
|---|---:|---|
| setup_30km | `step_length_km > 30` | Matches the prior southbound compact setup. |
| setup_10km | `step_length_km > 10` | More sensitive fly/no-fly split that treats smaller movement days as flight. |

The GPS prediction target is the same in both setups. The setup threshold changes triline fly/no-fly labels and the stationary/fly-day error slices, so baseline and direct GPS predictions are numerically identical across setups, while triline training and fly metrics can differ.

## Data

Input file:

```text
data/filtered/dataset2_southbound_paths_jun_dec_lat30_50_min50_drop5.csv
```

Dataset summary:

| Field | Value |
|---|---:|
| Rows | 8,180 |
| Southbound paths | 82 |
| Source birds | 33 |
| Date range | 2013-07-30 to 2023-12-18 |
| Grouping column | `path_id` |
| Bird identity column | `source_individual_local_identifier` |
| Weather features | No |

Each training example is built inside a single `path_id`. Windows never cross path boundaries, which prevents mixing different bird-year migration segments. The split is chronological 80/20 over constructed windows.

Compact no-weather feature vector:

```text
lat_median
lon_median
delta_lat
delta_lon
step_length_km
heading_sin
heading_cos
doy_sin
doy_cos
stopover_duration_days
```

Window counts decrease as the context length grows:

| k | Train samples | Test samples |
|---:|---:|---:|
| 1 | 6,441 | 1,611 |
| 7 | 5,936 | 1,484 |
| 14 | 5,415 | 1,354 |
| 30 | 4,328 | 1,082 |

## Model Families

Baselines:

| Model | Description |
|---|---|
| persistence | Predicts the last observed latitude/longitude as the next location. |
| constant velocity | Adds the last observed daily delta to the last observed location. |

Direct regression models predict normalized next-day `[delta_lat, delta_lon]`, then add the predicted delta to the last input location.

| Model | Contexts |
|---|---|
| Direct MLP last-day | k=1 |
| Direct MLP flattened sequence | k=30 |
| Direct Transformer 2-layer | k=7, 14, 30 |

Triline models decompose movement into three predictions: fly logit, `log1p(distance_km)`, and heading `[sin, cos]`. GPS is reconstructed from the predicted distance and heading using the last input location. Primary GPS metrics use ungated distance reconstruction. Secondary `gated_` metrics set predicted distance to zero when fly probability is below the tuned threshold.

| Model | Contexts |
|---|---|
| Triline Transformer 2-layer | k=7, 14, 30 |
| Triline Transformer 3-layer | k=30 |
| Triline Transformer 4-layer | k=7, 14, 30 |
| Triline LSTM 2-layer | k=7, 14, 30 |
| Triline LinearAR full features | k=7, 14, 30 |
| Triline LinearAR delta-only | k=30 |

## Training Setup

| Setting | Value |
|---|---:|
| Seed | 42 |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 4 |
| Max epochs | 100 |
| Early stopping patience | 20 |
| Device | CUDA |
| Triline loss weights | fly 1.0, distance 1.0, direction 0.5 |
| BCE positive weight | Computed from train split, capped at 10.0 |

The full comparison contains 50 rows: 25 model rows for each setup. Per setup, this is 6 baseline rows, 5 direct rows, and 14 triline rows.

## Metric Guide

GPS metrics:

| Metric | Meaning |
|---|---|
| `mean_error_km` | Average haversine distance between predicted and true next-day GPS location. Lower is better. Sensitive to large migration misses. |
| `median_error_km` | Median haversine error. Lower is better. Represents typical daily behavior and is less affected by rare large misses. |
| `p90_error_km` | 90th percentile GPS error. Lower is better. Shows tail behavior for difficult days. |
| `p95_error_km` | 95th percentile GPS error. Lower is better. A stricter tail-risk metric. |
| `stationary_error_mean_km` | Mean GPS error on target days where true movement is at or below the setup fly threshold. |
| `fly_error_mean_km` | Mean GPS error on target days where true movement is above the setup fly threshold. |
| `migration50_error_mean_km` | Mean GPS error on target days with true movement above 50 km. This is setup-independent and isolates larger migration moves. |
| `gated_*` | Same GPS metrics after triline fly/no-fly gating. These are secondary metrics because the primary GPS comparison uses ungated reconstruction. |

Training and model metrics:

| Metric | Meaning |
|---|---|
| `final_delta_mse_norm` | Direct-model MSE on normalized `[delta_lat, delta_lon]`; useful within direct models but not directly comparable to GPS kilometers. |
| `best_test_loss` | Lowest validation/test objective used for checkpoint selection. For triline models this is the weighted multitask loss. |
| `final_test_loss` | Final evaluated triline multitask loss after loading the best checkpoint. |
| `params` | Trainable parameter count. |
| `best_epoch` | Epoch where the best checkpoint was saved. |
| `train_samples`, `test_samples` | Number of constructed windows in each split. |
| `train_fly_rate`, `test_fly_rate` | Fraction of fly-label positives in each split. |
| `runtime_seconds` | Wall-clock training and evaluation time for that model row. |

Fly/no-fly classification metrics:

| Metric | Meaning |
|---|---|
| `fly_threshold` | Tuned probability threshold selected from threshold sweep, primarily maximizing F1. |
| `fly_precision` | Among predicted fly days, the fraction that truly flew. Higher means fewer false fly alarms. |
| `fly_recall` | Among true fly days, the fraction detected by the model. Higher means fewer missed fly days. |
| `fly_f1` | Harmonic mean of precision and recall. Useful when both false positives and false negatives matter. |
| `fly_specificity` | Among true non-fly days, the fraction correctly classified as non-fly. |
| `fly_false_positive_rate` | Fraction of true non-fly days incorrectly predicted as fly. Lower is better. |
| `fly_roc_auc` | Ranking quality across all thresholds for separating fly from non-fly days. 0.5 is random, 1.0 is perfect. |
| `fly_pr_auc` | Precision-recall area under curve. Especially useful under class imbalance. |
| `fixed_0_5_*` | Same classification metrics using a fixed 0.5 probability threshold instead of the tuned threshold. |

## GPS Result Analysis

### Best Model By Family

| Setup | Family | Best model | k | Mean km | Median km | P90 km | Migration50 mean km |
|---|---|---|---:|---:|---:|---:|---:|
| setup_30km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 143.1997 |
| setup_30km | direct | direct_mlp_last_day_k1 | 1 | 14.5327 | 4.1566 | 38.7376 | 88.1687 |
| setup_30km | triline | triline_lstm_2l_k7 | 7 | 13.0181 | 1.6979 | 35.9378 | 104.2883 |
| setup_10km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 143.1997 |
| setup_10km | direct | direct_mlp_last_day_k1 | 1 | 14.5327 | 4.1566 | 38.7376 | 88.1687 |
| setup_10km | triline | triline_lstm_2l_k7 | 7 | 12.7311 | 1.4469 | 37.5566 | 97.5540 |

The best triline row wins on mean GPS error for both setups. Against the best persistence baseline, triline LSTM k=7 improves mean error by 1.9052 km in `setup_30km` and 2.1922 km in `setup_10km`. Against the best direct model, it improves mean error by 1.5145 km in `setup_30km` and 1.8015 km in `setup_10km`.

### Baseline Behavior

Persistence has the best baseline mean and median error. Its median error is very small because many days have little movement; simply staying at the previous location often works for typical stationary days. However, persistence performs poorly on large migration days: its best `migration50_error_mean_km` is 143.1997 km.

Constant velocity is worse than persistence overall. It has larger median and tail errors, suggesting that the last observed daily delta is noisy when naively extrapolated to the next day.

### Direct Regression Behavior

The best direct model is the last-day MLP with k=1. This is a notable result: adding longer sequence context through a transformer or flattened MLP did not improve direct GPS prediction in this run. Direct regression has worse median error than persistence and triline models, but it is the best family on large migration moves:

| Direct model | k | Mean km | Median km | Migration50 mean km |
|---|---:|---:|---:|---:|
| direct_mlp_last_day_k1 | 1 | 14.5327 | 4.1566 | 88.1687 |
| direct_transformer_2l_k14 | 14 | 14.6008 | 2.5634 | 85.0915 |
| direct_transformer_2l_k7 | 7 | 15.1298 | 4.4337 | 88.7208 |
| direct_transformer_2l_k30 | 30 | 15.3430 | 3.5340 | 89.9244 |
| direct_mlp_sequence_k30 | 30 | 17.9104 | 5.3146 | 93.8895 |

The direct transformer k=14 has the best direct `migration50_error_mean_km` at 85.0915 km, even though its overall mean is slightly worse than the direct MLP. This indicates that direct coordinate regression is comparatively better at large displacement days, while it pays for that with larger typical-day errors.

### Triline Behavior

The strongest triline rows are short-context models. The top five by mean GPS error are:

| Setup | Model | k | Mean km | Median km | Migration50 mean km | Fly precision | Fly recall | Fly F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| setup_30km | triline_lstm_2l_k7 | 7 | 13.0181 | 1.6979 | 104.2883 | 0.6994 | 0.7961 | 0.7446 |
| setup_30km | triline_transformer_4l_k7 | 7 | 13.0360 | 1.6254 | 99.0598 | 0.6800 | 0.7829 | 0.7278 |
| setup_30km | triline_transformer_2l_k7 | 7 | 13.1277 | 1.4882 | 100.8956 | 0.6757 | 0.8224 | 0.7418 |
| setup_10km | triline_lstm_2l_k7 | 7 | 12.7311 | 1.4469 | 97.5540 | 0.7778 | 0.6837 | 0.7277 |
| setup_10km | triline_transformer_4l_k7 | 7 | 13.3564 | 1.5094 | 109.1816 | 0.7349 | 0.7349 | 0.7349 |
| setup_10km | triline_transformer_2l_k7 | 7 | 13.4055 | 1.4817 | 110.5179 | 0.7327 | 0.7395 | 0.7361 |

Longer context does not help in this no-weather southbound setting. k=14 and k=30 reduce the number of usable windows and may include stale movement history. The best models mostly use k=7, suggesting recent short-term movement is more valuable than a longer flattened history for this dataset.

### Gated vs Ungated Triline GPS

The primary triline GPS metric is ungated because it evaluates the raw distance and heading reconstruction. Gating is still useful as a secondary analysis because it asks whether the fly/no-fly head can suppress unnecessary movement predictions.

| Setup | Model | k | Ungated mean km | Gated mean km | Ungated median km | Gated median km |
|---|---|---:|---:|---:|---:|---:|
| setup_30km | triline_lstm_2l_k7 | 7 | 13.0181 | 12.4526 | 1.6979 | 0.7990 |
| setup_10km | triline_lstm_2l_k7 | 7 | 12.7311 | 12.3069 | 1.4469 | 0.8089 |

Gating improves both mean and median errors for the best triline model. This supports the interpretation that the fly/no-fly head is not just an auxiliary task; it can help decide when to predict no movement.

## Fly/No-Fly Classification Results

Only triline models produce fly/no-fly classification outputs. The reported `fly_*` metrics use a tuned threshold from `threshold_sweep.csv`; fixed 0.5 metrics are also saved for calibration comparison.

### Best Classification Rows

| Setup | Model | k | Tuned threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Fixed 0.5 F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| setup_30km | triline_lstm_2l_k14 | 14 | 0.822 | 0.731 | 0.785 | 0.757 | 0.937 | 0.742 | 0.702 |
| setup_30km | triline_transformer_2l_k14 | 14 | 0.714 | 0.683 | 0.826 | 0.748 | 0.948 | 0.742 | 0.692 |
| setup_30km | triline_transformer_4l_k14 | 14 | 0.747 | 0.678 | 0.832 | 0.747 | 0.945 | 0.731 | 0.672 |
| setup_30km | triline_lstm_2l_k7 | 7 | 0.761 | 0.699 | 0.796 | 0.745 | 0.949 | 0.760 | 0.704 |
| setup_30km | triline_transformer_2l_k7 | 7 | 0.749 | 0.676 | 0.822 | 0.742 | 0.953 | 0.752 | 0.701 |
| setup_10km | triline_transformer_4l_k14 | 14 | 0.537 | 0.738 | 0.752 | 0.745 | 0.910 | 0.743 | 0.733 |
| setup_10km | triline_transformer_2l_k7 | 7 | 0.694 | 0.733 | 0.740 | 0.736 | 0.906 | 0.733 | 0.679 |
| setup_10km | triline_transformer_4l_k7 | 7 | 0.635 | 0.735 | 0.735 | 0.735 | 0.912 | 0.730 | 0.726 |
| setup_10km | triline_lstm_2l_k14 | 14 | 0.768 | 0.790 | 0.681 | 0.731 | 0.906 | 0.749 | 0.698 |
| setup_10km | triline_lstm_2l_k7 | 7 | 0.697 | 0.778 | 0.684 | 0.728 | 0.916 | 0.748 | 0.713 |

### Classification Interpretation

For `setup_30km`, the best F1 is 0.757 from `triline_lstm_2l_k14`. This model balances precision 0.731 and recall 0.785. The highest recall is 0.832 from `triline_transformer_4l_k14`, while the best ranking score is ROC-AUC 0.953 from `triline_transformer_2l_k7`.

For `setup_10km`, the best F1 is 0.745 from `triline_transformer_4l_k14`. The lower fly threshold creates more positive labels, and the best F1 row has a much lower tuned probability threshold, 0.537, than the best 30 km row. The best precision is 0.790 from `triline_lstm_2l_k14`, while the best recall among top rows is 0.752 from `triline_transformer_4l_k14`.

The tuned thresholds are consistently better than a fixed 0.5 threshold for the strongest 30 km rows, showing that probability calibration matters. In the 10 km setup, fixed 0.5 is closer to tuned performance for the best transformer 4L k=14 row, suggesting the 10 km labels are easier to align with raw model scores.

## Autoregressive Rollout Analysis

The rollout visualization evaluates a harder use case than one-step prediction. It gives the model the first 30 observed days, then repeatedly feeds back predicted positions to forecast the rest of a representative test path. This compounds errors and should not be read as the same metric as one-step test error.

| Setup | Path | Direct model | Direct mean rollout km | Direct final rollout km | Triline model | Triline mean rollout km | Triline final rollout km |
|---|---|---|---:|---:|---|---:|---:|
| setup_30km | Sierit_DER_AN858_eobs2561__2020__southbound_01 | direct_mlp_last_day_k1 | 579.5691 | 361.7430 | triline_lstm_2l_k7 | 204.1434 | 344.6324 |
| setup_10km | Europa_DER_A1A26_eobs_4004__2021__southbound_01 | direct_mlp_last_day_k1 | 975.4728 | 1395.3242 | triline_lstm_2l_k7 | 172.8514 | 169.4631 |

The rollout results show substantial compounding error, especially for the direct model. The triline LSTM is much more stable in these selected rollouts, likely because its distance/heading representation and fly gating discourage runaway coordinate drift. These plots are useful qualitatively, but the one-step test metrics remain the primary quantitative comparison.

Rollout files:

```text
setup_30km/rollout/rollout_predictions.csv
setup_30km/rollout/rollout_plot.svg
setup_30km/rollout/rollout_summary.json
setup_10km/rollout/rollout_predictions.csv
setup_10km/rollout/rollout_plot.svg
setup_10km/rollout/rollout_summary.json
```

## Main Takeaways

1. Triline LSTM k=7 is the best one-step GPS model in both threshold setups.
2. Persistence remains a strong stationary-day baseline, but it fails badly on large migration moves.
3. Direct regression is worse on mean and median error but has the best large-migration error among learned families.
4. Short context is consistently best for GPS prediction; k=30 is generally weaker in this southbound no-weather experiment.
5. Fly/no-fly prediction is strong enough to help GPS reconstruction through gating.
6. Classification-optimal models are not always GPS-optimal models. k=14 often improves fly/no-fly F1, while k=7 gives the best GPS mean error.
7. Autoregressive rollouts are much harder than one-step prediction and expose drift, especially for direct coordinate regression.

## Output Index

| File | Contents |
|---|---|
| `comparison_summary.csv` | All model rows across both setups. |
| `baseline_summary.csv` | Persistence and constant-velocity GPS baselines. |
| `direct_summary.csv` | Direct regression model metrics. |
| `triline_summary.csv` | Triline GPS and fly/no-fly metrics. |
| `setup_30km/...` | Per-model checkpoints, logs, predictions, and metrics for the 30 km fly threshold. |
| `setup_10km/...` | Per-model checkpoints, logs, predictions, and metrics for the 10 km fly threshold. |
| `run_config.json` | Training configuration and model matrix. |
| `data_summary.json` | Dataset and feature metadata. |
