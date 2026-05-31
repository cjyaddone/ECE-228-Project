# DistanceWeightTune

## Setup

- Dataset: southbound path segments.
- Features: compact 18-feature inputs with 10 path features and 8 weather features.
- Change under test: triline distance and direction losses are weighted by true target step distance.
- Distance weights: 0-5 km = 1, >5-30 km = 2, >30-100 km = 4, >100 km = 8.
- Direct models and baselines are unchanged controls.
- Weather match: exact bird/date join with nearest same-bird date fallback.
- Split: chronological 80/20 within constructed windows.
- Identity: source bird ID embedding for neural models.

## Best Mean GPS Error By Setup

| Setup | Model | Family | k | Mean km | Median km | Fly recall | Params |
|---|---|---|---:|---:|---:|---:|---:|
| setup_10km | persistence | baseline | 7 | 14.9233 | 0.7386 |  | 0 |
| setup_30km | persistence | baseline | 7 | 14.9233 | 0.7386 |  | 0 |

## Distance-Weighted vs Weather Baseline

| Setup | Model | Family | k | Weighted Mean km | Weather Mean km | Mean Delta km | Migration50 Delta km | Improved |
|---|---|---|---:|---:|---:|---:|---:|---|
| setup_10km | persistence | baseline | 7 | 14.9233 | 14.9233 | 0.0000 | 0.0000 | no |
| setup_10km | triline_lstm_2l_k7 | triline | 7 | 15.7962 | 13.3131 | 2.4832 | -12.3124 | no |
| setup_10km | persistence | baseline | 30 | 16.3394 | 16.3394 | 0.0000 | 0.0000 | no |
| setup_10km | const_velocity | baseline | 7 | 17.8171 | 17.8171 | 0.0000 | -0.0000 | no |
| setup_10km | triline_transformer_2l_k7 | triline | 7 | 18.2508 | 13.2526 | 4.9982 | -11.6260 | no |
| setup_10km | direct_mlp_last_day_k1 | direct | 1 | 18.2882 | 14.3415 | 3.9467 | 3.9556 | no |
| setup_10km | triline_transformer_4l_k30 | triline | 30 | 18.6270 | 14.8265 | 3.8005 | 13.1903 | no |
| setup_10km | direct_transformer_2l_k7 | direct | 7 | 18.8140 | 14.2745 | 4.5394 | 2.6136 | no |
| setup_10km | triline_linear_ar_full_k7 | triline | 7 | 19.1629 | 14.4053 | 4.7576 | 10.8218 | no |
| setup_10km | triline_transformer_3l_k30 | triline | 30 | 19.3775 | 14.7181 | 4.6594 | -2.1792 | no |
| setup_10km | const_velocity | baseline | 30 | 19.5423 | 19.5423 | 0.0000 | 0.0000 | no |
| setup_10km | direct_mlp_sequence_k30 | direct | 30 | 19.7059 | 18.5861 | 1.1199 | 5.0373 | no |
| setup_10km | triline_linear_ar_delta_k30 | triline | 30 | 27.0356 | 15.4942 | 11.5414 | 25.8891 | no |
| setup_30km | persistence | baseline | 7 | 14.9233 | 14.9233 | 0.0000 | 0.0000 | no |
| setup_30km | triline_lstm_2l_k7 | triline | 7 | 15.2036 | 13.1441 | 2.0595 | -10.4223 | no |
| setup_30km | persistence | baseline | 30 | 16.3394 | 16.3394 | 0.0000 | 0.0000 | no |
| setup_30km | const_velocity | baseline | 7 | 17.8171 | 17.8171 | 0.0000 | -0.0000 | no |
| setup_30km | triline_transformer_2l_k7 | triline | 7 | 18.1164 | 13.5499 | 4.5666 | -16.0623 | no |
| setup_30km | direct_mlp_last_day_k1 | direct | 1 | 18.2882 | 14.3415 | 3.9467 | 3.9556 | no |
| setup_30km | direct_transformer_2l_k7 | direct | 7 | 18.8140 | 14.2745 | 4.5394 | 2.6136 | no |
| setup_30km | triline_linear_ar_full_k7 | triline | 7 | 19.2689 | 14.0921 | 5.1768 | 13.0332 | no |
| setup_30km | const_velocity | baseline | 30 | 19.5423 | 19.5423 | 0.0000 | 0.0000 | no |
| setup_30km | triline_transformer_3l_k30 | triline | 30 | 19.6214 | 15.0558 | 4.5656 | -7.3346 | no |
| setup_30km | direct_mlp_sequence_k30 | direct | 30 | 19.7059 | 18.5861 | 1.1199 | 5.0373 | no |
| setup_30km | triline_transformer_4l_k30 | triline | 30 | 20.3176 | 14.5275 | 5.7901 | 25.7924 | no |
| setup_30km | triline_linear_ar_delta_k30 | triline | 30 | 27.2015 | 15.8545 | 11.3470 | 23.7842 | no |

Distance-weighted triline rows improving migration50 error: 6/12.

## Full Comparison

| Setup | Family | Model | k | Mean km | Median km | P90 km | Fly Mean km | Migration50 Mean km | Migration100 Mean km |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| setup_10km | baseline | const_velocity | 7 | 17.8171 | 1.7637 | 54.1154 | 89.9617 | 116.0993 | 138.2874 |
| setup_10km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 93.6197 | 143.1997 | 184.7018 |
| setup_10km | baseline | const_velocity | 30 | 19.5423 | 2.2817 | 61.8870 | 92.8597 | 119.9864 | 141.6438 |
| setup_10km | baseline | persistence | 30 | 16.3394 | 0.8083 | 40.9240 | 94.7498 | 146.6291 | 189.8530 |
| setup_10km | direct | direct_mlp_last_day_k1 | 1 | 18.2882 | 7.1738 | 40.5164 | 70.6014 | 94.4186 | 120.9770 |
| setup_10km | direct | direct_transformer_2l_k7 | 7 | 18.8140 | 8.4028 | 48.7031 | 69.6287 | 88.2469 | 105.8861 |
| setup_10km | direct | direct_mlp_sequence_k30 | 30 | 19.7059 | 7.1653 | 47.7478 | 71.3130 | 100.9515 | 133.5046 |
| setup_10km | triline | triline_linear_ar_full_k7 | 7 | 19.1629 | 3.9370 | 49.7076 | 89.4933 | 114.0026 | 133.0469 |
| setup_10km | triline | triline_lstm_2l_k7 | 7 | 15.7962 | 3.0880 | 49.9857 | 73.1125 | 94.0457 | 113.9306 |
| setup_10km | triline | triline_transformer_2l_k7 | 7 | 18.2508 | 3.9809 | 63.2212 | 69.6539 | 92.4671 | 120.5016 |
| setup_10km | triline | triline_linear_ar_delta_k30 | 30 | 27.0356 | 9.4331 | 69.4802 | 104.5692 | 137.4928 | 162.9721 |
| setup_10km | triline | triline_transformer_3l_k30 | 30 | 19.3775 | 6.3666 | 50.2565 | 74.6220 | 107.2198 | 142.3964 |
| setup_10km | triline | triline_transformer_4l_k30 | 30 | 18.6270 | 5.1377 | 37.8272 | 82.0853 | 125.9337 | 165.8591 |
| setup_30km | baseline | const_velocity | 7 | 17.8171 | 1.7637 | 54.1154 | 109.1717 | 116.0993 | 138.2874 |
| setup_30km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 125.0660 | 143.1997 | 184.7018 |
| setup_30km | baseline | const_velocity | 30 | 19.5423 | 2.2817 | 61.8870 | 113.1596 | 119.9864 | 141.6438 |
| setup_30km | baseline | persistence | 30 | 16.3394 | 0.8083 | 40.9240 | 128.0119 | 146.6291 | 189.8530 |
| setup_30km | direct | direct_mlp_last_day_k1 | 1 | 18.2882 | 7.1738 | 40.5164 | 86.8858 | 94.4186 | 120.9770 |
| setup_30km | direct | direct_transformer_2l_k7 | 7 | 18.8140 | 8.4028 | 48.7031 | 83.0589 | 88.2469 | 105.8861 |
| setup_30km | direct | direct_mlp_sequence_k30 | 30 | 19.7059 | 7.1653 | 47.7478 | 90.8990 | 100.9515 | 133.5046 |
| setup_30km | triline | triline_linear_ar_full_k7 | 7 | 19.2689 | 3.9950 | 49.7210 | 108.9918 | 114.3530 | 133.0595 |
| setup_30km | triline | triline_lstm_2l_k7 | 7 | 15.2036 | 2.7072 | 49.4370 | 87.7755 | 94.3457 | 116.6218 |
| setup_30km | triline | triline_transformer_2l_k7 | 7 | 18.1164 | 3.4992 | 66.0754 | 84.5307 | 91.8645 | 119.6570 |
| setup_30km | triline | triline_linear_ar_delta_k30 | 30 | 27.2015 | 9.5935 | 71.4388 | 129.7436 | 137.7067 | 163.5791 |
| setup_30km | triline | triline_transformer_3l_k30 | 30 | 19.6214 | 6.7517 | 48.7413 | 96.8297 | 108.4135 | 144.5481 |
| setup_30km | triline | triline_transformer_4l_k30 | 30 | 20.3176 | 5.6066 | 49.0589 | 110.1455 | 126.1058 | 166.2548 |

## Training Logs

- Summarized 18 training logs in `training_log_summary.csv`.
- Weighted and unweighted triline loss columns are included for auditability.

## Rollout Visualization

- Interactive rollout explorer: `rollout_all/index.html`.
- Rollout paths: 40; skipped paths: 0.
- Rollout comparison rows against weather baseline: 40.

## Notes

- Triline rows report ungated GPS reconstruction as primary metrics.
- Gated GPS metrics are stored in each triline `metrics.json` and the CSV summaries with `gated_` prefixes.
- Direct model rows repeat per setup because setup-specific fly thresholds change stratified error slices.
- Autoregressive rollout uses actual future daily weather as exogenous forecast/oracle weather.
- Lower delta values in comparison files mean distance weighting improved over the original weather run.
