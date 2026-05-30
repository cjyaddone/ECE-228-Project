# Path Experiment With Weather

## Setup

- Dataset: southbound path segments.
- Features: compact 18-feature inputs with 10 path features and 8 weather features.
- Weather match: exact bird/date join with nearest same-bird date fallback.
- Split: chronological 80/20 within constructed windows.
- Identity: source bird ID embedding for neural models.

## Best Mean GPS Error By Setup

| Setup | Model | Family | k | Mean km | Median km | Fly recall | Params |
|---|---|---|---:|---:|---:|---:|---:|
| setup_10km | triline_linear_ar_full_k7 | triline | 7 | 13.8872 | 1.8635 | 0.6558 | 47460 |
| setup_30km | triline_linear_ar_full_k7 | triline | 7 | 13.8701 | 1.8942 | 0.8158 | 47460 |

## Weather vs No-Weather

| Setup | Model | Family | k | Weather Mean km | No-Weather Mean km | Delta km | Improved |
|---|---|---|---:|---:|---:|---:|---|
| setup_10km | triline_linear_ar_full_k7 | triline | 7 | 13.8872 | 13.7823 | 0.1049 | no |
| setup_10km | triline_lstm_2l_k7 | triline | 7 | 14.2622 | 12.7311 | 1.5310 | no |
| setup_10km | triline_transformer_2l_k7 | triline | 7 | 14.7120 | 13.4055 | 1.3065 | no |
| setup_10km | persistence | baseline | 7 | 14.9233 | 14.9233 | 0.0000 | no |
| setup_10km | triline_linear_ar_delta_k30 | triline | 30 | 16.0361 | 15.4942 | 0.5419 | no |
| setup_10km | persistence | baseline | 30 | 16.3394 | 16.3394 | 0.0000 | no |
| setup_10km | triline_transformer_3l_k30 | triline | 30 | 16.5118 | 14.6729 | 1.8389 | no |
| setup_10km | triline_transformer_4l_k30 | triline | 30 | 16.9597 | 14.7337 | 2.2261 | no |
| setup_10km | const_velocity | baseline | 7 | 17.8171 | 17.8171 | 0.0000 | no |
| setup_10km | direct_mlp_last_day_k1 | direct | 1 | 18.2882 | 14.5327 | 3.7556 | no |
| setup_10km | direct_transformer_2l_k7 | direct | 7 | 18.8140 | 15.1298 | 3.6841 | no |
| setup_10km | const_velocity | baseline | 30 | 19.5423 | 19.5423 | 0.0000 | no |
| setup_10km | direct_mlp_sequence_k30 | direct | 30 | 19.7059 | 17.9104 | 1.7955 | no |
| setup_30km | triline_linear_ar_full_k7 | triline | 7 | 13.8701 | 13.7011 | 0.1690 | no |
| setup_30km | triline_lstm_2l_k7 | triline | 7 | 14.1729 | 13.0181 | 1.1548 | no |
| setup_30km | triline_transformer_2l_k7 | triline | 7 | 14.8670 | 13.1277 | 1.7393 | no |
| setup_30km | persistence | baseline | 7 | 14.9233 | 14.9233 | 0.0000 | no |
| setup_30km | triline_linear_ar_delta_k30 | triline | 30 | 16.0662 | 15.8545 | 0.2117 | no |
| setup_30km | persistence | baseline | 30 | 16.3394 | 16.3394 | 0.0000 | no |
| setup_30km | triline_transformer_3l_k30 | triline | 30 | 16.6029 | 14.6190 | 1.9839 | no |
| setup_30km | triline_transformer_4l_k30 | triline | 30 | 16.8017 | 14.4831 | 2.3186 | no |
| setup_30km | const_velocity | baseline | 7 | 17.8171 | 17.8171 | 0.0000 | no |
| setup_30km | direct_mlp_last_day_k1 | direct | 1 | 18.2882 | 14.5327 | 3.7556 | no |
| setup_30km | direct_transformer_2l_k7 | direct | 7 | 18.8140 | 15.1298 | 3.6841 | no |
| setup_30km | const_velocity | baseline | 30 | 19.5423 | 19.5423 | 0.0000 | no |
| setup_30km | direct_mlp_sequence_k30 | direct | 30 | 19.7059 | 17.9104 | 1.7955 | no |

## Full Comparison

| Setup | Family | Model | k | Mean km | Median km | P90 km | Fly Mean km | Migration50 Mean km |
|---|---|---|---:|---:|---:|---:|---:|---:|
| setup_10km | baseline | const_velocity | 7 | 17.8171 | 1.7637 | 54.1154 | 89.9617 | 116.0993 |
| setup_10km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 93.6197 | 143.1997 |
| setup_10km | baseline | const_velocity | 30 | 19.5423 | 2.2817 | 61.8870 | 92.8597 | 119.9864 |
| setup_10km | baseline | persistence | 30 | 16.3394 | 0.8083 | 40.9240 | 94.7498 | 146.6291 |
| setup_10km | direct | direct_mlp_last_day_k1 | 1 | 18.2882 | 7.1738 | 40.5164 | 70.6014 | 94.4186 |
| setup_10km | direct | direct_transformer_2l_k7 | 7 | 18.8140 | 8.4028 | 48.7031 | 69.6287 | 88.2469 |
| setup_10km | direct | direct_mlp_sequence_k30 | 30 | 19.7059 | 7.1653 | 47.7478 | 71.3130 | 100.9515 |
| setup_10km | triline | triline_linear_ar_full_k7 | 7 | 13.8872 | 1.8635 | 38.0538 | 77.2373 | 111.4757 |
| setup_10km | triline | triline_lstm_2l_k7 | 7 | 14.2622 | 2.1520 | 32.3887 | 80.3346 | 120.9414 |
| setup_10km | triline | triline_transformer_2l_k7 | 7 | 14.7120 | 2.1181 | 29.4412 | 83.5303 | 127.0801 |
| setup_10km | triline | triline_linear_ar_delta_k30 | 30 | 16.0361 | 3.0323 | 41.6696 | 78.8870 | 117.8643 |
| setup_10km | triline | triline_transformer_3l_k30 | 30 | 16.5118 | 2.5656 | 38.1198 | 88.1501 | 136.7208 |
| setup_10km | triline | triline_transformer_4l_k30 | 30 | 16.9597 | 3.0809 | 38.8273 | 88.1239 | 136.5980 |
| setup_30km | baseline | const_velocity | 7 | 17.8171 | 1.7637 | 54.1154 | 109.1717 | 116.0993 |
| setup_30km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 125.0660 | 143.1997 |
| setup_30km | baseline | const_velocity | 30 | 19.5423 | 2.2817 | 61.8870 | 113.1596 | 119.9864 |
| setup_30km | baseline | persistence | 30 | 16.3394 | 0.8083 | 40.9240 | 128.0119 | 146.6291 |
| setup_30km | direct | direct_mlp_last_day_k1 | 1 | 18.2882 | 7.1738 | 40.5164 | 86.8858 | 94.4186 |
| setup_30km | direct | direct_transformer_2l_k7 | 7 | 18.8140 | 8.4028 | 48.7031 | 83.0589 | 88.2469 |
| setup_30km | direct | direct_mlp_sequence_k30 | 30 | 19.7059 | 7.1653 | 47.7478 | 90.8990 | 100.9515 |
| setup_30km | triline | triline_linear_ar_full_k7 | 7 | 13.8701 | 1.8942 | 37.4926 | 99.3807 | 111.3568 |
| setup_30km | triline | triline_lstm_2l_k7 | 7 | 14.1729 | 1.9222 | 32.5552 | 106.1838 | 121.6836 |
| setup_30km | triline | triline_transformer_2l_k7 | 7 | 14.8670 | 1.7845 | 30.6064 | 116.3924 | 133.4667 |
| setup_30km | triline | triline_linear_ar_delta_k30 | 30 | 16.0662 | 3.0768 | 41.4419 | 103.8708 | 118.1153 |
| setup_30km | triline | triline_transformer_3l_k30 | 30 | 16.6029 | 2.6478 | 37.1613 | 119.8244 | 137.5466 |
| setup_30km | triline | triline_transformer_4l_k30 | 30 | 16.8017 | 2.6569 | 37.6241 | 121.1356 | 138.9083 |

## Notes

- Triline rows report ungated GPS reconstruction as primary metrics.
- Gated GPS metrics are stored in each triline `metrics.json` and the CSV summaries with `gated_` prefixes.
- Direct model rows repeat per setup because setup-specific fly thresholds change stratified error slices.
- Autoregressive rollout uses actual future daily weather as exogenous forecast/oracle weather.
