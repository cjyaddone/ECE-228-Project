# Path Experiment No Weather

## Setup

- Dataset: southbound path segments.
- Features: compact 10-feature no-weather inputs.
- Split: chronological 80/20 within constructed windows.
- Identity: source bird ID embedding for neural models.

## Best Mean GPS Error By Setup

| Setup | Model | Family | k | Mean km | Median km | Fly recall | Params |
|---|---|---|---:|---:|---:|---:|---:|
| setup_10km | triline_linear_ar_full_k7 | triline | 7 | 13.7249 | 1.8463 | 0.6744 | 40292 |
| setup_30km | triline_linear_ar_full_k7 | triline | 7 | 13.7590 | 1.9123 | 0.7895 | 40292 |

## Full Comparison

| Setup | Family | Model | k | Mean km | Median km | P90 km | Fly Mean km | Migration50 Mean km |
|---|---|---|---:|---:|---:|---:|---:|---:|
| setup_10km | baseline | const_velocity | 7 | 17.8171 | 1.7637 | 54.1154 | 89.9617 | 116.0993 |
| setup_10km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 93.6197 | 143.1997 |
| setup_10km | baseline | const_velocity | 30 | 19.5423 | 2.2817 | 61.8870 | 92.8597 | 119.9864 |
| setup_10km | baseline | persistence | 30 | 16.3394 | 0.8083 | 40.9240 | 94.7498 | 146.6291 |
| setup_10km | direct | direct_mlp_last_day_k1 | 1 | 17.1181 | 5.9027 | 40.8472 | 70.2044 | 95.0372 |
| setup_10km | direct | direct_transformer_2l_k7 | 7 | 17.0547 | 6.8859 | 42.7481 | 67.3399 | 89.1614 |
| setup_10km | direct | direct_mlp_sequence_k30 | 30 | 20.1256 | 7.2313 | 49.3483 | 70.4762 | 99.7134 |
| setup_10km | triline | triline_linear_ar_full_k7 | 7 | 13.7249 | 1.8463 | 32.7221 | 76.8112 | 112.5288 |
| setup_10km | triline | triline_lstm_2l_k7 | 7 | 13.9895 | 2.2573 | 32.7313 | 76.8810 | 113.8426 |
| setup_10km | triline | triline_transformer_2l_k7 | 7 | 14.4190 | 2.2959 | 32.7251 | 80.2898 | 121.0994 |
| setup_10km | triline | triline_linear_ar_delta_k30 | 30 | 16.0361 | 3.0323 | 41.6696 | 78.8870 | 117.8643 |
| setup_10km | triline | triline_transformer_3l_k30 | 30 | 16.8312 | 2.5945 | 37.6510 | 90.5445 | 140.5156 |
| setup_10km | triline | triline_transformer_4l_k30 | 30 | 16.6880 | 2.5748 | 37.8524 | 89.7000 | 139.1969 |
| setup_30km | baseline | const_velocity | 7 | 17.8171 | 1.7637 | 54.1154 | 109.1717 | 116.0993 |
| setup_30km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 125.0660 | 143.1997 |
| setup_30km | baseline | const_velocity | 30 | 19.5423 | 2.2817 | 61.8870 | 113.1596 | 119.9864 |
| setup_30km | baseline | persistence | 30 | 16.3394 | 0.8083 | 40.9240 | 128.0119 | 146.6291 |
| setup_30km | direct | direct_mlp_last_day_k1 | 1 | 17.1181 | 5.9027 | 40.8472 | 87.2932 | 95.0372 |
| setup_30km | direct | direct_transformer_2l_k7 | 7 | 17.0547 | 6.8859 | 42.7481 | 82.4539 | 89.1614 |
| setup_30km | direct | direct_mlp_sequence_k30 | 30 | 20.1256 | 7.2313 | 49.3483 | 89.6230 | 99.7134 |
| setup_30km | triline | triline_linear_ar_full_k7 | 7 | 13.7590 | 1.9123 | 32.7864 | 100.0182 | 112.7421 |
| setup_30km | triline | triline_lstm_2l_k7 | 7 | 13.8151 | 2.0154 | 33.0666 | 100.2662 | 114.3542 |
| setup_30km | triline | triline_transformer_2l_k7 | 7 | 14.5441 | 2.1717 | 31.9140 | 109.6918 | 125.7208 |
| setup_30km | triline | triline_linear_ar_delta_k30 | 30 | 16.0662 | 3.0768 | 41.4419 | 103.8708 | 118.1153 |
| setup_30km | triline | triline_transformer_3l_k30 | 30 | 16.8666 | 2.6039 | 36.1450 | 122.6743 | 140.7218 |
| setup_30km | triline | triline_transformer_4l_k30 | 30 | 16.6035 | 2.2580 | 38.0897 | 122.7616 | 140.7981 |

## Notes

- Triline rows report ungated GPS reconstruction as primary metrics.
- Gated GPS metrics are stored in each triline `metrics.json` and the CSV summaries with `gated_` prefixes.
- Direct model rows repeat per setup because setup-specific fly thresholds change stratified error slices.
