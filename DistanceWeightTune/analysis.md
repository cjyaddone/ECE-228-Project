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
| setup_10km | triline_transformer_4l_k7 | triline | 7 | 14.2455 | 2.5724 | 0.6419 | 822662 |
| setup_30km | direct_transformer_2l_k7 | direct | 7 | 14.2745 | 3.3934 |  | 408964 |

## Distance-Weighted vs Weather Baseline

| Setup | Model | Family | k | Weighted Mean km | Weather Mean km | Mean Delta km | Migration50 Delta km | Improved |
|---|---|---|---:|---:|---:|---:|---:|---|
| setup_10km | triline_transformer_4l_k7 | triline | 7 | 14.2455 | 13.5762 | 0.6694 | -11.6310 | no |
| setup_10km | direct_transformer_2l_k7 | direct | 7 | 14.2745 | 14.2745 | 0.0000 | 0.0000 | no |
| setup_10km | direct_mlp_last_day_k1 | direct | 1 | 14.3415 | 14.3415 | 0.0000 | 0.0000 | no |
| setup_10km | triline_transformer_2l_k7 | triline | 7 | 14.4309 | 13.2526 | 1.1783 | -12.8170 | no |
| setup_10km | triline_lstm_2l_k14 | triline | 14 | 14.5161 | 13.9571 | 0.5591 | -14.9058 | no |
| setup_10km | triline_lstm_2l_k7 | triline | 7 | 14.5817 | 13.3131 | 1.2686 | -22.3615 | no |
| setup_10km | persistence | baseline | 7 | 14.9233 | 14.9233 | 0.0000 | 0.0000 | no |
| setup_10km | triline_transformer_2l_k14 | triline | 14 | 15.3320 | 13.9879 | 1.3441 | -16.0700 | no |
| setup_10km | persistence | baseline | 14 | 15.8945 | 15.8945 | 0.0000 | 0.0000 | no |
| setup_10km | triline_transformer_4l_k14 | triline | 14 | 15.9379 | 13.7664 | 2.1714 | -13.6567 | no |
| setup_10km | triline_transformer_4l_k30 | triline | 30 | 16.0207 | 14.8265 | 1.1942 | -25.9535 | no |
| setup_10km | triline_lstm_2l_k30 | triline | 30 | 16.1421 | 14.5792 | 1.5629 | -24.3519 | no |
| setup_10km | triline_transformer_3l_k30 | triline | 30 | 16.2639 | 14.7181 | 1.5458 | -21.5667 | no |
| setup_10km | direct_transformer_2l_k30 | direct | 30 | 16.3260 | 16.3260 | 0.0000 | 0.0000 | no |
| setup_10km | persistence | baseline | 30 | 16.3394 | 16.3394 | 0.0000 | 0.0000 | no |
| setup_10km | triline_linear_ar_full_k7 | triline | 7 | 16.9027 | 14.4053 | 2.4974 | -7.6013 | no |
| setup_10km | triline_transformer_2l_k30 | triline | 30 | 16.9188 | 14.8153 | 2.1034 | -15.6604 | no |
| setup_10km | const_velocity | baseline | 7 | 17.8171 | 17.8171 | 0.0000 | 0.0000 | no |
| setup_10km | direct_transformer_2l_k14 | direct | 14 | 17.9979 | 17.9979 | 0.0000 | 0.0000 | no |
| setup_10km | triline_linear_ar_full_k14 | triline | 14 | 18.5034 | 14.6606 | 3.8428 | -3.5318 | no |
| setup_10km | direct_mlp_sequence_k30 | direct | 30 | 18.5861 | 18.5861 | 0.0000 | 0.0000 | no |
| setup_10km | const_velocity | baseline | 14 | 18.6991 | 18.6991 | 0.0000 | 0.0000 | no |
| setup_10km | const_velocity | baseline | 30 | 19.5423 | 19.5423 | 0.0000 | 0.0000 | no |
| setup_10km | triline_linear_ar_delta_k30 | triline | 30 | 19.6802 | 15.4942 | 4.1860 | -4.6797 | no |
| setup_10km | triline_linear_ar_full_k30 | triline | 30 | 20.0783 | 15.9306 | 4.1477 | -3.8627 | no |
| setup_30km | direct_transformer_2l_k7 | direct | 7 | 14.2745 | 14.2745 | 0.0000 | 0.0000 | no |
| setup_30km | direct_mlp_last_day_k1 | direct | 1 | 14.3415 | 14.3415 | 0.0000 | 0.0000 | no |
| setup_30km | triline_transformer_2l_k7 | triline | 7 | 14.6170 | 13.5499 | 1.0672 | -18.5950 | no |
| setup_30km | triline_lstm_2l_k7 | triline | 7 | 14.6405 | 13.1441 | 1.4964 | -19.7900 | no |
| setup_30km | persistence | baseline | 7 | 14.9233 | 14.9233 | 0.0000 | 0.0000 | no |
| setup_30km | triline_transformer_4l_k7 | triline | 7 | 15.1976 | 13.4557 | 1.7419 | -28.0750 | no |
| setup_30km | triline_transformer_2l_k14 | triline | 14 | 15.5288 | 14.1886 | 1.3402 | -19.0199 | no |
| setup_30km | triline_lstm_2l_k14 | triline | 14 | 15.5922 | 13.8481 | 1.7441 | -17.9656 | no |
| setup_30km | triline_transformer_2l_k30 | triline | 30 | 15.7267 | 15.0642 | 0.6626 | -23.6904 | no |
| setup_30km | triline_transformer_4l_k30 | triline | 30 | 15.7970 | 14.5275 | 1.2695 | -10.6804 | no |
| setup_30km | persistence | baseline | 14 | 15.8945 | 15.8945 | 0.0000 | 0.0000 | no |
| setup_30km | triline_lstm_2l_k30 | triline | 30 | 15.9201 | 14.8131 | 1.1070 | -26.0222 | no |
| setup_30km | triline_transformer_4l_k14 | triline | 14 | 16.1159 | 13.8695 | 2.2464 | -20.5585 | no |
| setup_30km | direct_transformer_2l_k30 | direct | 30 | 16.3260 | 16.3260 | 0.0000 | 0.0000 | no |
| setup_30km | persistence | baseline | 30 | 16.3394 | 16.3394 | 0.0000 | 0.0000 | no |
| setup_30km | triline_transformer_3l_k30 | triline | 30 | 16.4146 | 15.0558 | 1.3588 | -28.6004 | no |
| setup_30km | triline_linear_ar_full_k7 | triline | 7 | 16.7314 | 14.0921 | 2.6392 | -5.7749 | no |
| setup_30km | const_velocity | baseline | 7 | 17.8171 | 17.8171 | 0.0000 | 0.0000 | no |
| setup_30km | direct_transformer_2l_k14 | direct | 14 | 17.9979 | 17.9979 | 0.0000 | 0.0000 | no |
| setup_30km | direct_mlp_sequence_k30 | direct | 30 | 18.5861 | 18.5861 | 0.0000 | 0.0000 | no |
| setup_30km | triline_linear_ar_full_k14 | triline | 14 | 18.6162 | 14.7038 | 3.9124 | -3.1148 | no |
| setup_30km | const_velocity | baseline | 14 | 18.6991 | 18.6991 | 0.0000 | 0.0000 | no |
| setup_30km | triline_linear_ar_full_k30 | triline | 30 | 19.3851 | 15.8696 | 3.5155 | -5.4428 | no |
| setup_30km | const_velocity | baseline | 30 | 19.5423 | 19.5423 | 0.0000 | 0.0000 | no |
| setup_30km | triline_linear_ar_delta_k30 | triline | 30 | 19.8905 | 15.8545 | 4.0359 | -4.6449 | no |

Distance-weighted triline rows improving migration50 error: 28/28.

## Full Comparison

| Setup | Family | Model | k | Mean km | Median km | P90 km | Fly Mean km | Migration50 Mean km | Migration100 Mean km |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| setup_10km | baseline | const_velocity | 7 | 17.8171 | 1.7637 | 54.1154 | 89.9617 | 116.0993 | 138.2874 |
| setup_10km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 93.6197 | 143.1997 | 184.7018 |
| setup_10km | baseline | const_velocity | 14 | 18.6991 | 1.7985 | 59.6849 | 90.1523 | 115.9770 | 138.6733 |
| setup_10km | baseline | persistence | 14 | 15.8945 | 0.7255 | 42.2899 | 93.9337 | 142.8895 | 185.2921 |
| setup_10km | baseline | const_velocity | 30 | 19.5423 | 2.2817 | 61.8870 | 92.8597 | 119.9864 | 141.6438 |
| setup_10km | baseline | persistence | 30 | 16.3394 | 0.8083 | 40.9240 | 94.7498 | 146.6291 | 189.8530 |
| setup_10km | direct | direct_mlp_last_day_k1 | 1 | 14.3415 | 3.7053 | 41.4564 | 64.5116 | 90.4630 | 109.2115 |
| setup_10km | direct | direct_transformer_2l_k7 | 7 | 14.2745 | 3.3934 | 42.7128 | 65.4065 | 85.6334 | 96.5770 |
| setup_10km | direct | direct_transformer_2l_k14 | 14 | 17.9979 | 5.6901 | 51.8707 | 65.3734 | 83.4258 | 99.7445 |
| setup_10km | direct | direct_mlp_sequence_k30 | 30 | 18.5861 | 5.9757 | 53.4965 | 71.2272 | 95.9142 | 118.3885 |
| setup_10km | direct | direct_transformer_2l_k30 | 30 | 16.3260 | 4.9817 | 46.2378 | 66.7362 | 91.9388 | 111.4150 |
| setup_10km | triline | triline_linear_ar_full_k7 | 7 | 16.9027 | 3.5918 | 55.0946 | 76.8103 | 95.5795 | 106.6342 |
| setup_10km | triline | triline_lstm_2l_k7 | 7 | 14.5817 | 2.5040 | 48.4289 | 68.0144 | 83.9965 | 100.9410 |
| setup_10km | triline | triline_transformer_2l_k7 | 7 | 14.4309 | 2.2093 | 46.9127 | 68.2472 | 91.2762 | 120.4670 |
| setup_10km | triline | triline_transformer_4l_k7 | 7 | 14.2455 | 2.5724 | 45.1202 | 67.2167 | 91.8968 | 121.3651 |
| setup_10km | triline | triline_linear_ar_full_k14 | 14 | 18.5034 | 3.9596 | 55.4982 | 79.5316 | 99.5774 | 117.5226 |
| setup_10km | triline | triline_lstm_2l_k14 | 14 | 14.5161 | 2.7005 | 46.3907 | 66.8798 | 88.9212 | 110.2558 |
| setup_10km | triline | triline_transformer_2l_k14 | 14 | 15.3320 | 2.9593 | 50.7165 | 66.7175 | 86.1616 | 109.2729 |
| setup_10km | triline | triline_transformer_4l_k14 | 14 | 15.9379 | 2.8183 | 57.0908 | 69.1111 | 86.2799 | 100.2769 |
| setup_10km | triline | triline_linear_ar_delta_k30 | 30 | 19.6802 | 5.5121 | 62.7266 | 84.7085 | 106.9240 | 118.2844 |
| setup_10km | triline | triline_linear_ar_full_k30 | 30 | 20.0783 | 4.4695 | 61.7715 | 83.1979 | 106.3149 | 116.6270 |
| setup_10km | triline | triline_lstm_2l_k30 | 30 | 16.1421 | 3.3039 | 57.0283 | 67.7981 | 84.4384 | 100.0571 |
| setup_10km | triline | triline_transformer_2l_k30 | 30 | 16.9188 | 3.0661 | 59.5302 | 69.1427 | 86.6525 | 100.6146 |
| setup_10km | triline | triline_transformer_3l_k30 | 30 | 16.2639 | 3.1943 | 54.7651 | 67.3347 | 87.8323 | 106.7619 |
| setup_10km | triline | triline_transformer_4l_k30 | 30 | 16.0207 | 2.4040 | 58.4864 | 69.4540 | 86.7899 | 101.0531 |
| setup_30km | baseline | const_velocity | 7 | 17.8171 | 1.7637 | 54.1154 | 109.1717 | 116.0993 | 138.2874 |
| setup_30km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 125.0660 | 143.1997 | 184.7018 |
| setup_30km | baseline | const_velocity | 14 | 18.6991 | 1.7985 | 59.6849 | 108.8351 | 115.9770 | 138.6733 |
| setup_30km | baseline | persistence | 14 | 15.8945 | 0.7255 | 42.2899 | 125.1869 | 142.8895 | 185.2921 |
| setup_30km | baseline | const_velocity | 30 | 19.5423 | 2.2817 | 61.8870 | 113.1596 | 119.9864 | 141.6438 |
| setup_30km | baseline | persistence | 30 | 16.3394 | 0.8083 | 40.9240 | 128.0119 | 146.6291 | 189.8530 |
| setup_30km | direct | direct_mlp_last_day_k1 | 1 | 14.3415 | 3.7053 | 41.4564 | 81.9896 | 90.4630 | 109.2115 |
| setup_30km | direct | direct_transformer_2l_k7 | 7 | 14.2745 | 3.3934 | 42.7128 | 78.8357 | 85.6334 | 96.5770 |
| setup_30km | direct | direct_transformer_2l_k14 | 14 | 17.9979 | 5.6901 | 51.8707 | 78.0894 | 83.4258 | 99.7445 |
| setup_30km | direct | direct_mlp_sequence_k30 | 30 | 18.5861 | 5.9757 | 53.4965 | 88.6203 | 95.9142 | 118.3885 |
| setup_30km | direct | direct_transformer_2l_k30 | 30 | 16.3260 | 4.9817 | 46.2378 | 83.4259 | 91.9388 | 111.4150 |
| setup_30km | triline | triline_linear_ar_full_k7 | 7 | 16.7314 | 3.4978 | 52.9020 | 90.8283 | 95.5449 | 107.8275 |
| setup_30km | triline | triline_lstm_2l_k7 | 7 | 14.6405 | 2.4481 | 48.7693 | 80.6061 | 84.9780 | 101.6727 |
| setup_30km | triline | triline_transformer_2l_k7 | 7 | 14.6170 | 2.3422 | 47.4418 | 82.4133 | 89.3319 | 113.5901 |
| setup_30km | triline | triline_transformer_4l_k7 | 7 | 15.1976 | 2.7706 | 48.3591 | 78.8528 | 84.2956 | 103.7580 |
| setup_30km | triline | triline_linear_ar_full_k14 | 14 | 18.6162 | 3.9683 | 54.3151 | 95.1497 | 99.8830 | 117.9305 |
| setup_30km | triline | triline_lstm_2l_k14 | 14 | 15.5922 | 2.4563 | 54.5140 | 80.1758 | 83.7615 | 97.7035 |
| setup_30km | triline | triline_transformer_2l_k14 | 14 | 15.5288 | 2.7098 | 52.6272 | 80.4896 | 85.7777 | 106.7352 |
| setup_30km | triline | triline_transformer_4l_k14 | 14 | 16.1159 | 2.6805 | 58.8470 | 77.7525 | 81.3157 | 97.1253 |
| setup_30km | triline | triline_linear_ar_delta_k30 | 30 | 19.8905 | 5.1756 | 65.5906 | 103.6781 | 109.2776 | 120.7486 |
| setup_30km | triline | triline_linear_ar_full_k30 | 30 | 19.3851 | 4.3963 | 58.9545 | 100.2288 | 104.5218 | 117.4022 |
| setup_30km | triline | triline_lstm_2l_k30 | 30 | 15.9201 | 3.2362 | 49.8131 | 82.4322 | 87.8286 | 106.0202 |
| setup_30km | triline | triline_transformer_2l_k30 | 30 | 15.7267 | 2.9605 | 52.4806 | 84.8542 | 92.0208 | 117.6521 |
| setup_30km | triline | triline_transformer_3l_k30 | 30 | 16.4146 | 3.5401 | 55.2366 | 81.6116 | 87.1477 | 106.1809 |
| setup_30km | triline | triline_transformer_4l_k30 | 30 | 15.7970 | 3.0688 | 49.6139 | 82.7978 | 89.6330 | 114.9024 |

## Training Logs

- Summarized 38 training logs in `training_log_summary.csv`.
- Weighted and unweighted triline loss columns are included for auditability.

## Rollout Visualization

- Interactive rollout explorer: `rollout_all/index.html`.
- Rollout paths: 40; skipped paths: 0.
- Rollout comparison rows against weather baseline: 40.

## Notes

- Triline rows report ungated GPS reconstruction as primary metrics.
- Gated GPS metrics are stored in each triline `metrics.json` and the CSV summaries with `gated_` prefixes.
- Triline rollout is fly-gated: days below the tuned fly probability threshold stay at the current location; fly days use predicted distance and heading.
- Direct model rows repeat per setup because setup-specific fly thresholds change stratified error slices.
- Autoregressive rollout uses actual future daily weather as exogenous forecast/oracle weather.
- Lower delta values in comparison files mean distance weighting improved over the original weather run.
