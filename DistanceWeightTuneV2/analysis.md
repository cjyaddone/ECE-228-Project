# DistanceWeightTune V2

## Setup

- Dataset: southbound path segments.
- Features: compact 18-feature inputs with 10 path features and 8 weather features.
- Change under test: focused triline ablations with milder distance weights, distance-only weighting, movement-category conditioning, and balanced checkpoint selection.
- Weight configs: mild_bins, medium_bins, distance_only, current_bins.
- Direct models and baselines are unchanged controls.
- Weather match: exact bird/date join with nearest same-bird date fallback.
- Split: chronological 80/20 within constructed windows.
- Identity: source bird ID embedding for neural models.

## Best Mean GPS Error By Setup

| Setup | Model | Family | k | Mean km | Median km | Fly recall | Params |
|---|---|---|---:|---:|---:|---:|---:|
| setup_10km | triline_transformer_2l_k7__mild_bins | triline | 7 | 13.2295 | 1.4480 | 0.6605 | 435154 |
| setup_30km | triline_lstm_2l_k7__mild_bins | triline | 7 | 13.1419 | 1.6234 | 0.7237 | 301906 |

## Distance-Weighted vs Weather Baseline

| Setup | Model | Family | k | Weighted Mean km | Weather Mean km | Mean Delta km | Migration50 Delta km | Improved |
|---|---|---|---:|---:|---:|---:|---:|---|
| setup_10km | triline_transformer_2l_k7__mild_bins | triline | 7 | 13.2295 | 13.2526 | -0.0231 | -5.0747 | yes |
| setup_10km | triline_lstm_2l_k7__mild_bins | triline | 7 | 13.2426 | 13.3131 | -0.0704 | -9.9616 | yes |
| setup_10km | triline_lstm_2l_k7__medium_bins | triline | 7 | 13.2891 | 13.3131 | -0.0239 | -13.8314 | yes |
| setup_10km | triline_lstm_2l_k7__distance_only | triline | 7 | 13.3333 | 13.3131 | 0.0202 | -13.7389 | no |
| setup_10km | triline_lstm_2l_k14__mild_bins | triline | 14 | 13.5440 | 13.9571 | -0.4130 | -6.2678 | yes |
| setup_10km | triline_transformer_4l_k7__mild_bins | triline | 7 | 13.5528 | 13.5762 | -0.0234 | -7.4848 | yes |
| setup_10km | triline_transformer_4l_k7__medium_bins | triline | 7 | 13.6624 | 13.5762 | 0.0863 | -9.7772 | no |
| setup_10km | triline_transformer_4l_k7__distance_only | triline | 7 | 13.7089 | 13.5762 | 0.1328 | -14.2449 | no |
| setup_10km | triline_transformer_2l_k7__current_bins | triline | 7 | 13.7108 | 13.2526 | 0.4582 | -13.6610 | no |
| setup_10km | triline_transformer_2l_k7__distance_only | triline | 7 | 13.7216 | 13.2526 | 0.4690 | -9.8590 | no |
| setup_10km | triline_transformer_2l_k7__medium_bins | triline | 7 | 13.7445 | 13.2526 | 0.4919 | -7.3177 | no |
| setup_10km | triline_lstm_2l_k14__medium_bins | triline | 14 | 13.8138 | 13.9571 | -0.1432 | -9.4405 | yes |
| setup_10km | triline_lstm_2l_k14__distance_only | triline | 14 | 13.8961 | 13.9571 | -0.0610 | -9.9311 | yes |
| setup_10km | triline_lstm_2l_k7__current_bins | triline | 7 | 14.0651 | 13.3131 | 0.7520 | -17.9084 | no |
| setup_10km | triline_transformer_4l_k7__current_bins | triline | 7 | 14.0734 | 13.5762 | 0.4972 | -15.1556 | no |
| setup_10km | triline_lstm_2l_k14__current_bins | triline | 14 | 14.2501 | 13.9571 | 0.2931 | -15.7355 | no |
| setup_10km | direct_transformer_2l_k7 | direct | 7 | 14.2745 | 14.2745 | 0.0000 | 0.0000 | no |
| setup_10km | direct_mlp_last_day_k1 | direct | 1 | 14.3415 | 14.3415 | 0.0000 | 0.0000 | no |
| setup_10km | persistence | baseline | 7 | 14.9233 | 14.9233 | 0.0000 | 0.0000 | no |
| setup_10km | persistence | baseline | 14 | 15.8945 | 15.8945 | 0.0000 | 0.0000 | no |
| setup_10km | persistence | baseline | 30 | 16.3394 | 16.3394 | 0.0000 | 0.0000 | no |
| setup_10km | const_velocity | baseline | 7 | 17.8171 | 17.8171 | 0.0000 | -0.0000 | no |
| setup_10km | direct_transformer_2l_k14 | direct | 14 | 17.9979 | 17.9979 | 0.0000 | 0.0000 | no |
| setup_10km | const_velocity | baseline | 14 | 18.6991 | 18.6991 | 0.0000 | 0.0000 | no |
| setup_10km | const_velocity | baseline | 30 | 19.5423 | 19.5423 | 0.0000 | 0.0000 | no |
| setup_30km | triline_lstm_2l_k7__mild_bins | triline | 7 | 13.1419 | 13.1441 | -0.0023 | -5.9378 | yes |
| setup_30km | triline_lstm_2l_k7__medium_bins | triline | 7 | 13.3142 | 13.1441 | 0.1701 | -12.4369 | no |
| setup_30km | triline_transformer_2l_k7__mild_bins | triline | 7 | 13.3174 | 13.5499 | -0.2325 | -9.9998 | yes |
| setup_30km | triline_lstm_2l_k7__distance_only | triline | 7 | 13.3333 | 13.1441 | 0.1892 | -12.4331 | no |
| setup_30km | triline_transformer_4l_k7__mild_bins | triline | 7 | 13.3502 | 13.4557 | -0.1055 | -18.7162 | yes |
| setup_30km | triline_transformer_2l_k7__distance_only | triline | 7 | 13.4238 | 13.5499 | -0.1260 | -11.6790 | yes |
| setup_30km | triline_transformer_4l_k7__medium_bins | triline | 7 | 13.4818 | 13.4557 | 0.0261 | -22.2058 | no |
| setup_30km | triline_lstm_2l_k14__mild_bins | triline | 14 | 13.7637 | 13.8481 | -0.0844 | -4.8759 | yes |
| setup_30km | triline_lstm_2l_k14__medium_bins | triline | 14 | 13.9303 | 13.8481 | 0.0822 | -7.5754 | no |
| setup_30km | triline_transformer_2l_k7__current_bins | triline | 7 | 13.9367 | 13.5499 | 0.3868 | -13.0992 | no |
| setup_30km | triline_transformer_4l_k7__distance_only | triline | 7 | 13.9527 | 13.4557 | 0.4970 | -19.3552 | no |
| setup_30km | triline_lstm_2l_k14__distance_only | triline | 14 | 13.9863 | 13.8481 | 0.1382 | -8.5472 | no |
| setup_30km | triline_transformer_2l_k7__medium_bins | triline | 7 | 14.0203 | 13.5499 | 0.4705 | -8.7745 | no |
| setup_30km | direct_transformer_2l_k7 | direct | 7 | 14.2745 | 14.2745 | 0.0000 | 0.0000 | no |
| setup_30km | direct_mlp_last_day_k1 | direct | 1 | 14.3415 | 14.3415 | 0.0000 | 0.0000 | no |
| setup_30km | triline_lstm_2l_k7__current_bins | triline | 7 | 14.4583 | 13.1441 | 1.3142 | -16.0716 | no |
| setup_30km | triline_lstm_2l_k14__current_bins | triline | 14 | 14.5983 | 13.8481 | 0.7502 | -14.4665 | no |
| setup_30km | persistence | baseline | 7 | 14.9233 | 14.9233 | 0.0000 | 0.0000 | no |
| setup_30km | triline_transformer_4l_k7__current_bins | triline | 7 | 14.9973 | 13.4557 | 1.5416 | -26.0957 | no |
| setup_30km | persistence | baseline | 14 | 15.8945 | 15.8945 | 0.0000 | 0.0000 | no |
| setup_30km | persistence | baseline | 30 | 16.3394 | 16.3394 | 0.0000 | 0.0000 | no |
| setup_30km | const_velocity | baseline | 7 | 17.8171 | 17.8171 | 0.0000 | -0.0000 | no |
| setup_30km | direct_transformer_2l_k14 | direct | 14 | 17.9979 | 17.9979 | 0.0000 | 0.0000 | no |
| setup_30km | const_velocity | baseline | 14 | 18.6991 | 18.6991 | 0.0000 | 0.0000 | no |
| setup_30km | const_velocity | baseline | 30 | 19.5423 | 19.5423 | 0.0000 | 0.0000 | no |

Distance-weighted triline rows improving migration50 error: 32/32.

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
| setup_10km | triline | triline_lstm_2l_k7__current_bins | 7 | 14.0651 | 2.1191 | 44.0333 | 68.0078 | 88.4497 | 109.1898 |
| setup_10km | triline | triline_lstm_2l_k7__distance_only | 7 | 13.3333 | 1.8799 | 42.7125 | 68.4795 | 92.6192 | 110.8110 |
| setup_10km | triline | triline_lstm_2l_k7__medium_bins | 7 | 13.2891 | 1.8692 | 41.8185 | 67.8970 | 92.5267 | 113.1820 |
| setup_10km | triline | triline_lstm_2l_k7__mild_bins | 7 | 13.2426 | 1.8351 | 38.9170 | 69.2052 | 96.3965 | 120.1674 |
| setup_10km | triline | triline_transformer_2l_k7__current_bins | 7 | 13.7108 | 1.8384 | 44.8006 | 66.8975 | 90.4321 | 114.7647 |
| setup_10km | triline | triline_transformer_2l_k7__distance_only | 7 | 13.7216 | 2.0057 | 44.2213 | 68.6873 | 94.2342 | 122.2612 |
| setup_10km | triline | triline_transformer_2l_k7__medium_bins | 7 | 13.7445 | 1.8536 | 42.6249 | 69.8297 | 96.7755 | 125.9898 |
| setup_10km | triline | triline_transformer_2l_k7__mild_bins | 7 | 13.2295 | 1.4480 | 42.2664 | 70.5867 | 99.0185 | 128.5048 |
| setup_10km | triline | triline_transformer_4l_k7__current_bins | 7 | 14.0734 | 2.3037 | 45.8031 | 66.2719 | 88.3721 | 113.3904 |
| setup_10km | triline | triline_transformer_4l_k7__distance_only | 7 | 13.7089 | 2.0838 | 46.0501 | 66.9180 | 89.2829 | 114.2703 |
| setup_10km | triline | triline_transformer_4l_k7__medium_bins | 7 | 13.6624 | 2.1096 | 42.9323 | 68.9187 | 93.7505 | 119.2023 |
| setup_10km | triline | triline_transformer_4l_k7__mild_bins | 7 | 13.5528 | 2.0392 | 41.1728 | 68.6752 | 96.0429 | 126.2621 |
| setup_10km | triline | triline_lstm_2l_k14__current_bins | 14 | 14.2501 | 2.3421 | 48.0874 | 66.3197 | 88.0916 | 108.6066 |
| setup_10km | triline | triline_lstm_2l_k14__distance_only | 14 | 13.8961 | 1.9517 | 45.8002 | 68.2184 | 93.8959 | 116.3840 |
| setup_10km | triline | triline_lstm_2l_k14__medium_bins | 14 | 13.8138 | 1.9095 | 44.9579 | 68.2061 | 94.3865 | 116.9693 |
| setup_10km | triline | triline_lstm_2l_k14__mild_bins | 14 | 13.5440 | 1.6331 | 43.7435 | 69.1138 | 97.5592 | 122.0016 |
| setup_30km | baseline | const_velocity | 7 | 17.8171 | 1.7637 | 54.1154 | 109.1717 | 116.0993 | 138.2874 |
| setup_30km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 125.0660 | 143.1997 | 184.7018 |
| setup_30km | baseline | const_velocity | 14 | 18.6991 | 1.7985 | 59.6849 | 108.8351 | 115.9770 | 138.6733 |
| setup_30km | baseline | persistence | 14 | 15.8945 | 0.7255 | 42.2899 | 125.1869 | 142.8895 | 185.2921 |
| setup_30km | baseline | const_velocity | 30 | 19.5423 | 2.2817 | 61.8870 | 113.1596 | 119.9864 | 141.6438 |
| setup_30km | baseline | persistence | 30 | 16.3394 | 0.8083 | 40.9240 | 128.0119 | 146.6291 | 189.8530 |
| setup_30km | direct | direct_mlp_last_day_k1 | 1 | 14.3415 | 3.7053 | 41.4564 | 81.9896 | 90.4630 | 109.2115 |
| setup_30km | direct | direct_transformer_2l_k7 | 7 | 14.2745 | 3.3934 | 42.7128 | 78.8357 | 85.6334 | 96.5770 |
| setup_30km | direct | direct_transformer_2l_k14 | 14 | 17.9979 | 5.6901 | 51.8707 | 78.0894 | 83.4258 | 99.7445 |
| setup_30km | triline | triline_lstm_2l_k7__current_bins | 7 | 14.4583 | 2.3560 | 45.2137 | 82.3482 | 88.6964 | 107.9146 |
| setup_30km | triline | triline_lstm_2l_k7__distance_only | 7 | 13.3333 | 1.7510 | 42.1840 | 84.1593 | 92.3349 | 119.2050 |
| setup_30km | triline | triline_lstm_2l_k7__medium_bins | 7 | 13.3142 | 1.7353 | 42.5084 | 84.1665 | 92.3311 | 119.3355 |
| setup_30km | triline | triline_lstm_2l_k7__mild_bins | 7 | 13.1419 | 1.6234 | 38.3730 | 88.5479 | 98.8302 | 129.0243 |
| setup_30km | triline | triline_transformer_2l_k7__current_bins | 7 | 13.9367 | 2.2365 | 43.0309 | 85.2550 | 94.8277 | 126.1575 |
| setup_30km | triline | triline_transformer_2l_k7__distance_only | 7 | 13.4238 | 1.8127 | 41.1527 | 86.1197 | 96.2478 | 125.1633 |
| setup_30km | triline | triline_transformer_2l_k7__medium_bins | 7 | 14.0203 | 2.2148 | 40.7438 | 88.2051 | 99.1523 | 131.4900 |
| setup_30km | triline | triline_transformer_2l_k7__mild_bins | 7 | 13.3174 | 1.5224 | 42.7698 | 87.8446 | 97.9271 | 126.5063 |
| setup_30km | triline | triline_transformer_4l_k7__current_bins | 7 | 14.9973 | 2.8859 | 48.3928 | 80.9473 | 86.2749 | 102.7226 |
| setup_30km | triline | triline_transformer_4l_k7__distance_only | 7 | 13.9527 | 2.0040 | 43.4967 | 83.9624 | 93.0155 | 120.7075 |
| setup_30km | triline | triline_transformer_4l_k7__medium_bins | 7 | 13.4818 | 1.9195 | 44.8827 | 81.7023 | 90.1648 | 116.8369 |
| setup_30km | triline | triline_transformer_4l_k7__mild_bins | 7 | 13.3502 | 1.8582 | 42.9970 | 84.5741 | 93.6544 | 121.8972 |
| setup_30km | triline | triline_lstm_2l_k14__current_bins | 14 | 14.5983 | 2.6850 | 47.8259 | 79.7673 | 87.2605 | 105.7959 |
| setup_30km | triline | triline_lstm_2l_k14__distance_only | 14 | 13.9863 | 2.0765 | 45.0627 | 83.6301 | 93.1798 | 116.2642 |
| setup_30km | triline | triline_lstm_2l_k14__medium_bins | 14 | 13.9303 | 2.0420 | 42.3219 | 84.3330 | 94.1517 | 117.6435 |
| setup_30km | triline | triline_lstm_2l_k14__mild_bins | 14 | 13.7637 | 1.9535 | 45.9504 | 87.0036 | 96.8511 | 118.8667 |

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
