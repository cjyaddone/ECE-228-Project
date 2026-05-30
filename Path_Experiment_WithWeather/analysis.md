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
| setup_10km | triline_transformer_2l_k7 | triline | 7 | 13.2526 | 1.5134 | 0.6791 | 426118 |
| setup_30km | triline_lstm_2l_k7 | triline | 7 | 13.1441 | 1.6358 | 0.7829 | 292870 |

## Weather vs No-Weather

| Setup | Model | Family | k | Weather Mean km | No-Weather Mean km | Delta km | Improved |
|---|---|---|---:|---:|---:|---:|---|
| setup_10km | triline_transformer_2l_k7 | triline | 7 | 13.2526 | 13.4055 | -0.1529 | yes |
| setup_10km | triline_lstm_2l_k7 | triline | 7 | 13.3131 | 12.7311 | 0.5819 | no |
| setup_10km | triline_transformer_4l_k7 | triline | 7 | 13.5762 | 13.3564 | 0.2198 | no |
| setup_10km | triline_transformer_4l_k14 | triline | 14 | 13.7664 | 14.0344 | -0.2679 | yes |
| setup_10km | triline_lstm_2l_k14 | triline | 14 | 13.9571 | 13.7773 | 0.1797 | no |
| setup_10km | triline_transformer_2l_k14 | triline | 14 | 13.9879 | 14.0842 | -0.0963 | yes |
| setup_10km | direct_transformer_2l_k7 | direct | 7 | 14.2745 | 15.1298 | -0.8553 | yes |
| setup_10km | direct_mlp_last_day_k1 | direct | 1 | 14.3415 | 14.5327 | -0.1912 | yes |
| setup_10km | triline_linear_ar_full_k7 | triline | 7 | 14.4053 | 13.7823 | 0.6230 | no |
| setup_10km | triline_lstm_2l_k30 | triline | 30 | 14.5792 | 14.3597 | 0.2195 | no |
| setup_10km | triline_linear_ar_full_k14 | triline | 14 | 14.6606 | 14.8741 | -0.2135 | yes |
| setup_10km | triline_transformer_3l_k30 | triline | 30 | 14.7181 | 14.6729 | 0.0452 | no |
| setup_10km | triline_transformer_2l_k30 | triline | 30 | 14.8153 | 14.5712 | 0.2442 | no |
| setup_10km | triline_transformer_4l_k30 | triline | 30 | 14.8265 | 14.7337 | 0.0928 | no |
| setup_10km | persistence | baseline | 7 | 14.9233 | 14.9233 | 0.0000 | no |
| setup_10km | triline_linear_ar_delta_k30 | triline | 30 | 15.4942 | 15.4942 | 0.0000 | no |
| setup_10km | persistence | baseline | 14 | 15.8945 | 15.8945 | 0.0000 | no |
| setup_10km | triline_linear_ar_full_k30 | triline | 30 | 15.9306 | 14.9202 | 1.0105 | no |
| setup_10km | direct_transformer_2l_k30 | direct | 30 | 16.3260 | 15.3430 | 0.9830 | no |
| setup_10km | persistence | baseline | 30 | 16.3394 | 16.3394 | 0.0000 | no |
| setup_10km | const_velocity | baseline | 7 | 17.8171 | 17.8171 | 0.0000 | no |
| setup_10km | direct_transformer_2l_k14 | direct | 14 | 17.9979 | 14.6008 | 3.3971 | no |
| setup_10km | direct_mlp_sequence_k30 | direct | 30 | 18.5861 | 17.9104 | 0.6757 | no |
| setup_10km | const_velocity | baseline | 14 | 18.6991 | 18.6991 | 0.0000 | no |
| setup_10km | const_velocity | baseline | 30 | 19.5423 | 19.5423 | 0.0000 | no |
| setup_30km | triline_lstm_2l_k7 | triline | 7 | 13.1441 | 13.0181 | 0.1260 | no |
| setup_30km | triline_transformer_4l_k7 | triline | 7 | 13.4557 | 13.0360 | 0.4197 | no |
| setup_30km | triline_transformer_2l_k7 | triline | 7 | 13.5499 | 13.1277 | 0.4221 | no |
| setup_30km | triline_lstm_2l_k14 | triline | 14 | 13.8481 | 13.7905 | 0.0576 | no |
| setup_30km | triline_transformer_4l_k14 | triline | 14 | 13.8695 | 13.9013 | -0.0318 | yes |
| setup_30km | triline_linear_ar_full_k7 | triline | 7 | 14.0921 | 13.7011 | 0.3910 | no |
| setup_30km | triline_transformer_2l_k14 | triline | 14 | 14.1886 | 13.8714 | 0.3172 | no |
| setup_30km | direct_transformer_2l_k7 | direct | 7 | 14.2745 | 15.1298 | -0.8553 | yes |
| setup_30km | direct_mlp_last_day_k1 | direct | 1 | 14.3415 | 14.5327 | -0.1912 | yes |
| setup_30km | triline_transformer_4l_k30 | triline | 30 | 14.5275 | 14.4831 | 0.0444 | no |
| setup_30km | triline_linear_ar_full_k14 | triline | 14 | 14.7038 | 15.4428 | -0.7390 | yes |
| setup_30km | triline_lstm_2l_k30 | triline | 30 | 14.8131 | 14.3155 | 0.4976 | no |
| setup_30km | persistence | baseline | 7 | 14.9233 | 14.9233 | 0.0000 | no |
| setup_30km | triline_transformer_3l_k30 | triline | 30 | 15.0558 | 14.6190 | 0.4368 | no |
| setup_30km | triline_transformer_2l_k30 | triline | 30 | 15.0642 | 14.7824 | 0.2818 | no |
| setup_30km | triline_linear_ar_delta_k30 | triline | 30 | 15.8545 | 15.8545 | 0.0000 | no |
| setup_30km | triline_linear_ar_full_k30 | triline | 30 | 15.8696 | 14.8732 | 0.9965 | no |
| setup_30km | persistence | baseline | 14 | 15.8945 | 15.8945 | 0.0000 | no |
| setup_30km | direct_transformer_2l_k30 | direct | 30 | 16.3260 | 15.3430 | 0.9830 | no |
| setup_30km | persistence | baseline | 30 | 16.3394 | 16.3394 | 0.0000 | no |
| setup_30km | const_velocity | baseline | 7 | 17.8171 | 17.8171 | 0.0000 | no |
| setup_30km | direct_transformer_2l_k14 | direct | 14 | 17.9979 | 14.6008 | 3.3971 | no |
| setup_30km | direct_mlp_sequence_k30 | direct | 30 | 18.5861 | 17.9104 | 0.6757 | no |
| setup_30km | const_velocity | baseline | 14 | 18.6991 | 18.6991 | 0.0000 | no |
| setup_30km | const_velocity | baseline | 30 | 19.5423 | 19.5423 | 0.0000 | no |

## Full Comparison

| Setup | Family | Model | k | Mean km | Median km | P90 km | Fly Mean km | Migration50 Mean km |
|---|---|---|---:|---:|---:|---:|---:|---:|
| setup_10km | baseline | const_velocity | 7 | 17.8171 | 1.7637 | 54.1154 | 89.9617 | 116.0993 |
| setup_10km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 93.6197 | 143.1997 |
| setup_10km | baseline | const_velocity | 14 | 18.6991 | 1.7985 | 59.6849 | 90.1523 | 115.9770 |
| setup_10km | baseline | persistence | 14 | 15.8945 | 0.7255 | 42.2899 | 93.9337 | 142.8895 |
| setup_10km | baseline | const_velocity | 30 | 19.5423 | 2.2817 | 61.8870 | 92.8597 | 119.9864 |
| setup_10km | baseline | persistence | 30 | 16.3394 | 0.8083 | 40.9240 | 94.7498 | 146.6291 |
| setup_10km | direct | direct_mlp_last_day_k1 | 1 | 14.3415 | 3.7053 | 41.4564 | 64.5116 | 90.4630 |
| setup_10km | direct | direct_transformer_2l_k7 | 7 | 14.2745 | 3.3934 | 42.7128 | 65.4065 | 85.6334 |
| setup_10km | direct | direct_transformer_2l_k14 | 14 | 17.9979 | 5.6901 | 51.8707 | 65.3734 | 83.4258 |
| setup_10km | direct | direct_mlp_sequence_k30 | 30 | 18.5861 | 5.9757 | 53.4965 | 71.2272 | 95.9142 |
| setup_10km | direct | direct_transformer_2l_k30 | 30 | 16.3260 | 4.9817 | 46.2378 | 66.7362 | 91.9388 |
| setup_10km | triline | triline_linear_ar_full_k7 | 7 | 14.4053 | 2.2518 | 41.3977 | 75.8648 | 103.1808 |
| setup_10km | triline | triline_lstm_2l_k7 | 7 | 13.3131 | 1.7432 | 37.3616 | 73.3612 | 106.3581 |
| setup_10km | triline | triline_transformer_2l_k7 | 7 | 13.2526 | 1.5134 | 36.0972 | 72.5299 | 104.0932 |
| setup_10km | triline | triline_transformer_4l_k7 | 7 | 13.5762 | 2.0705 | 37.8612 | 71.4707 | 103.5278 |
| setup_10km | triline | triline_linear_ar_full_k14 | 14 | 14.6606 | 2.4772 | 40.3377 | 73.7226 | 103.1092 |
| setup_10km | triline | triline_lstm_2l_k14 | 14 | 13.9571 | 1.8213 | 42.8486 | 72.3198 | 103.8271 |
| setup_10km | triline | triline_transformer_2l_k14 | 14 | 13.9879 | 1.4532 | 42.4032 | 72.1684 | 102.2317 |
| setup_10km | triline | triline_transformer_4l_k14 | 14 | 13.7664 | 1.8179 | 43.0479 | 70.2646 | 99.9366 |
| setup_10km | triline | triline_linear_ar_delta_k30 | 30 | 15.4942 | 2.4612 | 47.2297 | 78.7262 | 111.6037 |
| setup_10km | triline | triline_linear_ar_full_k30 | 30 | 15.9306 | 2.6347 | 46.8470 | 77.7398 | 110.1776 |
| setup_10km | triline | triline_lstm_2l_k30 | 30 | 14.5792 | 2.3388 | 41.1898 | 73.9674 | 108.7902 |
| setup_10km | triline | triline_transformer_2l_k30 | 30 | 14.8153 | 2.1298 | 48.5253 | 72.0123 | 102.3129 |
| setup_10km | triline | triline_transformer_3l_k30 | 30 | 14.7181 | 1.8045 | 45.5050 | 75.0015 | 109.3990 |
| setup_10km | triline | triline_transformer_4l_k30 | 30 | 14.8265 | 2.1660 | 41.4388 | 75.4682 | 112.7434 |
| setup_30km | baseline | const_velocity | 7 | 17.8171 | 1.7637 | 54.1154 | 109.1717 | 116.0993 |
| setup_30km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 125.0660 | 143.1997 |
| setup_30km | baseline | const_velocity | 14 | 18.6991 | 1.7985 | 59.6849 | 108.8351 | 115.9770 |
| setup_30km | baseline | persistence | 14 | 15.8945 | 0.7255 | 42.2899 | 125.1869 | 142.8895 |
| setup_30km | baseline | const_velocity | 30 | 19.5423 | 2.2817 | 61.8870 | 113.1596 | 119.9864 |
| setup_30km | baseline | persistence | 30 | 16.3394 | 0.8083 | 40.9240 | 128.0119 | 146.6291 |
| setup_30km | direct | direct_mlp_last_day_k1 | 1 | 14.3415 | 3.7053 | 41.4564 | 81.9896 | 90.4630 |
| setup_30km | direct | direct_transformer_2l_k7 | 7 | 14.2745 | 3.3934 | 42.7128 | 78.8357 | 85.6334 |
| setup_30km | direct | direct_transformer_2l_k14 | 14 | 17.9979 | 5.6901 | 51.8707 | 78.0894 | 83.4258 |
| setup_30km | direct | direct_mlp_sequence_k30 | 30 | 18.5861 | 5.9757 | 53.4965 | 88.6203 | 95.9142 |
| setup_30km | direct | direct_transformer_2l_k30 | 30 | 16.3260 | 4.9817 | 46.2378 | 83.4259 | 91.9388 |
| setup_30km | triline | triline_linear_ar_full_k7 | 7 | 14.0921 | 2.1807 | 38.8553 | 92.0658 | 101.3198 |
| setup_30km | triline | triline_lstm_2l_k7 | 7 | 13.1441 | 1.6358 | 35.7316 | 92.4872 | 104.7680 |
| setup_30km | triline | triline_transformer_2l_k7 | 7 | 13.5499 | 1.5161 | 36.4824 | 95.1116 | 107.9268 |
| setup_30km | triline | triline_transformer_4l_k7 | 7 | 13.4557 | 1.6424 | 35.2958 | 98.4962 | 112.3707 |
| setup_30km | triline | triline_linear_ar_full_k14 | 14 | 14.7038 | 2.5138 | 39.4708 | 93.0010 | 102.9978 |
| setup_30km | triline | triline_lstm_2l_k14 | 14 | 13.8481 | 1.5514 | 41.9472 | 90.8423 | 101.7271 |
| setup_30km | triline | triline_transformer_2l_k14 | 14 | 14.1886 | 1.7029 | 40.2584 | 93.3426 | 104.7976 |
| setup_30km | triline | triline_transformer_4l_k14 | 14 | 13.8695 | 1.6261 | 41.9484 | 90.6131 | 101.8742 |
| setup_30km | triline | triline_linear_ar_delta_k30 | 30 | 15.8545 | 2.3834 | 47.9186 | 102.9931 | 113.9225 |
| setup_30km | triline | triline_linear_ar_full_k30 | 30 | 15.8696 | 2.5930 | 45.4153 | 100.4785 | 109.9646 |
| setup_30km | triline | triline_lstm_2l_k30 | 30 | 14.8131 | 2.2832 | 39.0870 | 99.8794 | 113.8508 |
| setup_30km | triline | triline_transformer_2l_k30 | 30 | 15.0642 | 2.3396 | 42.3179 | 101.6510 | 115.7111 |
| setup_30km | triline | triline_transformer_3l_k30 | 30 | 15.0558 | 1.9724 | 40.8484 | 101.9662 | 115.7482 |
| setup_30km | triline | triline_transformer_4l_k30 | 30 | 14.5275 | 1.8110 | 46.9698 | 90.9738 | 100.3134 |

## Notes

- Triline rows report ungated GPS reconstruction as primary metrics.
- Gated GPS metrics are stored in each triline `metrics.json` and the CSV summaries with `gated_` prefixes.
- Direct model rows repeat per setup because setup-specific fly thresholds change stratified error slices.
- Autoregressive rollout uses actual future daily weather as exogenous forecast/oracle weather.
