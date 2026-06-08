# Final Path Experiment Without Weather

- Setup: `fly_threshold_10km`.
- Fly threshold: 10.0 km.

## Results

| Model | Family | k | Mean km | Median km | P90 km | Fly recall |
|---|---|---:|---:|---:|---:|---:|
| direct_mlp_sequence_k7 | direct | 7 | 19.2112 | 4.4726 | 62.1823 |  |
| direct_transformer_4l_k7 | direct | 7 | 17.6510 | 2.4655 | 58.4136 |  |
| triline_lstm_4l_k7 | triline | 7 | 17.7684 | 1.5885 | 63.2448 | 0.7236 |
| triline_transformer_v2_k7 | triline | 7 | 17.5775 | 1.5214 | 62.7518 | 0.6957 |
| direct_mlp_sequence_k14 | direct | 14 | 20.6319 | 5.1427 | 64.6713 |  |
| direct_transformer_4l_k14 | direct | 14 | 19.1797 | 3.6874 | 62.6445 |  |
| triline_lstm_4l_k14 | triline | 14 | 18.3059 | 1.6503 | 62.2891 | 0.7313 |
| triline_transformer_v2_k14 | triline | 14 | 18.1062 | 1.6447 | 56.6631 | 0.7381 |
| direct_mlp_sequence_k30 | direct | 30 | 20.8360 | 5.7480 | 62.4816 |  |
| direct_transformer_4l_k30 | direct | 30 | 18.2162 | 4.3492 | 57.9296 |  |
| triline_lstm_4l_k30 | triline | 30 | 16.5131 | 1.7341 | 50.4987 | 0.7021 |
| triline_transformer_v2_k30 | triline | 30 | 16.7601 | 1.8898 | 49.4933 | 0.7182 |
