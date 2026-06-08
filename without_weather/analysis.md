# Final Path Experiment Without Weather

- Setup: `fly_threshold_10km`.
- Fly threshold: 10.0 km.

## Results

| Model | Family | k | Mean km | Median km | P90 km | Fly recall |
|---|---|---:|---:|---:|---:|---:|
| direct_mlp_sequence_k7 | direct | 7 | 19.2112 | 4.4726 | 62.1823 |  |
| direct_transformer_4l_k7 | direct | 7 | 17.6510 | 2.4655 | 58.4136 |  |
| triline_lstm_4l_k7 | triline | 7 | 18.1902 | 2.2640 | 63.0585 | 0.7236 |
| triline_transformer_v2_k7 | triline | 7 | 18.1774 | 2.5556 | 62.3935 | 0.6957 |
| direct_mlp_sequence_k14 | direct | 14 | 20.6319 | 5.1427 | 64.6713 |  |
| direct_transformer_4l_k14 | direct | 14 | 19.1797 | 3.6874 | 62.6445 |  |
| triline_lstm_4l_k14 | triline | 14 | 18.8507 | 2.4857 | 62.2857 | 0.7313 |
| triline_transformer_v2_k14 | triline | 14 | 18.6557 | 2.5361 | 56.6631 | 0.7381 |
| direct_mlp_sequence_k30 | direct | 30 | 20.8360 | 5.7480 | 62.4816 |  |
| direct_transformer_4l_k30 | direct | 30 | 18.2162 | 4.3492 | 57.9296 |  |
| triline_lstm_4l_k30 | triline | 30 | 17.0553 | 2.4537 | 49.6591 | 0.7021 |
| triline_transformer_v2_k30 | triline | 30 | 17.1975 | 2.2739 | 49.3433 | 0.7182 |
