# Final Path Experiment Without Weather

- Setup: `fly_threshold_10km`.
- Fly threshold: 10.0 km.

## Results

| Model | Family | k | Mean km | Median km | P90 km | Fly recall |
|---|---|---:|---:|---:|---:|---:|
| direct_mlp_sequence_k7 | direct | 7 | 18.5871 | 4.2411 | 61.1995 |  |
| direct_transformer_4l_k7 | direct | 7 | 18.1154 | 3.7743 | 59.7921 |  |
| triline_lstm_4l_k7 | triline | 7 | 17.9439 | 2.3699 | 58.2920 | 0.7269 |
| triline_transformer_v2_k7 | triline | 7 | 17.3795 | 2.6264 | 56.1957 | 0.7363 |
| direct_mlp_sequence_k14 | direct | 14 | 18.8258 | 3.9037 | 62.4367 |  |
| direct_transformer_4l_k14 | direct | 14 | 18.0337 | 3.2474 | 59.6372 |  |
| triline_lstm_4l_k14 | triline | 14 | 18.2085 | 2.3016 | 59.0562 | 0.7365 |
| triline_transformer_v2_k14 | triline | 14 | 17.6209 | 2.3971 | 55.9456 | 0.7158 |
| direct_mlp_sequence_k30 | direct | 30 | 18.4319 | 5.2563 | 50.9838 |  |
| direct_transformer_4l_k30 | direct | 30 | 16.5664 | 3.7591 | 55.8641 |  |
| triline_lstm_4l_k30 | triline | 30 | 15.8964 | 2.3795 | 49.5459 | 0.6629 |
| triline_transformer_v2_k30 | triline | 30 | 15.7615 | 2.3074 | 49.4614 | 0.7110 |
