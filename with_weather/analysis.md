# Final Path Experiment With Weather

- Setup: `fly_threshold_10km`.
- Fly threshold: 10.0 km.

## Results

| Model | Family | k | Mean km | Median km | P90 km | Fly recall |
|---|---|---:|---:|---:|---:|---:|
| direct_mlp_sequence_k7 | direct | 7 | 17.9619 | 4.6671 | 55.3166 |  |
| direct_transformer_4l_k7 | direct | 7 | 15.9849 | 2.7860 | 52.3893 |  |
| triline_lstm_4l_k7 | triline | 7 | 17.8174 | 2.2638 | 52.0240 | 0.6930 |
| triline_transformer_v2_k7 | triline | 7 | 17.8918 | 2.7052 | 55.8930 | 0.7213 |
| direct_mlp_sequence_k14 | direct | 14 | 18.9126 | 4.4254 | 60.6028 |  |
| direct_transformer_4l_k14 | direct | 14 | 17.9767 | 4.4264 | 56.5979 |  |
| triline_lstm_4l_k14 | triline | 14 | 17.7293 | 2.4352 | 60.1241 | 0.6598 |
| triline_transformer_v2_k14 | triline | 14 | 18.1601 | 2.3566 | 58.9602 | 0.7178 |
| direct_mlp_sequence_k30 | direct | 30 | 17.6555 | 4.6629 | 51.3861 |  |
| direct_transformer_4l_k30 | direct | 30 | 15.4114 | 2.9497 | 46.7151 |  |
| triline_lstm_4l_k30 | triline | 30 | 16.0566 | 2.3892 | 51.1449 | 0.6686 |
| triline_transformer_v2_k30 | triline | 30 | 15.9464 | 2.2794 | 45.4561 | 0.7280 |
