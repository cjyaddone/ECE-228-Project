# Final Path Experiment Without Weather

- Setup: `fly_threshold_10km`.
- Fly threshold: 10.0 km.

## Results

| Model | Family | k | Mean km | Median km | P90 km | Fly recall |
|---|---|---:|---:|---:|---:|---:|
| direct_mlp_sequence_k7 | direct | 7 | 19.4523 | 5.1853 | 61.2415 |  |
| direct_transformer_2l_k7 | direct | 7 | 17.3626 | 3.2953 | 57.9022 |  |
| triline_lstm_2l_k7 | triline | 7 | 17.2921 | 2.3225 | 61.0164 | 0.7307 |
| triline_transformer_2l_k7 | triline | 7 | 17.5027 | 2.7457 | 56.2518 | 0.6855 |
| direct_mlp_sequence_k14 | direct | 14 | 20.1950 | 5.3072 | 61.1268 |  |
| direct_transformer_2l_k14 | direct | 14 | 20.0944 | 6.5404 | 59.8188 |  |
| triline_lstm_2l_k14 | triline | 14 | 18.0981 | 2.3907 | 55.0829 | 0.6563 |
| triline_transformer_2l_k14 | triline | 14 | 18.0855 | 2.6207 | 59.6721 | 0.7308 |
| direct_mlp_sequence_k30 | direct | 30 | 19.4097 | 5.8826 | 56.9827 |  |
| direct_transformer_2l_k30 | direct | 30 | 16.3678 | 3.4552 | 56.2703 |  |
| triline_lstm_2l_k30 | triline | 30 | 15.9834 | 2.5234 | 45.8842 | 0.6714 |
| triline_transformer_2l_k30 | triline | 30 | 16.2418 | 2.3073 | 47.5778 | 0.7025 |
