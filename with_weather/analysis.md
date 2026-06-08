# Final Path Experiment With Weather

- Setup: `fly_threshold_10km`.
- Fly threshold: 10.0 km.

## Results

| Model | Family | k | Mean km | Median km | P90 km | Fly recall |
|---|---|---:|---:|---:|---:|---:|
| direct_mlp_sequence_k7 | direct | 7 | 18.2550 | 4.3657 | 57.7439 |  |
| direct_transformer_2l_k7 | direct | 7 | 18.8345 | 4.8979 | 63.7175 |  |
| triline_lstm_2l_k7 | triline | 7 | 17.1005 | 2.3520 | 60.0547 | 0.6949 |
| triline_transformer_2l_k7 | triline | 7 | 17.5173 | 2.4260 | 62.1237 | 0.6403 |
| direct_mlp_sequence_k14 | direct | 14 | 19.6976 | 4.5368 | 64.4559 |  |
| direct_transformer_2l_k14 | direct | 14 | 17.9172 | 3.9683 | 56.9761 |  |
| triline_lstm_2l_k14 | triline | 14 | 17.8225 | 2.5187 | 58.2902 | 0.6335 |
| triline_transformer_2l_k14 | triline | 14 | 18.5615 | 2.3297 | 64.3211 | 0.6253 |
| direct_mlp_sequence_k30 | direct | 30 | 18.7579 | 5.7661 | 49.9647 |  |
| direct_transformer_2l_k30 | direct | 30 | 15.3115 | 2.9636 | 46.5190 |  |
| triline_lstm_2l_k30 | triline | 30 | 16.3311 | 2.7980 | 47.2200 | 0.6516 |
| triline_transformer_2l_k30 | triline | 30 | 16.4581 | 2.8915 | 42.5225 | 0.6232 |
