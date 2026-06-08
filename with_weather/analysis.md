# Final Path Experiment With Weather

- Setup: `fly_threshold_10km`.
- Fly threshold: 10.0 km.

## Results

| Model | Family | k | Mean km | Median km | P90 km | Fly recall |
|---|---|---:|---:|---:|---:|---:|
| direct_mlp_sequence_k7 | direct | 7 | 19.0185 | 4.8512 | 59.9813 |  |
| direct_transformer_4l_k7 | direct | 7 | 17.7878 | 3.6925 | 56.8686 |  |
| triline_lstm_4l_k7 | triline | 7 | 17.8607 | 2.4063 | 53.5377 | 0.6149 |
| triline_transformer_v2_k7 | triline | 7 | 17.9643 | 2.4124 | 57.6024 | 0.6584 |
| direct_mlp_sequence_k14 | direct | 14 | 20.6816 | 5.1258 | 66.2040 |  |
| direct_transformer_4l_k14 | direct | 14 | 21.5652 | 8.2191 | 60.1648 |  |
| triline_lstm_4l_k14 | triline | 14 | 18.6888 | 2.7364 | 59.0594 | 0.6786 |
| triline_transformer_v2_k14 | triline | 14 | 18.8497 | 2.5083 | 61.3586 | 0.6837 |
| direct_mlp_sequence_k30 | direct | 30 | 19.0773 | 5.0243 | 58.8373 |  |
| direct_transformer_4l_k30 | direct | 30 | 17.4664 | 3.4517 | 59.7563 |  |
| triline_lstm_4l_k30 | triline | 30 | 17.0342 | 2.4681 | 51.5723 | 0.7644 |
| triline_transformer_v2_k30 | triline | 30 | 17.3340 | 2.4172 | 52.7752 | 0.6697 |
