# Transformer Optimization Experiment Results

- Mode: full
- Epochs max: 200, Patience: 30
- Fly threshold: 10.0 km
- Split: year-grouped (80/20)

## With Weather

| Model | k | Mean km | Median km | P90 km | Fly Recall | Params |
|---|---:|---:|---:|---:|---:|---:|
## With Weather

| Model | k | Mean km | Median km | P90 km | Fly Recall | Params |
|---|---:|---:|---:|---:|---:|---:|
| triline_lstm_2l | 7 | 31.2879 | 21.2900 | 59.6585 | 0.6211 | 341,956 |
| triline_lstm_2l | 7 | 30.0168 | 19.8562 | 58.7029 | 0.6879 | 340,932 |
| triline_lstm_4l | 7 | 21.7154 | 7.6768 | 58.3798 | 0.6817 | 2,235,460 |
| triline_lstm_4l | 7 | 23.1882 | 9.5756 | 62.2853 | 0.7220 | 2,233,412 |
| triline_transformer_v2 | 7 | 25.4491 | 13.5609 | 63.4812 | 0.6366 | 3,291,460 |
| triline_transformer_v2 | 7 | 32.4505 | 21.0878 | 64.9817 | 0.6988 | 3,289,412 |
| triline_lstm_2l | 14 | 31.8228 | 21.6316 | 62.4776 | 0.6888 | 341,956 |
| triline_lstm_2l | 14 | 31.9377 | 21.8462 | 67.6327 | 0.7330 | 340,932 |
| triline_lstm_4l | 14 | 20.7101 | 5.6441 | 58.4801 | 0.7211 | 2,235,460 |
| triline_lstm_4l | 14 | 22.0021 | 5.6750 | 62.3566 | 0.6888 | 2,233,412 |
| triline_transformer_v2 | 14 | 29.6220 | 18.6486 | 63.6272 | 0.7041 | 3,293,252 |
| triline_transformer_v2 | 14 | 31.5472 | 21.6205 | 65.1955 | 0.7262 | 3,291,204 |
| triline_lstm_2l | 30 | 28.2782 | 18.1805 | 58.4125 | 0.6397 | 341,956 |
| triline_lstm_2l | 30 | 27.7474 | 17.3710 | 62.1692 | 0.6767 | 340,932 |
| triline_lstm_4l | 30 | 21.9745 | 8.6778 | 52.4294 | 0.6928 | 2,235,460 |
| triline_lstm_4l | 30 | 23.3405 | 10.5870 | 58.3778 | 0.6998 | 2,233,412 |
| triline_transformer_v2 | 30 | 23.0134 | 9.6087 | 58.3041 | 0.6952 | 3,297,348 |
| triline_transformer_v2 | 30 | 30.6611 | 19.7759 | 60.7988 | 0.7229 | 3,295,300 |

## Without Weather

| Model | k | Mean km | Median km | P90 km | Fly Recall | Params |
|---|---:|---:|---:|---:|---:|---:|

