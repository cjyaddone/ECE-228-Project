# Final Path Experiment - Chronological Split Results

**Date:** 2026-06-08  
**Setup:** `fly_threshold_10km` (10 km fly/stationary threshold)  
**Split:** chronological 80/20 by target date (first 80% train, last 20% test; no shuffling)  
**k values:** 7, 14, 30  
**Epochs:** max 200, patience 30  

## Important Evaluation Note

The current training loop uses the chronological test partition for learning-rate scheduling, early stopping, and checkpoint selection. There is no separate validation set in this experiment.

## With Weather

### k=7

| Model | Mean km | Median km | P90 km | Mig.50 mean km | Fly recall | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| direct_transformer_4l_k7 | 15.98 | 2.79 | 52.39 | 80.59 |  | 12 |
| triline_lstm_4l_k7 | 17.82 | 2.26 | 52.02 | 106.33 | 0.693 | 3 |
| triline_transformer_v2_k7 | 17.89 | 2.71 | 55.89 | 97.93 | 0.721 | 10 |
| direct_mlp_sequence_k7 | 17.96 | 4.67 | 55.32 | 83.64 |  | 11 |

Winner by mean error: **direct_transformer_4l_k7** (15.98 km).

### k=14

| Model | Mean km | Median km | P90 km | Mig.50 mean km | Fly recall | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| triline_lstm_4l_k14 | 17.73 | 2.44 | 60.12 | 94.90 | 0.660 | 5 |
| direct_transformer_4l_k14 | 17.98 | 4.43 | 56.60 | 86.62 |  | 4 |
| triline_transformer_v2_k14 | 18.16 | 2.36 | 58.96 | 98.79 | 0.718 | 21 |
| direct_mlp_sequence_k14 | 18.91 | 4.43 | 60.60 | 90.85 |  | 7 |

Winner by mean error: **triline_lstm_4l_k14** (17.73 km).

### k=30

| Model | Mean km | Median km | P90 km | Mig.50 mean km | Fly recall | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| direct_transformer_4l_k30 | 15.41 | 2.95 | 46.72 | 83.55 |  | 12 |
| triline_transformer_v2_k30 | 15.95 | 2.28 | 45.46 | 98.99 | 0.728 | 21 |
| triline_lstm_4l_k30 | 16.06 | 2.39 | 51.14 | 89.67 | 0.669 | 5 |
| direct_mlp_sequence_k30 | 17.66 | 4.66 | 51.39 | 95.75 |  | 5 |

Winner by mean error: **direct_transformer_4l_k30** (15.41 km).

## Without Weather

### k=7

| Model | Mean km | Median km | P90 km | Mig.50 mean km | Fly recall | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| triline_transformer_v2_k7 | 17.38 | 2.63 | 56.20 | 95.35 | 0.736 | 27 |
| triline_lstm_4l_k7 | 17.94 | 2.37 | 58.29 | 100.39 | 0.727 | 7 |
| direct_transformer_4l_k7 | 18.12 | 3.77 | 59.79 | 79.37 |  | 10 |
| direct_mlp_sequence_k7 | 18.59 | 4.24 | 61.20 | 85.62 |  | 10 |

Winner by mean error: **triline_transformer_v2_k7** (17.38 km).

### k=14

| Model | Mean km | Median km | P90 km | Mig.50 mean km | Fly recall | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| triline_transformer_v2_k14 | 17.62 | 2.40 | 55.95 | 99.17 | 0.716 | 26 |
| direct_transformer_4l_k14 | 18.03 | 3.25 | 59.64 | 81.79 |  | 14 |
| triline_lstm_4l_k14 | 18.21 | 2.30 | 59.06 | 104.56 | 0.737 | 7 |
| direct_mlp_sequence_k14 | 18.83 | 3.90 | 62.44 | 89.72 |  | 6 |

Winner by mean error: **triline_transformer_v2_k14** (17.62 km).

### k=30

| Model | Mean km | Median km | P90 km | Mig.50 mean km | Fly recall | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| triline_transformer_v2_k30 | 15.76 | 2.31 | 49.46 | 91.97 | 0.711 | 19 |
| triline_lstm_4l_k30 | 15.90 | 2.38 | 49.55 | 91.30 | 0.663 | 11 |
| direct_transformer_4l_k30 | 16.57 | 3.76 | 55.86 | 82.70 |  | 11 |
| direct_mlp_sequence_k30 | 18.43 | 5.26 | 50.98 | 90.07 |  | 5 |

Winner by mean error: **triline_transformer_v2_k30** (15.76 km).

## Weather Impact

Weather effect is reported as `weather - no weather`; negative values mean weather improves mean error.

| Model | k=7 | k=14 | k=30 |
|---|---:|---:|---:|
| Triline LSTM 4L | -0.13 | -0.48 | +0.16 |
| Triline Transformer V2 | +0.51 | +0.54 | +0.18 |

Weather now helps the LSTM at k=7 and k=14, slightly hurts it at k=30, and hurts Transformer mean error at all k values. Weather still improves some tail-error slices, including Transformer P90 at k=30.

## Rollout Evaluation

| Condition | Direct mean km | LSTM mean km | Transformer mean km | Persistence mean km |
|---|---:|---:|---:|---:|
| with_weather | 316.03 | 136.61 | 144.11 | 388.62 |
| without_weather | 534.17 | 178.37 | 129.55 | 388.62 |

Rollout results are mixed: weather substantially improves Direct and LSTM rollouts but the no-weather Transformer has the lowest mean rollout error. All learned rollouts remain sensitive to compounding path-level drift.

## Conclusions

1. The chronological split produces strong one-step performance, especially for long-context Transformer models.
2. With weather at k=30, Direct Transformer 4L has the best mean error (15.41 km) and Triline Transformer V2 has the best median/P90 among triline models.
3. Without weather, Triline Transformer V2 is the strongest model across k=7,14,30 by mean error.
4. Weather is useful but not uniformly: it improves Direct models strongly and helps some LSTM settings, while Transformer mean error increases slightly with weather.
5. Autoregressive rollout remains the harder setting because errors compound over full migration paths.
