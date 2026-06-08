# Final Path Experiment Results

## Setup

- Final folder: `Final_Path_Experiment`
- Dataset: combined Dataset 2 + Dataset 3 southbound paths
- Fly threshold: `10 km`
- Context lengths: `k = 7, 14, 30`
- Models: sequence MLP, 2-layer direct Transformer, 2-layer Triline LSTM, 2-layer Triline Transformer
- Tests: with corrected weather features vs without weather features

## Precipitation Correction

Dataset 2 precipitation was corrected before rerunning the experiment:

```text
precipitation_mm_fixed = precipitation_mm / n_weather_points * 24
```

Correction audit:

| Item | Value |
|---|---:|
| Total rows | 13,550 |
| Dataset 2 rows fixed | 8,180 |
| Dataset 2 rows missing `n_weather_points` | 0 |
| Dataset 3 rows unchanged | 5,370 |
| Dataset 2 original precip mean | 43.40 mm |
| Dataset 2 fixed precip mean | 2.12 mm |
| Dataset 2 original precip max | 6881.28 mm |
| Dataset 2 fixed precip max | 80.33 mm |

Full audit file: `data/precipitation_correction_report.json`.

## Validation

The corrected full run passed validation:

| Test | Rows | Model folders | k values | Fly threshold | Passed |
|---|---:|---:|---|---:|---|
| With weather | 12 | 12 | 7, 14, 30 | 10.0 km | yes |
| Without weather | 12 | 12 | 7, 14, 30 | 10.0 km | yes |

The joined weather-vs-no-weather comparison contains 12 matched rows.

## Best Models

Best mean GPS error with weather:

| Rank | Model | k | Mean km | Median km | P90 km |
|---:|---|---:|---:|---:|---:|
| 1 | `direct_transformer_2l_k30` | 30 | 15.31 | 2.96 | 46.52 |
| 2 | `triline_lstm_2l_k30` | 30 | 16.33 | 2.80 | 47.22 |
| 3 | `triline_transformer_2l_k30` | 30 | 16.46 | 2.89 | 42.52 |
| 4 | `triline_lstm_2l_k7` | 7 | 17.10 | 2.35 | 60.05 |
| 5 | `triline_transformer_2l_k7` | 7 | 17.52 | 2.43 | 62.12 |

Best mean GPS error without weather:

| Rank | Model | k | Mean km | Median km | P90 km |
|---:|---|---:|---:|---:|---:|
| 1 | `triline_lstm_2l_k30` | 30 | 15.98 | 2.52 | 45.88 |
| 2 | `triline_transformer_2l_k30` | 30 | 16.24 | 2.31 | 47.58 |
| 3 | `direct_transformer_2l_k30` | 30 | 16.37 | 3.46 | 56.27 |
| 4 | `triline_lstm_2l_k7` | 7 | 17.29 | 2.32 | 61.02 |
| 5 | `direct_transformer_2l_k7` | 7 | 17.36 | 3.30 | 57.90 |

## Weather vs No-Weather

Negative delta means weather improved mean GPS error.

| Model | k | Weather mean km | No-weather mean km | Delta km | Weather better |
|---|---:|---:|---:|---:|---|
| `direct_mlp_sequence_k7` | 7 | 18.25 | 19.45 | -1.20 | yes |
| `direct_transformer_2l_k7` | 7 | 18.83 | 17.36 | 1.47 | no |
| `triline_lstm_2l_k7` | 7 | 17.10 | 17.29 | -0.19 | yes |
| `triline_transformer_2l_k7` | 7 | 17.52 | 17.50 | 0.01 | no |
| `direct_mlp_sequence_k14` | 14 | 19.70 | 20.19 | -0.50 | yes |
| `direct_transformer_2l_k14` | 14 | 17.92 | 20.09 | -2.18 | yes |
| `triline_lstm_2l_k14` | 14 | 17.82 | 18.10 | -0.28 | yes |
| `triline_transformer_2l_k14` | 14 | 18.56 | 18.09 | 0.48 | no |
| `direct_mlp_sequence_k30` | 30 | 18.76 | 19.41 | -0.65 | yes |
| `direct_transformer_2l_k30` | 30 | 15.31 | 16.37 | -1.06 | yes |
| `triline_lstm_2l_k30` | 30 | 16.33 | 15.98 | 0.35 | no |
| `triline_transformer_2l_k30` | 30 | 16.46 | 16.24 | 0.22 | no |

Weather improved mean error in 7 of 12 matched comparisons. The strongest improvement was `direct_transformer_2l_k14`, improving by 2.18 km. The best overall model was `direct_transformer_2l_k30` with weather, at 15.31 km mean GPS error.

## Takeaways

- The precipitation correction materially changes the weather input scale and makes Dataset 2 more comparable to Dataset 3.
- Weather features helped the direct models more consistently than the Triline models.
- Longer context was generally strongest: all top three with-weather models used `k=30`.
- The best no-weather model was `triline_lstm_2l_k30`, but the best corrected-weather model beat it by 0.67 km mean error.

Primary result files:

- `with_weather/comparison_summary.csv`
- `without_weather/comparison_summary.csv`
- `weather_vs_noweather_comparison.csv`
- `validation_full.json`
