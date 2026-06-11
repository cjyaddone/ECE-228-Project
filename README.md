# Weather-Conditioned Deep Learning for Bird Migration Forecasting

ECE 228 Course Final Project, Team 53  
Xinyan Cai and Jiayi Chen, UC San Diego

## Overview

This project studies next-day southbound bird migration forecasting from daily GPS tracks and matched weather observations. Given a recent context window of bird movement history, the goal is to predict the next day's latitude and longitude. We compare direct coordinate-displacement regression with a triline movement formulation that separately predicts fly/no-fly, distance, and heading, then reconstructs the next location.

The final experiments evaluate Direct MLP, Triline LSTM, and Triline Transformer V2 models with context lengths `k = 7, 14, 30`, both with and without weather covariates. All reported results use a chronological 80/20 train/test split by target date.

For full details, see the project report in [`report/report_template.pdf`](report/report_template.pdf).

## Key Findings

- Triline sequence models generally outperform direct displacement regression for next-day prediction, especially on mean and tail GPS error.
- At `k = 30`, Triline Transformer V2 gives the best mean, median, and P90 next-day errors in both weather and no-weather settings.
- Without weather, Triline Transformer V2 is the strongest learned model across all context lengths by mean error.
- Weather is useful but model-dependent: it improves the LSTM at shorter contexts and improves some rollout/tail-error cases, but it does not uniformly improve every architecture.
- Autoregressive full-path rollout is much harder than one-step prediction because small daily errors compound over migration paths.

Representative `k = 30` next-day mean errors:

| Setting | Best Model | Mean Error |
| --- | --- | ---: |
| No weather | Triline Transformer V2 | 15.76 km |
| Weather | Triline Transformer V2 | 15.95 km |

Autoregressive rollout over 27 held-out paths shows the same compounding-error challenge: the no-weather Transformer has the lowest mean rollout error, while weather substantially improves the LSTM rollout.

## Repository Layout

```text
data/
  combined_southbound_paths_with_weather_matched.csv   # cleaned paths with weather features

model_files/
  model.py                                             # model definitions
  run_final_path_experiment.py                         # final weather/no-weather experiment runner
  run_path_experiment_with_weather.py                  # weather training utilities
  run_path_experiment_no_weather.py                    # no-weather training utilities
  run_current_rollout_gifs.py                          # rollout summaries and GIF generation

with_weather/
  fly_threshold_10km/                                  # trained models, predictions, and metrics

without_weather/
  fly_threshold_10km/                                  # trained models, predictions, and metrics

rollout/
  with_weather/, without_weather/                      # autoregressive rollout predictions and GIFs
  rollout_aggregate_summary.csv                        # aggregate rollout metrics

report/
  report_template.tex                                  # final report source
  report_template.pdf                                  # compiled report
  figures/                                             # report figures

weather_vs_noweather_comparison.csv                    # weather ablation summary
validation_full.json                                   # final run validation summary
```

## Reproducing Results

Run from the repository root.

Install the minimal Python dependencies:

```powershell
pip install -r requirements.txt
```

The requirements include PyTorch, NumPy, Pandas, scikit-learn, Matplotlib, and Pillow.

```powershell
python model_files/run_final_path_experiment.py --mode smoke --device auto
```

Full experiment:

```powershell
python model_files/run_final_path_experiment.py --mode full --max-epochs 200 --patience 30 --device auto
```

Generate rollout summaries and GIFs:

```powershell
python model_files/run_current_rollout_gifs.py
```

The completed full run is summarized in [`validation_full.json`](validation_full.json), which reports successful weather and no-weather experiment outputs.
