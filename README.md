# Weather-Conditioned Bird Migration Trajectory Forecasting

This repository contains the final ECE 228 project for next-day southbound bird migration trajectory prediction. The project compares direct GPS displacement regression with a triline movement decomposition that predicts fly/no-fly, distance, and heading, and evaluates whether weather covariates improve chronological future-date generalization.

## Repository Layout

- `data/combined_southbound_paths_with_weather_matched.csv`: cleaned daily migration path data with matched weather covariates.
- `model_files/run_final_path_experiment.py`: final experiment runner for weather and no-weather settings.
- `model_files/run_path_experiment_with_weather.py`: training/evaluation utilities with weather features.
- `model_files/run_path_experiment_no_weather.py`: training/evaluation utilities without weather features.
- `model_files/model.py`: Triline Transformer V2 model definition.
- `with_weather/`: completed final experiment outputs with weather covariates.
- `without_weather/`: completed final experiment outputs without weather covariates.
- `rollout/`: autoregressive rollout summaries, predictions, and representative GIFs.
- `report/`: final LaTeX report, generated figures, and compiled PDF.

## Reproducing Experiments

Run commands from the repository root.

Smoke test:

```powershell
python model_files/run_final_path_experiment.py --mode smoke --device auto
```

Full experiment:

```powershell
python model_files/run_final_path_experiment.py --mode full --max-epochs 200 --patience 30 --device auto
```

Generate current rollout summaries and GIFs:

```powershell
python model_files/run_current_rollout_gifs.py
```

The full run uses seed `42`, context lengths `k = 7, 14, 30`, a 10 km fly/stationary threshold, AdamW with learning rate `1e-3` and weight decay `1e-4`, batch size `128`, and a chronological 80/20 train/test split by target date. The current training loop uses the chronological test partition for learning-rate scheduling, early stopping, and checkpoint selection; there is no separate validation split.

## Results Artifacts

Key CSV and JSON outputs:

- `with_weather/fly_threshold_10km/comparison_summary.csv`
- `without_weather/fly_threshold_10km/comparison_summary.csv`
- `weather_vs_noweather_comparison.csv`
- `rollout/rollout_aggregate_summary.csv`
- `validation_full.json`

The completed full experiment in `validation_full.json` should report `passed: true` for both weather and no-weather outputs.

## Report

Compile the report from the `report/` directory:

```powershell
cd report
pdflatex report_template.tex
pdflatex report_template.tex
```

The report figures are generated from the completed CSV artifacts and stored in `report/figures/`.
