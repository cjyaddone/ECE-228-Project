# ECE 228 Bird Trajectory Project

## Path Experiment No Weather

This repository includes a no-weather southbound path trajectory experiment implemented in:

- `Triline-Transformer/run_path_experiment_no_weather.py`

The experiment reuses the southbound compact setup conventions:

- Dataset: `data/filtered/dataset2_southbound_paths_jun_dec_lat30_50_min50_drop5.csv`
- Features: compact 10-feature movement/location/seasonality inputs, with no weather data
- Grouping: windows stay within `path_id`
- Identity: source bird ID embedding for neural models
- Split: chronological 80/20
- Environment: conda env `torch`

Run smoke validation:

```powershell
conda run -n torch python Triline-Transformer\run_path_experiment_no_weather.py --smoke --output-dir Path_Experiment_NoWeather_smoke
```

Run the full matrix:

```powershell
conda run -n torch python Triline-Transformer\run_path_experiment_no_weather.py --matrix full --output-dir Path_Experiment_NoWeather
```

## Analysis

The completed full run is saved in `Path_Experiment_NoWeather/`.

Key output files:

- `Path_Experiment_NoWeather/analysis.md`
- `Path_Experiment_NoWeather/comparison_summary.csv`
- `Path_Experiment_NoWeather/baseline_summary.csv`
- `Path_Experiment_NoWeather/direct_summary.csv`
- `Path_Experiment_NoWeather/triline_summary.csv`

The full matrix produced 68 result rows, with matching per-run `metrics.json`, `predictions.csv`, and `training_log.csv` artifacts.

Best mean GPS error in both threshold setups came from the persistence baseline at `k=120`:

| Setup | Model | k | Mean km | Median km |
|---|---|---:|---:|---:|
| `setup_10km` | persistence | 120 | 2.5204 | 0.1689 |
| `setup_30km` | persistence | 120 | 2.5204 | 0.1689 |

Best neural results by mean GPS error:

| Setup | Model | k | Mean km | Median km | Fly recall |
|---|---|---:|---:|---:|---:|
| `setup_10km` | Triline Transformer 2L | 120 | 3.0921 | 1.0278 | 0.4000 |
| `setup_30km` | Triline Transformer 2L | 120 | 2.8846 | 0.7350 | 1.0000 |

Interpretation: the long-context `k=120` split is unusually easy for mean GPS error, and even the persistence baseline dominates there. For model comparison, inspect `comparison_summary.csv` alongside sample counts and migration-day metrics, not only the global mean error.
