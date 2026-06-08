# Final Path Experiment

Clean, reproducible path-prediction experiment workspace.

## Matrix

- Tests: `with_weather`, `without_weather`
- Fly threshold: `10 km` only
- Context lengths: `k = 7, 14, 30`
- Models per k:
  - `direct_mlp_sequence`
  - `direct_transformer_2l`
  - `triline_lstm_2l`
  - `triline_transformer_2l`

The cleaned weather input is copied to:

```text
Final_Path_Experiment/data/combined_southbound_paths_with_weather_matched.csv
```

## Run

Smoke test:

```powershell
python Final_Path_Experiment/scripts/run_final_path_experiment.py --mode smoke --device cpu
```

Full run:

```powershell
python Final_Path_Experiment/scripts/run_final_path_experiment.py --mode full --device auto --max-epochs 100 --patience 20
```

## Outputs

Each test writes:

- `comparison_summary.csv`
- `direct_summary.csv`
- `triline_summary.csv`
- `run_config.json`
- `analysis.md`

The top-level comparison is:

```text
Final_Path_Experiment/weather_vs_noweather_comparison.csv
```
