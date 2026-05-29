# Path Experiment No Weather

## Summary
Implement a no-weather southbound trajectory-prediction experiment that uses the existing best southbound compact setup conventions: southbound path dataset, compact 10-feature inputs, `path_id` window grouping, source bird ID embeddings, chronological 80/20 splits, seed `42`, and conda env `torch`.

Run the full `Experiment.md` model comparison for both fly-threshold setups:
- `southbound_compact`: 30 km/day fly label
- `southbound_compact_threshold10`: 10 km/day fly label

All logs, checkpoints, predictions, metrics, summaries, and analysis will be saved under `Path_Experiment_NoWeather`.

## Key Changes
- Add a new experiment runner, e.g. `Triline-Transformer/run_path_experiment_no_weather.py`, reusing existing southbound preprocessing logic.
- Build trajectory windows grouped by `path_id`, with no weather columns:
  `lat_median`, `lon_median`, `delta_lat`, `delta_lon`, `step_length_km`, `heading_sin`, `heading_cos`, `doy_sin`, `doy_cos`, `stopover_duration_days`.
- Add/extend model definitions for:
  - Direct MLP last-day, k=1
  - Direct MLP flattened sequence, k=30
  - Direct Transformer, 2 layers
  - Triline Transformer, 2/3/4 layers
  - Triline LSTM, 2 layers
  - Triline LinearAR full features
  - Triline LinearAR delta-only
  - Persistence and constant-velocity baselines
- Direct models predict normalized next-day `[delta_lat, delta_lon]`.
- Triline models predict fly logit, `log1p(distance_km)`, and heading `[sin, cos]`, then reconstruct GPS from the last input location.
- Primary GPS metrics use ungated triline distance reconstruction; secondary gated metrics will also be saved using fly probability thresholding.

## Experiment Matrix
For each setup, run:
- Baselines: k=`7,14,30`
- Direct Transformer 2L: k=`7,14,30`
- Direct MLP last-day: k=`1`
- Direct MLP flattened: k=`30`
- Triline Transformer 2L: k=`7,14,30`
- Triline Transformer 3L: k=`30`
- Triline Transformer 4L: k=`7,14,30`
- Triline LSTM 2L: k=`7,14,30`
- Triline LinearAR full: k=`7,14,30`
- Triline LinearAR delta-only: k=`30`

Training config:
- batch size `128`
- AdamW `lr=1e-3`, `weight_decay=1e-4`
- ReduceLROnPlateau `factor=0.5`, `patience=4`
- early stopping patience `20`
- max epochs `100`
- triline loss weights: fly `1.0`, distance `1.0`, direction `0.5`
- BCE `pos_weight` computed from train split and capped at `10.0`
- device auto-selects CUDA; current `torch` env has CUDA available on RTX 5090

## Outputs
Create this structure:
- `Path_Experiment_NoWeather/run_config.json`
- `Path_Experiment_NoWeather/data_summary.json`
- `Path_Experiment_NoWeather/run.log`
- `Path_Experiment_NoWeather/setup_30km/...`
- `Path_Experiment_NoWeather/setup_10km/...`
- per model: `training_log.csv`, `metrics.json`, `predictions.csv`, `best_model.pt`
- root summaries:
  - `comparison_summary.csv`
  - `baseline_summary.csv`
  - `direct_summary.csv`
  - `triline_summary.csv`
  - `analysis.md`

Metrics will include mean/median/p90/p95 haversine error, stationary/fly-day error by setup threshold, 50 km migration error, direct delta loss, triline fly precision/recall/F1, parameter count, best epoch, train/test sample counts, and runtime.

## Test Plan
- Add a smoke mode that builds windows for k=7, runs one mini forward pass for every model family, reconstructs GPS, and writes temporary metrics.
- Run:
  `conda run -n torch python Triline-Transformer/run_path_experiment_no_weather.py --smoke --output-dir Path_Experiment_NoWeather_smoke`
- Then run the full experiment:
  `conda run -n torch python Triline-Transformer/run_path_experiment_no_weather.py --matrix full --output-dir Path_Experiment_NoWeather`
- Validate that every planned model row appears in `comparison_summary.csv`, every model directory has metrics/predictions/logs, and `analysis.md` identifies the best model per setup.

## Assumptions
- “Best setup” means reuse the southbound compact dataset design and the best-performing existing conventions: compact no-weather features, bird ID enabled, chronological split, and source bird ID rather than `path_id` identity.
- The full matrix is run for both 30 km and 10 km fly-label setups.
- Existing dirty worktree changes are preserved; implementation will add new files or make scoped extensions without reverting prior work.

## Visualization: Autoregressive Path Reconstruction

Add an evaluation/visualization mode that selects one representative path from the test set and compares the real GPS trajectory against autoregressive predictions from the best models.

### Goal

Visualize how well the best models can roll out a full trajectory when only the first input window is given.

For each setup:

- Select one `path_id` from the test split.
- Use the first `k=30` observed days as the initial context.
- Preserve and plot the original 30-day input window.
- Autoregressively predict the remaining path one day at a time.
- Overlay:
  - ground-truth GPS trajectory
  - initial observed 30-day context
  - predicted trajectory from the best Direct model
  - predicted trajectory from the best Triline model
  - optional persistence / constant-velocity baseline

### Path Selection

Choose one test path satisfying:

- belongs entirely to the chronological test split
- has at least `30 + N` valid daily points, where `N` is the rollout length
- preferably contains meaningful migration movement, e.g. total path distance greater than `50 km`
- not dominated by stationary days

Default selection rule:

```text
Pick the longest valid test path with total displacement >= 50 km.
