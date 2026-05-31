# DistanceWeightTune V2

## Setup

- Dataset: southbound path segments.
- Features: compact 18-feature inputs with 10 path features and 8 weather features.
- Change under test: focused triline ablations with milder distance weights, distance-only weighting, movement-category conditioning, and balanced checkpoint selection.
- Weight configs: mild_bins, medium_bins, distance_only, current_bins.
- Direct models and baselines are unchanged controls.
- Weather match: exact bird/date join with nearest same-bird date fallback.
- Split: chronological 80/20 within constructed windows.
- Identity: source bird ID embedding for neural models.

## Best Mean GPS Error By Setup

| Setup | Model | Family | k | Mean km | Median km | Fly recall | Params |
|---|---|---|---:|---:|---:|---:|---:|
| setup_10km | triline_lstm_2l_k7__mild_bins | triline | 7 | 13.9704 | 2.2527 | 0.6419 | 301906 |
| setup_30km | triline_lstm_2l_k7__mild_bins | triline | 7 | 13.8812 | 2.0410 | 0.7368 | 301906 |

## Distance-Weighted vs Weather Baseline

| Setup | Model | Family | k | Weighted Mean km | Weather Mean km | Mean Delta km | Migration50 Delta km | Improved |
|---|---|---|---:|---:|---:|---:|---:|---|
| setup_10km | triline_lstm_2l_k7__mild_bins | triline | 7 | 13.9704 | 13.3131 | 0.6573 | 1.1209 | no |
| setup_10km | persistence | baseline | 7 | 14.9233 | 14.9233 | 0.0000 | 0.0000 | no |
| setup_10km | triline_transformer_2l_k7__distance_only | triline | 7 | 16.1586 | 13.2526 | 2.9060 | 1.4708 | no |
| setup_10km | persistence | baseline | 30 | 16.3394 | 16.3394 | 0.0000 | 0.0000 | no |
| setup_10km | const_velocity | baseline | 7 | 17.8171 | 17.8171 | 0.0000 | -0.0000 | no |
| setup_10km | direct_transformer_2l_k7 | direct | 7 | 18.8140 | 14.2745 | 4.5394 | 2.6136 | no |
| setup_10km | const_velocity | baseline | 30 | 19.5423 | 19.5423 | 0.0000 | 0.0000 | no |
| setup_30km | triline_lstm_2l_k7__mild_bins | triline | 7 | 13.8812 | 13.1441 | 0.7371 | 3.4949 | no |
| setup_30km | persistence | baseline | 7 | 14.9233 | 14.9233 | 0.0000 | 0.0000 | no |
| setup_30km | triline_transformer_2l_k7__distance_only | triline | 7 | 15.3369 | 13.5499 | 1.7870 | 5.1832 | no |
| setup_30km | persistence | baseline | 30 | 16.3394 | 16.3394 | 0.0000 | 0.0000 | no |
| setup_30km | const_velocity | baseline | 7 | 17.8171 | 17.8171 | 0.0000 | -0.0000 | no |
| setup_30km | direct_transformer_2l_k7 | direct | 7 | 18.8140 | 14.2745 | 4.5394 | 2.6136 | no |
| setup_30km | const_velocity | baseline | 30 | 19.5423 | 19.5423 | 0.0000 | 0.0000 | no |

Distance-weighted triline rows improving migration50 error: 0/4.

## Full Comparison

| Setup | Family | Model | k | Mean km | Median km | P90 km | Fly Mean km | Migration50 Mean km | Migration100 Mean km |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| setup_10km | baseline | const_velocity | 7 | 17.8171 | 1.7637 | 54.1154 | 89.9617 | 116.0993 | 138.2874 |
| setup_10km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 93.6197 | 143.1997 | 184.7018 |
| setup_10km | baseline | const_velocity | 30 | 19.5423 | 2.2817 | 61.8870 | 92.8597 | 119.9864 | 141.6438 |
| setup_10km | baseline | persistence | 30 | 16.3394 | 0.8083 | 40.9240 | 94.7498 | 146.6291 | 189.8530 |
| setup_10km | direct | direct_transformer_2l_k7 | 7 | 18.8140 | 8.4028 | 48.7031 | 69.6287 | 88.2469 | 105.8861 |
| setup_10km | triline | triline_lstm_2l_k7__mild_bins | 7 | 13.9704 | 2.2527 | 38.9466 | 74.5100 | 107.4790 | 141.4262 |
| setup_10km | triline | triline_transformer_2l_k7__distance_only | 7 | 16.1586 | 3.2446 | 46.4500 | 73.3371 | 105.5639 | 141.6074 |
| setup_30km | baseline | const_velocity | 7 | 17.8171 | 1.7637 | 54.1154 | 109.1717 | 116.0993 | 138.2874 |
| setup_30km | baseline | persistence | 7 | 14.9233 | 0.7386 | 33.2401 | 125.0660 | 143.1997 | 184.7018 |
| setup_30km | baseline | const_velocity | 30 | 19.5423 | 2.2817 | 61.8870 | 113.1596 | 119.9864 | 141.6438 |
| setup_30km | baseline | persistence | 30 | 16.3394 | 0.8083 | 40.9240 | 128.0119 | 146.6291 | 189.8530 |
| setup_30km | direct | direct_transformer_2l_k7 | 7 | 18.8140 | 8.4028 | 48.7031 | 83.0589 | 88.2469 | 105.8861 |
| setup_30km | triline | triline_lstm_2l_k7__mild_bins | 7 | 13.8812 | 2.0410 | 38.7205 | 95.7464 | 108.2629 | 142.7573 |
| setup_30km | triline | triline_transformer_2l_k7__distance_only | 7 | 15.3369 | 2.4776 | 39.6321 | 99.1735 | 113.1100 | 151.4122 |

## Training Logs

- Summarized 6 training logs in `training_log_summary.csv`.
- Weighted and unweighted triline loss columns are included for auditability.

## Rollout Visualization

- Interactive rollout explorer: `rollout_all/index.html`.
- Rollout paths: 40; skipped paths: 0.
- Rollout comparison rows against weather baseline: 40.

## Notes

- Triline rows report ungated GPS reconstruction as primary metrics.
- Gated GPS metrics are stored in each triline `metrics.json` and the CSV summaries with `gated_` prefixes.
- Triline rollout is fly-gated: days below the tuned fly probability threshold stay at the current location; fly days use predicted distance and heading.
- Direct model rows repeat per setup because setup-specific fly thresholds change stratified error slices.
- Autoregressive rollout uses actual future daily weather as exogenous forecast/oracle weather.
- Lower delta values in comparison files mean distance weighting improved over the original weather run.
