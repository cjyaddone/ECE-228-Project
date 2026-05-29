# Southbound Compact Fly/No-Fly Analysis

## Setup

This experiment uses the selected southbound path dataset:

`D:\UCSD\ECE 228\ECE-228-Project\data\filtered\dataset2_southbound_paths_jun_dec_lat30_50_min50_drop5.csv`

The dataset contains daily records from extracted southbound migration path segments. A path is a bird-year segment between a northern anchor and a later southern anchor with at least a 5 degree latitude drop, at least 50 path rows, and latitude constrained to 30-50 degrees.

- Rows: 8180
- Paths: 82
- Source birds: 33
- Dates: 2013-07-30 00:00:00+00:00 to 2023-12-18 00:00:00+00:00
- Row fly rate at 30 km/day: 0.1286

## Difference From Prior General Experiment

The prior compact experiment used the broader daily movement dataset and predicted Sep-Dec target days after generic latitude and bird-count filtering. This southbound experiment uses only pre-extracted southbound migration paths and uses all Jun-Dec target days present inside those paths. Windows are grouped by `path_id`, so a sequence never crosses from one path/year into another.

## Model And Features

Each example uses the previous `k` days to predict whether the next day is a fly day:

```text
features: [batch, k, 10]
target: next-day step_length_km > 30
```

Compact daily features:

```text
lat_median
lon_median
delta_lat
delta_lon
step_length_km
heading_sin
heading_cos
doy_sin
doy_cos
stopover_duration_days
```

Two neural variants were tested:

- `compact_with_bird_id`: source bird ID embedding is enabled.
- `compact_without_bird_id`: identity embedding is disabled.

The with-ID variant uses `source_individual_local_identifier`, not `path_id`, to avoid leaking a unique path/year identifier.

## All Neural Results

| Variant | Split | k | Train | Test | Test fly rate | Threshold | Accuracy | Precision | Recall | F1 | FP | FPR | Fixed 0.5 F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact_with_bird_id | chronological | 10 | 5708 | 1427 | 0.1051 | 0.6501 | 0.9355 | 0.6510 | 0.8333 | 0.7310 | 67 | 0.0525 | 0.7100 | 0.9429 | 0.7336 |
| compact_with_bird_id | stratified | 10 | 5708 | 1427 | 0.1261 | 0.7868 | 0.9222 | 0.6667 | 0.7667 | 0.7132 | 69 | 0.0553 | 0.6260 | 0.9267 | 0.7272 |
| compact_with_bird_id | chronological | 20 | 4999 | 1250 | 0.1176 | 0.7627 | 0.9368 | 0.7024 | 0.8027 | 0.7492 | 50 | 0.0453 | 0.7221 | 0.9495 | 0.7405 |
| compact_with_bird_id | stratified | 20 | 4999 | 1250 | 0.1200 | 0.8389 | 0.9256 | 0.6792 | 0.7200 | 0.6990 | 51 | 0.0464 | 0.5781 | 0.9022 | 0.6947 |
| compact_with_bird_id | chronological | 30 | 4328 | 1082 | 0.1100 | 0.8324 | 0.9381 | 0.6781 | 0.8319 | 0.7472 | 47 | 0.0488 | 0.6624 | 0.9243 | 0.6978 |
| compact_with_bird_id | stratified | 30 | 4328 | 1082 | 0.1146 | 0.8053 | 0.9242 | 0.6382 | 0.7823 | 0.7029 | 55 | 0.0574 | 0.5960 | 0.9253 | 0.7030 |
| compact_with_bird_id | chronological | 50 | 3033 | 759 | 0.1120 | 0.7507 | 0.9407 | 0.6852 | 0.8706 | 0.7668 | 34 | 0.0504 | 0.7037 | 0.9387 | 0.7167 |
| compact_with_bird_id | stratified | 50 | 3033 | 759 | 0.1133 | 0.7816 | 0.9209 | 0.6226 | 0.7674 | 0.6875 | 40 | 0.0594 | 0.5966 | 0.9282 | 0.7227 |
| compact_without_bird_id | chronological | 10 | 5708 | 1427 | 0.1051 | 0.8271 | 0.9432 | 0.7018 | 0.8000 | 0.7477 | 51 | 0.0399 | 0.6975 | 0.9448 | 0.7479 |
| compact_without_bird_id | stratified | 10 | 5708 | 1427 | 0.1261 | 0.8710 | 0.9327 | 0.7593 | 0.6833 | 0.7193 | 39 | 0.0313 | 0.6464 | 0.9204 | 0.7374 |
| compact_without_bird_id | chronological | 20 | 4999 | 1250 | 0.1176 | 0.5611 | 0.9280 | 0.6432 | 0.8707 | 0.7399 | 71 | 0.0644 | 0.7232 | 0.9463 | 0.7371 |
| compact_without_bird_id | stratified | 20 | 4999 | 1250 | 0.1200 | 0.8304 | 0.9280 | 0.6923 | 0.7200 | 0.7059 | 48 | 0.0436 | 0.6406 | 0.9116 | 0.7024 |
| compact_without_bird_id | chronological | 30 | 4328 | 1082 | 0.1100 | 0.8282 | 0.9353 | 0.6690 | 0.8151 | 0.7348 | 48 | 0.0498 | 0.7138 | 0.9377 | 0.7158 |
| compact_without_bird_id | stratified | 30 | 4328 | 1082 | 0.1146 | 0.7584 | 0.9251 | 0.6352 | 0.8145 | 0.7138 | 58 | 0.0605 | 0.5989 | 0.9266 | 0.7113 |
| compact_without_bird_id | chronological | 50 | 3033 | 759 | 0.1120 | 0.7348 | 0.9354 | 0.6667 | 0.8471 | 0.7461 | 36 | 0.0534 | 0.6878 | 0.9323 | 0.7687 |
| compact_without_bird_id | stratified | 50 | 3033 | 759 | 0.1133 | 0.7245 | 0.9223 | 0.6262 | 0.7791 | 0.6943 | 40 | 0.0594 | 0.6256 | 0.9204 | 0.6783 |

## All Baseline Results

The compact baselines use the same 10 features. Rolling-rule baselines are skipped because the compact feature set does not include rolling mean/max features.

| Model | Split | k | Train | Test | Test fly rate | Threshold | Precision | Recall | F1 | FP | FPR | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression_balanced | chronological | 10 | 5708 | 1427 | 0.1051 | 0.6756 | 0.6359 | 0.7800 | 0.7006 | 67 | 0.0525 | 0.9334 | 0.6771 |
| random_forest_balanced | chronological | 10 | 5708 | 1427 | 0.1051 | 0.3271 | 0.6971 | 0.8133 | 0.7508 | 53 | 0.0415 | 0.9548 | 0.7902 |
| logistic_regression_balanced | stratified | 10 | 5708 | 1427 | 0.1261 | 0.7200 | 0.7069 | 0.6833 | 0.6949 | 51 | 0.0409 | 0.8997 | 0.6931 |
| random_forest_balanced | stratified | 10 | 5708 | 1427 | 0.1261 | 0.3629 | 0.7098 | 0.7611 | 0.7346 | 56 | 0.0449 | 0.9384 | 0.7485 |
| logistic_regression_balanced | chronological | 20 | 4999 | 1250 | 0.1176 | 0.8251 | 0.6986 | 0.6939 | 0.6962 | 44 | 0.0399 | 0.9112 | 0.6691 |
| random_forest_balanced | chronological | 20 | 4999 | 1250 | 0.1176 | 0.2510 | 0.6702 | 0.8571 | 0.7522 | 62 | 0.0562 | 0.9548 | 0.7820 |
| logistic_regression_balanced | stratified | 20 | 4999 | 1250 | 0.1200 | 0.7452 | 0.6713 | 0.6400 | 0.6553 | 47 | 0.0427 | 0.8968 | 0.6601 |
| random_forest_balanced | stratified | 20 | 4999 | 1250 | 0.1200 | 0.2642 | 0.6484 | 0.7867 | 0.7108 | 64 | 0.0582 | 0.9107 | 0.7120 |
| logistic_regression_balanced | chronological | 30 | 4328 | 1082 | 0.1100 | 0.8900 | 0.7113 | 0.5798 | 0.6389 | 28 | 0.0291 | 0.8636 | 0.6043 |
| random_forest_balanced | chronological | 30 | 4328 | 1082 | 0.1100 | 0.3311 | 0.6992 | 0.7227 | 0.7107 | 37 | 0.0384 | 0.9482 | 0.7536 |
| logistic_regression_balanced | stratified | 30 | 4328 | 1082 | 0.1146 | 0.7003 | 0.6015 | 0.6452 | 0.6226 | 53 | 0.0553 | 0.8738 | 0.5928 |
| random_forest_balanced | stratified | 30 | 4328 | 1082 | 0.1146 | 0.2861 | 0.6794 | 0.7177 | 0.6980 | 42 | 0.0438 | 0.9290 | 0.7401 |
| logistic_regression_balanced | chronological | 50 | 3033 | 759 | 0.1120 | 0.9713 | 0.7869 | 0.5647 | 0.6575 | 13 | 0.0193 | 0.8726 | 0.6061 |
| random_forest_balanced | chronological | 50 | 3033 | 759 | 0.1120 | 0.2054 | 0.6545 | 0.8471 | 0.7385 | 38 | 0.0564 | 0.9441 | 0.7404 |
| logistic_regression_balanced | stratified | 50 | 3033 | 759 | 0.1133 | 0.9400 | 0.6721 | 0.4767 | 0.5578 | 20 | 0.0297 | 0.8459 | 0.5599 |
| random_forest_balanced | stratified | 50 | 3033 | 759 | 0.1133 | 0.3032 | 0.6739 | 0.7209 | 0.6966 | 30 | 0.0446 | 0.9226 | 0.6810 |

## Interpretation

- Best neural result: `compact_with_bird_id` chronological k=50 with F1=0.7668, precision=0.6852, recall=0.8706, FP=34.
- Best baseline result: `random_forest_balanced` chronological k=20 with F1=0.7522, precision=0.6702, recall=0.8571, FP=62.
- The southbound dataset has a higher fly-day rate than the previous general Sep-Dec setup, because it is restricted to migration-path segments.
- Comparing with-ID and without-ID rows indicates how much source-bird identity helps beyond compact movement/location/seasonality features.

## Output Files

- `compact_with_bird_id/summary.csv`: neural results with source bird ID embedding.
- `compact_without_bird_id/summary.csv`: neural results without identity embedding.
- `comparison_summary.csv`: neural and baseline rows for each variant.
- Per-split folders contain metrics, predictions, threshold sweeps, training logs, and checkpoints.
