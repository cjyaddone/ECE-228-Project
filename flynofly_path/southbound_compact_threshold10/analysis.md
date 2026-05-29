# Southbound Compact Fly/No-Fly Analysis

## Setup

This experiment uses the selected southbound path dataset:

`D:\UCSD\ECE 228\ECE-228-Project\data\filtered\dataset2_southbound_paths_jun_dec_lat30_50_min50_drop5.csv`

The dataset contains daily records from extracted southbound migration path segments. A path is a bird-year segment between a northern anchor and a later southern anchor with at least a 5 degree latitude drop, at least 50 path rows, and latitude constrained to 30-50 degrees.

- Rows: 8180
- Paths: 82
- Source birds: 33
- Dates: 2013-07-30 00:00:00+00:00 to 2023-12-18 00:00:00+00:00
- Fly threshold: 10 km/day
- Row fly rate at threshold: 0.1845

## Difference From Prior General Experiment

The prior compact experiment used the broader daily movement dataset and predicted Sep-Dec target days after generic latitude and bird-count filtering. This southbound experiment uses only pre-extracted southbound migration paths and uses all Jun-Dec target days present inside those paths. Windows are grouped by `path_id`, so a sequence never crosses from one path/year into another.

## Model And Features

Each example uses the previous `k` days to predict whether the next day is a fly day:

```text
features: [batch, k, 10]
target: next-day step_length_km > 10
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
| compact_with_bird_id | chronological | 10 | 5708 | 1427 | 0.1479 | 0.6951 | 0.9201 | 0.7462 | 0.6967 | 0.7206 | 50 | 0.0411 | 0.6806 | 0.9114 | 0.7292 |
| compact_with_bird_id | stratified | 10 | 5708 | 1427 | 0.1815 | 0.7720 | 0.8928 | 0.7477 | 0.6178 | 0.6765 | 54 | 0.0462 | 0.6442 | 0.8810 | 0.7016 |
| compact_with_bird_id | chronological | 20 | 4999 | 1250 | 0.1632 | 0.5096 | 0.9144 | 0.7366 | 0.7402 | 0.7384 | 54 | 0.0516 | 0.7379 | 0.9121 | 0.7405 |
| compact_with_bird_id | stratified | 20 | 4999 | 1250 | 0.1784 | 0.6579 | 0.8752 | 0.6489 | 0.6547 | 0.6518 | 79 | 0.0769 | 0.6210 | 0.8628 | 0.6564 |
| compact_with_bird_id | chronological | 30 | 4328 | 1082 | 0.1571 | 0.7057 | 0.9177 | 0.7647 | 0.6882 | 0.7245 | 36 | 0.0395 | 0.6925 | 0.9021 | 0.7314 |
| compact_with_bird_id | stratified | 30 | 4328 | 1082 | 0.1738 | 0.6511 | 0.8771 | 0.6382 | 0.6755 | 0.6563 | 72 | 0.0805 | 0.6042 | 0.8715 | 0.6670 |
| compact_with_bird_id | chronological | 50 | 3033 | 759 | 0.1502 | 0.7624 | 0.9236 | 0.7692 | 0.7018 | 0.7339 | 24 | 0.0372 | 0.6946 | 0.8734 | 0.7246 |
| compact_with_bird_id | stratified | 50 | 3033 | 759 | 0.1713 | 0.6935 | 0.8880 | 0.6991 | 0.6077 | 0.6502 | 34 | 0.0541 | 0.6199 | 0.8595 | 0.6927 |
| compact_without_bird_id | chronological | 10 | 5708 | 1427 | 0.1479 | 0.6042 | 0.9166 | 0.7233 | 0.7062 | 0.7146 | 57 | 0.0469 | 0.6996 | 0.8997 | 0.7311 |
| compact_without_bird_id | stratified | 10 | 5708 | 1427 | 0.1815 | 0.6944 | 0.8928 | 0.7227 | 0.6641 | 0.6922 | 66 | 0.0565 | 0.6575 | 0.8881 | 0.7009 |
| compact_without_bird_id | chronological | 20 | 4999 | 1250 | 0.1632 | 0.6550 | 0.9128 | 0.7317 | 0.7353 | 0.7335 | 55 | 0.0526 | 0.7196 | 0.9004 | 0.7391 |
| compact_without_bird_id | stratified | 20 | 4999 | 1250 | 0.1784 | 0.7416 | 0.8816 | 0.6812 | 0.6323 | 0.6558 | 66 | 0.0643 | 0.6128 | 0.8613 | 0.6804 |
| compact_without_bird_id | chronological | 30 | 4328 | 1082 | 0.1571 | 0.6953 | 0.9150 | 0.7438 | 0.7000 | 0.7212 | 41 | 0.0450 | 0.7052 | 0.8829 | 0.7206 |
| compact_without_bird_id | stratified | 30 | 4328 | 1082 | 0.1738 | 0.6853 | 0.8826 | 0.6631 | 0.6596 | 0.6613 | 63 | 0.0705 | 0.6194 | 0.8619 | 0.6623 |
| compact_without_bird_id | chronological | 50 | 3033 | 759 | 0.1502 | 0.6935 | 0.9223 | 0.7670 | 0.6930 | 0.7281 | 24 | 0.0372 | 0.6926 | 0.8792 | 0.7238 |
| compact_without_bird_id | stratified | 50 | 3033 | 759 | 0.1713 | 0.6254 | 0.8735 | 0.6214 | 0.6692 | 0.6444 | 53 | 0.0843 | 0.6195 | 0.8655 | 0.6859 |

## All Baseline Results

The compact baselines use the same 10 features. Rolling-rule baselines are skipped because the compact feature set does not include rolling mean/max features.

| Model | Split | k | Train | Test | Test fly rate | Threshold | Precision | Recall | F1 | FP | FPR | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression_balanced | chronological | 10 | 5708 | 1427 | 0.1479 | 0.6388 | 0.7353 | 0.7109 | 0.7229 | 54 | 0.0444 | 0.8872 | 0.7137 |
| random_forest_balanced | chronological | 10 | 5708 | 1427 | 0.1479 | 0.3801 | 0.7565 | 0.6919 | 0.7228 | 47 | 0.0387 | 0.9115 | 0.7585 |
| logistic_regression_balanced | stratified | 10 | 5708 | 1427 | 0.1815 | 0.6239 | 0.6855 | 0.6564 | 0.6706 | 78 | 0.0668 | 0.8729 | 0.6789 |
| random_forest_balanced | stratified | 10 | 5708 | 1427 | 0.1815 | 0.4323 | 0.7661 | 0.6448 | 0.7002 | 51 | 0.0437 | 0.8963 | 0.7542 |
| logistic_regression_balanced | chronological | 20 | 4999 | 1250 | 0.1632 | 0.6582 | 0.7087 | 0.7157 | 0.7122 | 60 | 0.0574 | 0.8797 | 0.7128 |
| random_forest_balanced | chronological | 20 | 4999 | 1250 | 0.1632 | 0.3066 | 0.7426 | 0.7353 | 0.7389 | 52 | 0.0497 | 0.9162 | 0.7762 |
| logistic_regression_balanced | stratified | 20 | 4999 | 1250 | 0.1784 | 0.6088 | 0.6140 | 0.6278 | 0.6208 | 88 | 0.0857 | 0.8572 | 0.6160 |
| random_forest_balanced | stratified | 20 | 4999 | 1250 | 0.1784 | 0.3379 | 0.6901 | 0.6592 | 0.6743 | 66 | 0.0643 | 0.8861 | 0.7104 |
| logistic_regression_balanced | chronological | 30 | 4328 | 1082 | 0.1571 | 0.6707 | 0.6494 | 0.6647 | 0.6570 | 61 | 0.0669 | 0.8502 | 0.6646 |
| random_forest_balanced | chronological | 30 | 4328 | 1082 | 0.1571 | 0.3647 | 0.7793 | 0.6647 | 0.7175 | 32 | 0.0351 | 0.8999 | 0.7415 |
| logistic_regression_balanced | stratified | 30 | 4328 | 1082 | 0.1738 | 0.6866 | 0.5977 | 0.5532 | 0.5746 | 70 | 0.0783 | 0.8211 | 0.5865 |
| random_forest_balanced | stratified | 30 | 4328 | 1082 | 0.1738 | 0.2746 | 0.6485 | 0.6968 | 0.6718 | 71 | 0.0794 | 0.8813 | 0.6685 |
| logistic_regression_balanced | chronological | 50 | 3033 | 759 | 0.1502 | 0.9001 | 0.8026 | 0.5351 | 0.6421 | 15 | 0.0233 | 0.8359 | 0.6693 |
| random_forest_balanced | chronological | 50 | 3033 | 759 | 0.1502 | 0.4216 | 0.8022 | 0.6404 | 0.7122 | 18 | 0.0279 | 0.8951 | 0.6957 |
| logistic_regression_balanced | stratified | 50 | 3033 | 759 | 0.1713 | 0.7953 | 0.6667 | 0.4923 | 0.5664 | 32 | 0.0509 | 0.8129 | 0.6030 |
| random_forest_balanced | stratified | 50 | 3033 | 759 | 0.1713 | 0.3381 | 0.7333 | 0.5923 | 0.6553 | 28 | 0.0445 | 0.8840 | 0.6936 |

## Interpretation

- Best neural result: `compact_with_bird_id` chronological k=20 with F1=0.7384, precision=0.7366, recall=0.7402, FP=54.
- Best baseline result: `random_forest_balanced` chronological k=20 with F1=0.7389, precision=0.7426, recall=0.7353, FP=52.
- The southbound dataset has a higher fly-day rate than the previous general Sep-Dec setup, because it is restricted to migration-path segments.
- Comparing with-ID and without-ID rows indicates how much source-bird identity helps beyond compact movement/location/seasonality features.

## Output Files

- `compact_with_bird_id/summary.csv`: neural results with source bird ID embedding.
- `compact_without_bird_id/summary.csv`: neural results without identity embedding.
- `comparison_summary.csv`: neural and baseline rows for each variant.
- Per-split folders contain metrics, predictions, threshold sweeps, training logs, and checkpoints.
