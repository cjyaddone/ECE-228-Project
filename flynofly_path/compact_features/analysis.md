# Compact Fly/No-Fly Feature Analysis

## Purpose

This experiment tests whether the Fly/No-Fly task needs the full 40-feature input set and whether the neural model depends on bird identity. The compact model uses only 10 daily features and is run with and without the learned `bird_id` embedding.

The compact feature set is:

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

Each training example is a sequence of the previous `k` daily records:

```text
features: [batch, k, 10]
target: next-day fly/no-fly label, where fly = step_length_km > 30
```

The neural model is still the Triline Transformer: it predicts fly probability, log distance, and heading direction with a multi-task loss. The reported Fly/No-Fly metrics use the fly probability.

## Original Method

The Original method reproduces the older `FlyNofly` setup.

1. Start from `data/dataset2_daily_movement.csv`.
2. Keep birds with at least 100 original daily records.
3. Keep only records with latitude between 30 and 50 degrees.
4. Filter the data to Sep-Dec before windowing.
5. Build calendar-consecutive windows from this already filtered Sep-Dec data.
6. Use chronological 80/20 split by target date.
7. Train the Triline Transformer for each `k` in `10, 20, 30, 50`.
8. Evaluate with a fixed probability threshold of `0.5`.

This method is historically useful but has a weakness: because Sep-Dec filtering happens before window construction, longer context windows cannot use pre-September movement context. For example, a `k=50` window only sees Sep-Dec history, not the approach into autumn migration.

## Improved Method

The Improved method reproduces the newer `FlyNofly_Improved` setup.

1. Start from `data/dataset2_daily_movement.csv`.
2. Keep birds with at least 100 original daily records.
3. Keep records with latitude between 30 and 50 degrees.
4. Preserve all months as model context.
5. Build calendar-consecutive windows using all-month history.
6. Restrict only the target day to Sep-Dec.
7. Evaluate both chronological and stratified splits.
8. Train the Triline Transformer for each `k` in `10, 20, 30, 50`.
9. Select an operating threshold using a threshold sweep that maximizes F1.
10. Also report fixed-threshold `0.5` F1 for comparison.

This method is more realistic for migration behavior because summer and pre-migration context can inform Sep-Dec predictions. It also avoids the severe target fly-rate collapse seen by the Original method for longer `k`.

## Bird ID Variants

Two neural variants were tested for both methods:

- `compact_with_bird_id`: adds a learned bird embedding to every timestep, matching the original architecture.
- `compact_without_bird_id`: disables the bird embedding entirely. The model receives only compact movement, location, seasonality, and stopover features.

Prediction files still include `bird_id` for inspection. In the no-bird-ID variant, the ID is not fed into the neural model.

## Reference Best Full-Feature Results

These are the best full-feature rerun results from the previous experiment folders, included as reference points.

| Experiment | Split | Model | k | F1 | Precision | Recall | FP | Test fly rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Original full features | chronological | triline_multitask | 10 | 0.4646 | 0.3151 | 0.8846 | 200 | 0.0596 |
| Improved full features | chronological | triline_multitask | 10 | 0.6909 | 0.6738 | 0.7090 | 46 | 0.0693 |

## Original Compact Results

### With Bird ID

| k | Train | Test | Test fly rate | Accuracy | Precision | Recall | F1 | FP | FPR | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 6972 | 1744 | 0.0596 | 0.8991 | 0.3571 | 0.8654 | 0.5056 | 162 | 0.0988 | 0.9293 | 0.6636 |
| 20 | 6084 | 1522 | 0.0263 | 0.9093 | 0.1818 | 0.7000 | 0.2887 | 126 | 0.0850 | 0.8989 | 0.2798 |
| 30 | 5288 | 1322 | 0.0121 | 0.8366 | 0.0495 | 0.6875 | 0.0924 | 211 | 0.1616 | 0.7879 | 0.2296 |
| 50 | 3841 | 961 | 0.0083 | 0.9917 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.6364 | 0.0324 |

### Without Bird ID

| k | Train | Test | Test fly rate | Accuracy | Precision | Recall | F1 | FP | FPR | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 6972 | 1744 | 0.0596 | 0.8802 | 0.3158 | 0.8654 | 0.4627 | 195 | 0.1189 | 0.9433 | 0.6771 |
| 20 | 6084 | 1522 | 0.0263 | 0.9402 | 0.2661 | 0.7250 | 0.3893 | 80 | 0.0540 | 0.9170 | 0.4053 |
| 30 | 5288 | 1322 | 0.0121 | 0.8071 | 0.0456 | 0.7500 | 0.0860 | 251 | 0.1922 | 0.8018 | 0.0857 |
| 50 | 3841 | 961 | 0.0083 | 0.8876 | 0.0192 | 0.2500 | 0.0357 | 102 | 0.1070 | 0.5691 | 0.0138 |

## Improved Compact Neural Results

### With Bird ID

| Split | k | Train | Test | Test fly rate | Selected threshold | Accuracy | Precision | Recall | F1 | FP | FPR | Fixed 0.5 F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chronological | 10 | 7737 | 1935 | 0.0693 | 0.7299 | 0.9571 | 0.6604 | 0.7836 | 0.7167 | 54 | 0.0300 | 0.5665 | 0.9478 | 0.7019 |
| stratified | 10 | 7737 | 1935 | 0.0708 | 0.8948 | 0.9437 | 0.6186 | 0.5328 | 0.5725 | 45 | 0.0250 | 0.4615 | 0.8952 | 0.5478 |
| chronological | 20 | 7514 | 1879 | 0.0703 | 0.8013 | 0.9510 | 0.6333 | 0.7197 | 0.6738 | 55 | 0.0315 | 0.5124 | 0.9412 | 0.6804 |
| stratified | 20 | 7514 | 1879 | 0.0697 | 0.8107 | 0.9425 | 0.5742 | 0.6794 | 0.6224 | 66 | 0.0378 | 0.5411 | 0.9148 | 0.5779 |
| chronological | 30 | 7374 | 1844 | 0.0710 | 0.7797 | 0.9539 | 0.6438 | 0.7863 | 0.7079 | 57 | 0.0333 | 0.5786 | 0.9357 | 0.7088 |
| stratified | 30 | 7374 | 1844 | 0.0700 | 0.8371 | 0.9420 | 0.5846 | 0.5891 | 0.5869 | 54 | 0.0315 | 0.5206 | 0.9016 | 0.5422 |
| chronological | 50 | 7151 | 1788 | 0.0716 | 0.7630 | 0.9541 | 0.6494 | 0.7812 | 0.7092 | 54 | 0.0325 | 0.6407 | 0.9500 | 0.6753 |
| stratified | 50 | 7151 | 1788 | 0.0705 | 0.7914 | 0.9396 | 0.5643 | 0.6270 | 0.5940 | 61 | 0.0367 | 0.4694 | 0.8723 | 0.5206 |

### Without Bird ID

| Split | k | Train | Test | Test fly rate | Selected threshold | Accuracy | Precision | Recall | F1 | FP | FPR | Fixed 0.5 F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chronological | 10 | 7737 | 1935 | 0.0693 | 0.8376 | 0.9545 | 0.6494 | 0.7463 | 0.6944 | 54 | 0.0300 | 0.5707 | 0.9441 | 0.6715 |
| stratified | 10 | 7737 | 1935 | 0.0708 | 0.6835 | 0.9354 | 0.5370 | 0.6350 | 0.5819 | 75 | 0.0417 | 0.5243 | 0.8850 | 0.5431 |
| chronological | 20 | 7514 | 1879 | 0.0703 | 0.7808 | 0.9516 | 0.6358 | 0.7273 | 0.6784 | 55 | 0.0315 | 0.4496 | 0.9399 | 0.6643 |
| stratified | 20 | 7514 | 1879 | 0.0697 | 0.8714 | 0.9447 | 0.6063 | 0.5878 | 0.5969 | 50 | 0.0286 | 0.4866 | 0.9100 | 0.5712 |
| chronological | 30 | 7374 | 1844 | 0.0710 | 0.8673 | 0.9544 | 0.6643 | 0.7252 | 0.6934 | 48 | 0.0280 | 0.4947 | 0.9251 | 0.6699 |
| stratified | 30 | 7374 | 1844 | 0.0700 | 0.8632 | 0.9403 | 0.5693 | 0.6047 | 0.5865 | 59 | 0.0344 | 0.4889 | 0.8997 | 0.5430 |
| chronological | 50 | 7151 | 1788 | 0.0716 | 0.8899 | 0.9519 | 0.6346 | 0.7734 | 0.6972 | 57 | 0.0343 | 0.5775 | 0.9296 | 0.6645 |
| stratified | 50 | 7151 | 1788 | 0.0705 | 0.8530 | 0.9452 | 0.6250 | 0.5556 | 0.5882 | 42 | 0.0253 | 0.4646 | 0.8897 | 0.5372 |

## Improved Compact Baseline Results

The improved runner also evaluates logistic regression and random forest baselines on the same compact features. These baselines do not use the neural bird embedding, so the baseline outputs are the same for the with-bird-ID and without-bird-ID compact runs. The table below documents the baseline result set once.

| Model | Split | k | Train | Test | Test fly rate | Selected threshold | Precision | Recall | F1 | FP | FPR | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression_balanced | chronological | 10 | 7737 | 1935 | 0.0693 | 0.7851 | 0.6333 | 0.7090 | 0.6690 | 55 | 0.0305 | 0.9314 | 0.6676 |
| random_forest_balanced | chronological | 10 | 7737 | 1935 | 0.0693 | 0.2851 | 0.6623 | 0.7463 | 0.7018 | 51 | 0.0283 | 0.9471 | 0.7095 |
| logistic_regression_balanced | stratified | 10 | 7737 | 1935 | 0.0708 | 0.7814 | 0.5000 | 0.5547 | 0.5260 | 76 | 0.0423 | 0.8706 | 0.4796 |
| random_forest_balanced | stratified | 10 | 7737 | 1935 | 0.0708 | 0.4829 | 0.7340 | 0.5036 | 0.5974 | 25 | 0.0139 | 0.8868 | 0.5684 |
| logistic_regression_balanced | chronological | 20 | 7514 | 1879 | 0.0703 | 0.8960 | 0.6720 | 0.6364 | 0.6537 | 41 | 0.0235 | 0.9294 | 0.6612 |
| random_forest_balanced | chronological | 20 | 7514 | 1879 | 0.0703 | 0.2976 | 0.6993 | 0.7576 | 0.7273 | 43 | 0.0246 | 0.9509 | 0.7013 |
| logistic_regression_balanced | stratified | 20 | 7514 | 1879 | 0.0697 | 0.7713 | 0.4841 | 0.5802 | 0.5278 | 81 | 0.0463 | 0.8877 | 0.5082 |
| random_forest_balanced | stratified | 20 | 7514 | 1879 | 0.0697 | 0.4004 | 0.6783 | 0.5954 | 0.6341 | 37 | 0.0212 | 0.9147 | 0.5980 |
| logistic_regression_balanced | chronological | 30 | 7374 | 1844 | 0.0710 | 0.9003 | 0.6508 | 0.6260 | 0.6381 | 44 | 0.0257 | 0.9057 | 0.6350 |
| random_forest_balanced | chronological | 30 | 7374 | 1844 | 0.0710 | 0.3063 | 0.7218 | 0.7328 | 0.7273 | 37 | 0.0216 | 0.9525 | 0.7055 |
| logistic_regression_balanced | stratified | 30 | 7374 | 1844 | 0.0700 | 0.8440 | 0.4923 | 0.4961 | 0.4942 | 66 | 0.0385 | 0.8563 | 0.4833 |
| random_forest_balanced | stratified | 30 | 7374 | 1844 | 0.0700 | 0.2786 | 0.5923 | 0.5969 | 0.5946 | 53 | 0.0309 | 0.9070 | 0.5548 |
| logistic_regression_balanced | chronological | 50 | 7151 | 1788 | 0.0716 | 0.9150 | 0.6441 | 0.5938 | 0.6179 | 42 | 0.0253 | 0.8936 | 0.6101 |
| random_forest_balanced | chronological | 50 | 7151 | 1788 | 0.0716 | 0.3298 | 0.7176 | 0.7344 | 0.7259 | 37 | 0.0223 | 0.9520 | 0.6799 |
| logistic_regression_balanced | stratified | 50 | 7151 | 1788 | 0.0705 | 0.8950 | 0.5405 | 0.4762 | 0.5063 | 51 | 0.0307 | 0.8275 | 0.4542 |
| random_forest_balanced | stratified | 50 | 7151 | 1788 | 0.0705 | 0.2256 | 0.5638 | 0.6667 | 0.6109 | 65 | 0.0391 | 0.8979 | 0.5541 |

## Interpretation

- The compact feature set is not weaker than the full 40-feature neural setup in this rerun. The best improved compact neural result is `F1=0.7167`, compared with the previous improved full-feature best of `F1=0.6909`.
- Bird identity helps in the Original method at `k=10`, improving F1 from `0.4627` to `0.5056`, mostly by reducing false positives from `195` to `162`.
- In the Improved method, removing bird identity only slightly reduces the best chronological neural result: `0.7167` with bird ID vs `0.6972` without bird ID.
- The no-bird-ID result is still strong, which suggests the compact movement, location, seasonality, and stopover features capture much of the general behavior.
- Random forest baselines are very competitive on compact features. The best baseline result is chronological random forest at `k=20` or `k=30`, both with `F1=0.7273`.
- The Original method deteriorates for larger `k` because the Sep-Dec-only prefilter removes useful pre-September context and causes the test fly rate to collapse for long windows.

## Output Files

- Original compact with bird ID: `flynofly_path/compact_features/Original/compact_with_bird_id`
- Original compact without bird ID: `flynofly_path/compact_features/Original/compact_without_bird_id`
- Improved compact with bird ID: `flynofly_path/compact_features/Improved/compact_with_bird_id`
- Improved compact without bird ID: `flynofly_path/compact_features/Improved/compact_without_bird_id`
