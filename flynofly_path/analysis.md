# Fly/No-Fly Rerun Summary

Generated with the conda `torch` environment:

`D:\Project\anaconda3\envs\torch\python.exe`

## Runs

- Original FlyNofly rerun: `flynofly_path/FlyNofly`
- Improved FlyNofly rerun: `flynofly_path/FlyNofly_Improved`

## Original FlyNofly

The original rerun matches the previous experiment setup: birds with at least 100 records, latitude 30-50, Sep-Dec records only, chronological 80/20 validation split, and a fixed fly/no-fly threshold of `0.5` on model probability.

Best original result:

| k | F1 | Precision | Recall | FP | Test fly rate |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.4646 | 0.3151 | 0.8846 | 200 | 0.0596 |

The rerun reproduced the same train/test sample counts as the previous `FlyNofly` folder, but the exact neural metrics changed because this run used the current PyTorch/CUDA stack.

## Improved FlyNofly

The improved rerun uses all-month context while restricting target days to Sep-Dec. It evaluates both chronological and stratified splits, then compares the Triline multi-task model against logistic regression, random forest, and movement-rule baselines.

Best improved result overall:

| Split | Model | k | F1 | Precision | Recall | FP | Test fly rate |
|---|---|---:|---:|---:|---:|---:|---:|
| chronological | triline_multitask | 10 | 0.6909 | 0.6738 | 0.7090 | 46 | 0.0693 |

Best stratified result:

| Split | Model | k | F1 | Precision | Recall | FP | Test fly rate |
|---|---|---:|---:|---:|---:|---:|---:|
| stratified | random_forest_balanced | 20 | 0.6189 | 0.6119 | 0.6260 | 52 | 0.0697 |

## Takeaway

The improved setup is substantially better than the original rerun on the main chronological estimate: F1 improves from `0.4646` to `0.6909`, while false positives drop from `200` to `46`. The main reason is that the improved windowing keeps pre-September movement context, which stabilizes the Sep-Dec target distribution across context lengths.

See the per-run analyses for full tables:

- `FlyNofly/analysis.md`
- `FlyNofly_Improved/analysis.md`
