# Improved TestFlyNofly Analysis

## Data

- Source rows/birds: 34421 rows, 92 birds
- All-month context rows/birds: 29977 rows, 46 birds
- Sep-Dec target candidates: 10307 rows
- Sep-Dec candidate fly rate: 0.0739
- Context is allowed before September; only target days are restricted to Sep-Dec.

## Comparison To Original

- Original best F1 was k=10 with F1=0.5263, precision=0.3782, recall=0.8654.
- Improved best overall: random_forest_balanced chronological k=20 with F1=0.7273, precision=0.6993, recall=0.7576, FP=43.

## Best By Split

| Split | Model | k | F1 | Precision | Recall | FP | FPR | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chronological | random_forest_balanced | 30 | 0.7273 | 0.7218 | 0.7328 | 37 | 0.0216 | 0.9525 | 0.7055 |
| stratified | random_forest_balanced | 20 | 0.6341 | 0.6783 | 0.5954 | 37 | 0.0212 | 0.9147 | 0.5980 |

## Triline Multi-Task Results

| Split | k | Test fly rate | Threshold | F1 | Precision | Recall | FP | FPR | Fixed-0.5 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chronological | 10 | 0.0693 | 0.8376 | 0.6944 | 0.6494 | 0.7463 | 54 | 0.0300 | 0.5707 |
| chronological | 20 | 0.0703 | 0.7808 | 0.6784 | 0.6358 | 0.7273 | 55 | 0.0315 | 0.4496 |
| chronological | 30 | 0.0710 | 0.8673 | 0.6934 | 0.6643 | 0.7252 | 48 | 0.0280 | 0.4947 |
| chronological | 50 | 0.0716 | 0.8899 | 0.6972 | 0.6346 | 0.7734 | 57 | 0.0343 | 0.5775 |
| stratified | 10 | 0.0708 | 0.6835 | 0.5819 | 0.5370 | 0.6350 | 75 | 0.0417 | 0.5243 |
| stratified | 20 | 0.0697 | 0.8714 | 0.5969 | 0.6063 | 0.5878 | 50 | 0.0286 | 0.4866 |
| stratified | 30 | 0.0700 | 0.8632 | 0.5865 | 0.5693 | 0.6047 | 59 | 0.0344 | 0.4889 |
| stratified | 50 | 0.0705 | 0.8530 | 0.5882 | 0.6250 | 0.5556 | 42 | 0.0253 | 0.4646 |

## Interpretation

- Target-only Sep-Dec windowing restores the chronological test fly rate to roughly 7% for all k, instead of below 1-3% for longer contexts.
- Threshold tuning is essential because class-weighted training does not imply that 0.5 is the right operating point.
- If baselines beat the Transformer for a split, the task is currently driven more by hand-engineered movement history than by attention over daily tokens.
- Stratified results are diagnostic and less temporally realistic; chronological results are the stronger estimate for future-like performance.
