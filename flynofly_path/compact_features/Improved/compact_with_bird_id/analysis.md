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
| chronological | 10 | 0.0693 | 0.7299 | 0.7167 | 0.6604 | 0.7836 | 54 | 0.0300 | 0.5665 |
| chronological | 20 | 0.0703 | 0.8013 | 0.6738 | 0.6333 | 0.7197 | 55 | 0.0315 | 0.5124 |
| chronological | 30 | 0.0710 | 0.7797 | 0.7079 | 0.6438 | 0.7863 | 57 | 0.0333 | 0.5786 |
| chronological | 50 | 0.0716 | 0.7630 | 0.7092 | 0.6494 | 0.7812 | 54 | 0.0325 | 0.6407 |
| stratified | 10 | 0.0708 | 0.8948 | 0.5725 | 0.6186 | 0.5328 | 45 | 0.0250 | 0.4615 |
| stratified | 20 | 0.0697 | 0.8107 | 0.6224 | 0.5742 | 0.6794 | 66 | 0.0378 | 0.5411 |
| stratified | 30 | 0.0700 | 0.8371 | 0.5869 | 0.5846 | 0.5891 | 54 | 0.0315 | 0.5206 |
| stratified | 50 | 0.0705 | 0.7914 | 0.5940 | 0.5643 | 0.6270 | 61 | 0.0367 | 0.4694 |

## Interpretation

- Target-only Sep-Dec windowing restores the chronological test fly rate to roughly 7% for all k, instead of below 1-3% for longer contexts.
- Threshold tuning is essential because class-weighted training does not imply that 0.5 is the right operating point.
- If baselines beat the Transformer for a split, the task is currently driven more by hand-engineered movement history than by attention over daily tokens.
- Stratified results are diagnostic and less temporally realistic; chronological results are the stronger estimate for future-like performance.
