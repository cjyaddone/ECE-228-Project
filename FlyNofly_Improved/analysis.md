# Improved TestFlyNofly Analysis

## Data

- Source rows/birds: 34421 rows, 92 birds
- All-month context rows/birds: 29977 rows, 46 birds
- Sep-Dec target candidates: 10307 rows
- Sep-Dec candidate fly rate: 0.0739
- Context is allowed before September; only target days are restricted to Sep-Dec.

## Comparison To Original

- Original best F1 was k=10 with F1=0.5263, precision=0.3782, recall=0.8654.
- Improved best overall: triline_multitask chronological k=10 with F1=0.6909, precision=0.6738, recall=0.7090, FP=46.

## Best By Split

| Split | Model | k | F1 | Precision | Recall | FP | FPR | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chronological | triline_multitask | 10 | 0.6909 | 0.6738 | 0.7090 | 46 | 0.0255 | 0.9410 | 0.7018 |
| stratified | random_forest_balanced | 20 | 0.6189 | 0.6119 | 0.6260 | 52 | 0.0297 | 0.9157 | 0.6137 |

## Triline Multi-Task Results

| Split | k | Test fly rate | Threshold | F1 | Precision | Recall | FP | FPR | Fixed-0.5 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chronological | 10 | 0.0693 | 0.8600 | 0.6909 | 0.6738 | 0.7090 | 46 | 0.0255 | 0.5659 |
| chronological | 20 | 0.0703 | 0.8131 | 0.6645 | 0.5917 | 0.7576 | 69 | 0.0395 | 0.4884 |
| chronological | 30 | 0.0710 | 0.7872 | 0.6712 | 0.6037 | 0.7557 | 65 | 0.0379 | 0.5033 |
| chronological | 50 | 0.0716 | 0.7144 | 0.6709 | 0.5676 | 0.8203 | 80 | 0.0482 | 0.6145 |
| stratified | 10 | 0.0708 | 0.8590 | 0.5606 | 0.5827 | 0.5401 | 53 | 0.0295 | 0.4976 |
| stratified | 20 | 0.0697 | 0.8630 | 0.5934 | 0.5704 | 0.6183 | 61 | 0.0349 | 0.4842 |
| stratified | 30 | 0.0700 | 0.7580 | 0.5827 | 0.5436 | 0.6279 | 68 | 0.0397 | 0.5127 |
| stratified | 50 | 0.0705 | 0.8309 | 0.6050 | 0.6429 | 0.5714 | 40 | 0.0241 | 0.4323 |

## Interpretation

- Target-only Sep-Dec windowing restores the chronological test fly rate to roughly 7% for all k, instead of below 1-3% for longer contexts.
- Threshold tuning is essential because class-weighted training does not imply that 0.5 is the right operating point.
- If baselines beat the Transformer for a split, the task is currently driven more by hand-engineered movement history than by attention over daily tokens.
- Stratified results are diagnostic and less temporally realistic; chronological results are the stronger estimate for future-like performance.
