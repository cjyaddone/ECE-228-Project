# TestFlyNofly Analysis

## Data

- Source rows/birds: 34421 rows, 92 birds
- Filtered rows/birds: 10307 rows, 46 birds
- Filters: birds with at least 100 original daily records, latitude 30-50, dates Sep 1-Dec 31
- Fly threshold: 30 km next-day step length

## Results

| k | Train | Test | Test fly rate | Accuracy | Precision | Recall | F1 | FP | FPR | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 6972 | 1744 | 0.0596 | 0.8991 | 0.3571 | 0.8654 | 0.5056 | 162 | 0.0988 | 0.9293 | 0.6636 |
| 20 | 6084 | 1522 | 0.0263 | 0.9093 | 0.1818 | 0.7000 | 0.2887 | 126 | 0.0850 | 0.8989 | 0.2798 |
| 30 | 5288 | 1322 | 0.0121 | 0.8366 | 0.0495 | 0.6875 | 0.0924 | 211 | 0.1616 | 0.7879 | 0.2296 |
| 50 | 3841 | 961 | 0.0083 | 0.9917 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.6364 | 0.0324 |

## Notes

- Best F1 was k=10 with F1=0.5056.
- Best recall was k=10 with recall=0.8654.
- Validation is used as the requested test split; no separate held-out test split was created.
