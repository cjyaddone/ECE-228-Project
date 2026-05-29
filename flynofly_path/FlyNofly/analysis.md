# TestFlyNofly Analysis

## Data

- Source rows/birds: 34421 rows, 92 birds
- Filtered rows/birds: 10307 rows, 46 birds
- Filters: birds with at least 100 original daily records, latitude 30-50, dates Sep 1-Dec 31
- Fly threshold: 30 km next-day step length

## Results

| k | Train | Test | Test fly rate | Accuracy | Precision | Recall | F1 | FP | FPR | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 6972 | 1744 | 0.0596 | 0.8784 | 0.3151 | 0.8846 | 0.4646 | 200 | 0.1220 | 0.9430 | 0.6638 |
| 20 | 6084 | 1522 | 0.0263 | 0.8922 | 0.1737 | 0.8250 | 0.2870 | 157 | 0.1059 | 0.9196 | 0.4336 |
| 30 | 5288 | 1322 | 0.0121 | 0.8699 | 0.0618 | 0.6875 | 0.1134 | 167 | 0.1279 | 0.7712 | 0.0823 |
| 50 | 3841 | 961 | 0.0083 | 0.8512 | 0.0280 | 0.5000 | 0.0530 | 139 | 0.1459 | 0.7030 | 0.1864 |

## Notes

- Best F1 was k=10 with F1=0.4646.
- Best recall was k=10 with recall=0.8846.
- Validation is used as the requested test split; no separate held-out test split was created.
