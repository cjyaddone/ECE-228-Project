# TestFlyNofly Analysis

## Data

- Source rows/birds: 34421 rows, 92 birds
- Filtered rows/birds: 10307 rows, 46 birds
- Filters: birds with at least 100 original daily records, latitude 30-50, dates Sep 1-Dec 31
- Fly threshold: 30 km next-day step length

## Results

| k | Train | Test | Test fly rate | Accuracy | Precision | Recall | F1 | FP | FPR | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 6972 | 1744 | 0.0596 | 0.9071 | 0.3782 | 0.8654 | 0.5263 | 148 | 0.0902 | 0.9503 | 0.6668 |
| 20 | 6084 | 1522 | 0.0263 | 0.7779 | 0.0884 | 0.8000 | 0.1592 | 330 | 0.2227 | 0.8632 | 0.3364 |
| 30 | 5288 | 1322 | 0.0121 | 0.7897 | 0.0288 | 0.5000 | 0.0544 | 270 | 0.2067 | 0.7540 | 0.0812 |
| 50 | 3841 | 961 | 0.0083 | 0.9553 | 0.0732 | 0.3750 | 0.1224 | 38 | 0.0399 | 0.8387 | 0.1889 |

## Notes

- Best F1 was k=10 with F1=0.5263.
- Best recall was k=10 with recall=0.8654.
- Validation is used as the requested test split; no separate held-out test split was created.
