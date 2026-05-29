# TestFlyNofly Analysis

## Data

- Source rows/birds: 34421 rows, 92 birds
- Filtered rows/birds: 10307 rows, 46 birds
- Filters: birds with at least 100 original daily records, latitude 30-50, dates Sep 1-Dec 31
- Fly threshold: 30 km next-day step length

## Results

| k | Train | Test | Test fly rate | Accuracy | Precision | Recall | F1 | FP | FPR | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 6972 | 1744 | 0.0596 | 0.8802 | 0.3158 | 0.8654 | 0.4627 | 195 | 0.1189 | 0.9433 | 0.6771 |
| 20 | 6084 | 1522 | 0.0263 | 0.9402 | 0.2661 | 0.7250 | 0.3893 | 80 | 0.0540 | 0.9170 | 0.4053 |
| 30 | 5288 | 1322 | 0.0121 | 0.8071 | 0.0456 | 0.7500 | 0.0860 | 251 | 0.1922 | 0.8018 | 0.0857 |
| 50 | 3841 | 961 | 0.0083 | 0.8876 | 0.0192 | 0.2500 | 0.0357 | 102 | 0.1070 | 0.5691 | 0.0138 |

## Notes

- Best F1 was k=10 with F1=0.4627.
- Best recall was k=10 with recall=0.8654.
- Validation is used as the requested test split; no separate held-out test split was created.
