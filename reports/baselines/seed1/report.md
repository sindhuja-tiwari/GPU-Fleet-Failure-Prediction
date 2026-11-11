| Horizon | Model | PR-AUC | Failures caught | Predictable caught | False alerts / 1k GPU-days | Median lead (h) | Precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24h | rules | 0.228 | 25/29 (86%) | 24/25 (96%) | 63.7 | 9 | 5% |
| 24h | logistic | 0.544 | 24/29 (83%) | 24/25 (96%) | 2.7 | 11 | 56% |
| 24h | lightgbm | 0.566 | 24/29 (83%) | 24/25 (96%) | 4.9 | 11 | 41% |
| 72h | rules | 0.243 | 25/29 (86%) | 24/25 (96%) | 60.6 | 26 | 10% |
| 72h | logistic | 0.444 | 24/29 (83%) | 24/25 (96%) | 4.1 | 32 | 62% |
| 72h | lightgbm | 0.467 | 23/29 (79%) | 23/25 (92%) | 5.6 | 44 | 55% |
