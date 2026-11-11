| Horizon | Model | PR-AUC | Failures caught | Predictable caught | False alerts / 1k GPU-days | Median lead (h) | Precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24h | rules | 0.280 | 23/31 (74%) | 23/24 (96%) | 51.1 | 15 | 6% |
| 24h | logistic | 0.430 | 21/31 (68%) | 21/24 (88%) | 3.9 | 8 | 43% |
| 24h | lightgbm | 0.545 | 22/31 (71%) | 22/24 (92%) | 2.0 | 11 | 61% |
| 72h | rules | 0.278 | 23/31 (74%) | 23/24 (96%) | 48.4 | 30 | 11% |
| 72h | logistic | 0.439 | 22/31 (71%) | 22/24 (92%) | 1.8 | 31 | 75% |
| 72h | lightgbm | 0.493 | 22/31 (71%) | 21/24 (88%) | 2.8 | 43 | 70% |
