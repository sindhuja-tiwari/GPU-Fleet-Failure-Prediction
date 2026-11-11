| Horizon | Model | PR-AUC | Failures caught | Predictable caught | False alerts / 1k GPU-days | Median lead (h) | Precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24h | rules | 0.202 | 20/29 (69%) | 20/24 (83%) | 49.3 | 8 | 5% |
| 24h | logistic | 0.394 | 16/29 (55%) | 16/24 (67%) | 0.8 | 7 | 73% |
| 24h | lightgbm | 0.444 | 21/29 (72%) | 21/24 (88%) | 1.1 | 7 | 72% |
| 72h | rules | 0.196 | 20/29 (69%) | 20/24 (83%) | 47.6 | 17 | 9% |
| 72h | logistic | 0.331 | 17/29 (59%) | 17/24 (71%) | 0.7 | 13 | 83% |
| 72h | lightgbm | 0.365 | 9/29 (31%) | 9/24 (38%) | 0.3 | 49 | 90% |
