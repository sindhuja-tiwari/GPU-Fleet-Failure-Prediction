Mean of 3 fleets, LightGBM 24h, week 4 protocol. Last columns: share of each failure mode caught.

| Features removed | Count | PR-AUC | Change | Predictable caught | False alerts / 1k | Median lead (h) | memory | nvlink | power | thermal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none (full model) | 0 | 0.518 |  | 92% | 2.7 | 10 | 100% | 100% | 43% | 100% |
| ECC and retired pages | 15 | 0.403 | -0.115 | 81% | 2.4 | 10 | 79% | 100% | 43% | 92% |
| NVLink | 5 | 0.412 | -0.106 | 67% | 2.1 | 12 | 100% | 0% | 49% | 100% |
| PCIe | 5 | 0.510 | -0.008 | 82% | 2.1 | 12 | 100% | 100% | 0% | 100% |
| temperature and thermal throttling | 9 | 0.466 | -0.052 | 79% | 3.1 | 10 | 100% | 100% | 42% | 50% |
| power | 6 | 0.504 | -0.014 | 88% | 3.2 | 10 | 100% | 100% | 27% | 100% |
| clock and step time | 6 | 0.512 | -0.006 | 94% | 2.1 | 10 | 100% | 100% | 66% | 100% |
| all peer-relative (vs node) | 12 | 0.488 | -0.030 | 85% | 2.6 | 11 | 100% | 100% | 43% | 82% |
| since-install totals | 4 | 0.522 | +0.004 | 93% | 2.3 | 10 | 100% | 100% | 49% | 100% |
| acceleration and trends | 5 | 0.498 | -0.020 | 92% | 2.9 | 11 | 100% | 100% | 42% | 100% |
| 6h windows | 14 | 0.518 | +0.000 | 88% | 2.2 | 11 | 100% | 100% | 27% | 100% |
| 72h windows | 16 | 0.511 | -0.007 | 96% | 2.4 | 9 | 100% | 100% | 82% | 100% |
