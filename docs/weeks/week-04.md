# Week 4: baselines and evaluation

Exit check from the plan: a results table in the README, and LightGBM beats the rules
baseline on PR-AUC and false alarms. Both are met. Against the project targets, LightGBM
at 24 hours hits three of four. Warning time falls short.

| Target (24h horizon) | Goal | LightGBM, mean of 3 fleets | Met? |
| --- | --- | --- | --- |
| Predictable failures caught | at least 70% | 92% | yes |
| False alerts | under 5 per 1,000 GPU-days | 2.7 | yes |
| PR-AUC vs rules | at least 2x | 0.52 vs 0.24 (2.2x) | yes |
| Median warning time | at least 12 h | 10 h | no |

## How it is measured

```bash
./scripts/benchmark.sh          # 3 fleets x (simulate, features, baselines), about 6 minutes
```

- **Data:** three independent simulated fleets (seeds 1 to 3), each 512 GPUs over 60 days,
  with about 100 failures per fleet. The failure rate is raised so there is enough to learn
  from; false alerts per GPU-day do not depend on it.
- **Split:** train on the first 33 days, validate on the next 10, test on the last 14,
  with a 72-hour embargo between them.
- **Alerts:** the first hour a GPU's score crosses the threshold raises an alert; the GPU
  is then quiet for 24 hours. An alert is true if the GPU fails within the horizon after it.
- **Threshold:** for the models, the lowest threshold that stays within 5 false alerts per
  1,000 GPU-days *on validation*, applied unchanged to test. The rules baseline alerts
  whenever any rule fires.
- **Sudden failures** (16 of 89 test failures across the three fleets) have no warning
  signs by design and are left out of "predictable failures caught".

## Results (test period, mean and range over 3 fleets)

| Horizon | Model | PR-AUC | Predictable failures caught | False alerts / 1k GPU-days | Median lead (h) | Alert precision |
| --- | --- | --- | --- | --- | --- | --- |
| 24h | rules | 0.24 (0.20-0.28) | 92% (83%-96%) | 54.7 (49.3-63.7) | 11 (8-15) | 6% (5%-6%) |
| 24h | logistic | 0.46 (0.39-0.54) | 83% (67%-96%) | 2.5 (0.8-3.9) | 9 (7-11) | 57% (43%-73%) |
| 24h | lightgbm | 0.52 (0.44-0.57) | 92% (88%-96%) | 2.7 (1.1-4.9) | 10 (7-11) | 58% (41%-72%) |
| 72h | rules | 0.24 (0.20-0.28) | 92% (83%-96%) | 52.2 (47.6-60.6) | 24 (17-30) | 10% (9%-11%) |
| 72h | logistic | 0.40 (0.33-0.44) | 86% (71%-96%) | 2.2 (0.7-4.1) | 25 (13-32) | 73% (62%-83%) |
| 72h | lightgbm | 0.44 (0.36-0.49) | 72% (38%-92%) | 2.9 (0.3-5.6) | 45 (43-49) | 72% (55%-90%) |

Per-run tables are in `reports/baselines/seed*/`.

### What the numbers say

- **The rules catch as much as the models but page about 20 times more often.** At 55 false
  alerts per 1,000 GPU-days, a 512-GPU fleet gets about 28 false pages a day. LightGBM
  catches the same 92% of predictable failures with about 1.4 false pages a day.
- **The 24-hour models warn too late to meet the 12-hour target.** Their median lead is 10
  hours. Part of that is the horizon itself (no alert can count as more than 24 hours early)
  and part is the strict threshold, which makes the model wait until it is sure.
- **The 72-hour LightGBM warns about two days ahead, but its recall swings** from 38% to 92%
  between fleets. Its threshold comes from only about 20 validation failures, which is too
  few to set it reliably.

### By failure mode (24h, all three fleets)

| Mode | Test failures | LightGBM caught | LightGBM median lead | Rules caught | Rules median lead |
| --- | --- | --- | --- | --- | --- |
| memory | 25 | 25 | 13 h | 25 | 10 h |
| thermal | 16 | 16 | 10 h | 16 | 19 h |
| nvlink | 19 | 19 | 8 h | 19 | 8 h |
| power | 13 | 7 | 3 h | 7 | 4 h |
| sudden | 16 | 0 | none | 1 | none |

Power failures give 2 to 12 hours of warning, so about half are missed at any horizon.
Thermal failures are caught *later* by LightGBM than by the "over 85 C" rule, because the
model's threshold is stricter. That is the clearest place to win lead time back.

### What LightGBM relies on (share of gain, mean of 3 fleets, 24h)

| Feature | Share |
| --- | --- |
| `sbe_accel` (corrected ECC acceleration) | 35% |
| `sbe_rate_6h` | 25% |
| `nvlink_accel` | 11% |
| `throttle_thermal_frac_72h` | 10% |
| `temp_vs_node_72h` | 5% |
| `clock_vs_node_72h` | 4% |

The acceleration features from week 3 carry the model, which matches the exploration in
week 2: failing GPUs speed up, noisy healthy GPUs stay flat.

## Carry-overs to week 6

- **Lead time:** combine the 72h and 24h models (alert on whichever fires first, within
  the same false-alert budget), and pick thresholds with a cost curve that values early
  warnings, not only the false-alert budget.
- **Threshold stability:** choose thresholds with more validation data (rolling-origin
  validation over several windows) or from calibrated probabilities, so recall stops
  swinging between fleets.
- **Thermal:** a survival or trend model should warn days ahead, as the week 2 exploration
  showed is possible.
