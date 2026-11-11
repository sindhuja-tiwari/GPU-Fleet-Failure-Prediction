# Week 8: explain and ship

Exit check from the plan: a reader can understand the results and run the demo from the
README. The README now leads with results and a one-command demo. A [model
card](../model-card.md) covers intended use and limits, and a [demo
video](../demo/dashboard.mp4) shows the dashboard during a replay.

## Every alert says why

Each alert carries the three features that pushed its score up most. They are SHAP values
from the 72h model, computed by LightGBM itself (`pred_contrib`), and labelled in plain
words:

```json
{"gpu": "node-001 gpu7", "level": "urgent", "p24": 0.79, "p72": 0.81,
 "reasons": [
   {"label": "NVLink CRC errors accelerating", "value": 0.508, "contribution": 2.1},
   {"label": "NVLink CRC errors since install", "value": 12, "contribution": 0.6},
   {"label": "NVLink CRC errors (last 72h)", "value": 0.203, "contribution": 0.5}]}
```

The reasons show up in the API, the dashboard, and the Prometheus alert description.

**Do the reasons point at the real problem?** `scripts/explain_check.py` scores two fleets
the shipped bundle never saw (fleets 2 and 3) and raises two-tier alerts. For every true
alert, it checks whether the top reason belongs to the failing part: ECC and retired pages
for memory failures, NVLink for NVLink failures, temperature for thermal failures, PCIe and
power for power failures.

| Failure mode | True alerts | Top reason names the failing part | One of the top 3 does |
| --- | --- | --- | --- |
| memory | 166 | 48% | 99% (165) |
| NVLink | 66 | 100% | 100% |
| power | 38 | 100% | 100% |
| thermal | 117 | 97% | 100% |
| **all** | **387** | **77%** | **99.7%** |

NVLink and power failures are explained correctly every time. Memory failures lead with
"lower clocks than its node peers (last 6h)" about half the time. In the simulator, a
failing memory GPU also loses clock speed, so this is a real symptom, but an operator would
rather see the ECC counts first. The ECC reason is still in the top three for 165 of 166
of them.

## Ablations: what each feature family is worth

`gpufleet ablations` retrains LightGBM (24h, week 4 protocol) with one feature family
removed at a time, on all three fleets. Full table:
[reports/ablations/summary.md](../../reports/ablations/summary.md).

| Features removed | PR-AUC | Change | Predictable caught | False alerts / 1k | Median lead |
| --- | --- | --- | --- | --- | --- |
| none (full model) | 0.518 |  | 92% | 2.7 | 10 h |
| ECC and retired pages | 0.403 | -0.115 | 81% | 2.4 | 10 h |
| NVLink | 0.412 | -0.106 | 67% | 2.1 | 12 h |
| PCIe | 0.510 | -0.008 | 82% | 2.1 | 12 h |
| temperature and thermal throttling | 0.466 | -0.052 | 79% | 3.1 | 10 h |
| power | 0.504 | -0.014 | 88% | 3.2 | 10 h |
| clock and step time | 0.512 | -0.006 | 94% | 2.1 | 10 h |
| all peer-relative (vs node) | 0.488 | -0.030 | 85% | 2.6 | 11 h |
| since-install totals | 0.522 | +0.004 | 93% | 2.3 | 10 h |
| acceleration and trends | 0.498 | -0.020 | 92% | 2.9 | 11 h |
| 6h windows | 0.518 | +0.000 | 88% | 2.2 | 11 h |
| 72h windows | 0.511 | -0.007 | 96% | 2.4 | 9 h |

What it shows:

- **Each failure mode leans on one family.** Remove NVLink features and NVLink recall
  drops from 100% to 0%. Remove PCIe and power recall drops to 0%. Remove temperature and
  half the thermal failures are missed. Remove ECC and memory recall falls to 79%; clock
  slowdown still catches the rest.
- **Peer-relative features help with thermal failures.** Without them, thermal recall falls
  from 100% to 82%. Absolute temperature can't tell a hot GPU from a hot room.
- **Some features carry little.** Since-install totals and the 6h windows change PR-AUC by
  less than 0.005 when removed. They can be dropped without loss, which makes the
  scorer's state smaller.
- **Power failures are noisy to measure.** There are only about 5 per fleet, so that
  column moves in steps of 20 points. Read the power numbers as rough.

## Model card

[docs/model-card.md](../model-card.md): what the model does and doesn't do, the data it
was trained on (simulated only), evaluation on three fleets, calibration, and limits.

## Demo video

[docs/demo/dashboard.mp4](../demo/dashboard.mp4), about 90 seconds. The shipped bundle
(trained on fleet 1) scores a fleet it has never seen: 128 GPUs, a different random seed,
streamed in at 20 scrape rounds a second. After the 24-hour warm-up, urgent and warning
alerts arrive with their reasons. Clicking a GPU shows its risk history climbing towards
the thresholds before it fails. The stragglers table shows the likely cause of each
slowdown in plain words.

Recorded with Playwright against `gpufleet serve` and `gpufleet replay`. The steps are in
the README.

## Where the project ended up

| Target | Goal | Result (mean of 3 fleets) | Met? |
| --- | --- | --- | --- |
| Predictable failures caught (horizon 72h) | at least 70% | 93% | yes |
| False alerts | under 5 per 1,000 GPU-days | 1.3 | yes |
| Median warning time | at least 12 h | 27 h | yes |
| PR-AUC vs rules (24h model) | at least 2x | 2.0x | yes |
| Straggler precision / recall | 0.9 / 0.9 | 0.95 / 1.00 | yes |
| Scoring latency, 512 GPUs | under 1 s per scrape | 66 ms median, 0.45 s worst, over HTTP | yes |

## What would come next

- **Real data.** Everything here is simulated. The loader for DCGM exporter and Prometheus
  data is ready. The next step is a few months of a real fleet's telemetry plus its
  RMA/repair log as labels.
- **Survival model.** The plan listed one. Two calibrated horizons (24h and 72h) cover what
  the alerts need. A proper time-to-failure model would add a full risk curve per GPU.
- **A GRU or temporal CNN.** PyTorch wasn't installable in the build environment, so the
  sequence model is an MLP over 48 hours of raw signals (week 6). On these fleets it
  doesn't beat LightGBM on hand-made features. On real data, with failure patterns nobody
  wrote by hand, a learned model may do better.
