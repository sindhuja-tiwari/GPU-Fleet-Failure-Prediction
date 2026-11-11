# Model card: GPU failure risk (two-tier LightGBM)

## What it does

Every hour, for every GPU, the model gives two calibrated probabilities:

- `p24`: chance the GPU fails within the next 24 hours.
- `p72`: chance it fails within the next 72 hours.

Two thresholds turn these into alerts:

- **urgent**, when `p24` crosses its threshold: drain the node at the next checkpoint and
  swap the GPU.
- **warning**, when `p72` crosses its threshold: schedule a health check (for example
  `dcgmi diag -r 3`) at the next maintenance window.

Each GPU alerts at most once a day. Every alert lists the three features that pushed its
score up most (SHAP values), in plain words, such as "NVLink CRC errors accelerating".

A separate rule-based detector (not a trained model) flags **stragglers**: GPUs whose
training steps are slower than their node peers. It gives a likely cause: thermal, clock,
or not the GPU.

## Intended use

- **For:** operators of GPU training clusters who collect DCGM telemetry (or similar) every
  few minutes and want to move work off failing GPUs before they crash a job.
- **Decision it supports:** drain and inspect or swap a GPU. A person makes the call. The
  reasons in each alert are there so they can check the evidence first.
- **Not for:** automatic hardware replacement without review, warranty claims, judging
  vendors or hardware generations, or failures with no warning signs (see limits).

## Model

| Part | Detail |
| --- | --- |
| Input | 54 hourly features per GPU from 6h, 24h and 72h windows: error rates (corrected and uncorrected ECC, retired pages, NVLink CRC, PCIe replays, Xid 63), their acceleration, temperature, clock, power and step time relative to node peers, throttling, trends, totals since install, data coverage. No feature looks past the scoring hour. |
| Models | Two LightGBM classifiers (24h and 72h labels), 31 leaves, learning rate 0.05, early stopping. |
| Calibration | Isotonic regression fitted on rolling out-of-fold scores: 4 folds of 5 days, each scored by a model trained only on earlier data with a 72-hour gap. |
| Thresholds | Grid search over (t24, t72) for the lowest expected cost within 5 false alerts per 1,000 GPU-days. Costs, in units of one false alert: missed failure 20, caught failure 2, plus up to 8 more when there is less than 12 hours of warning. Near-ties go to the pair with the most warning time. |
| Explanations | LightGBM's built-in SHAP values (`pred_contrib`) from the 72h model, top 3 positive contributions. |
| Code | `src/gpufleet/bundle.py` (training, scoring, explaining), `src/gpufleet/policy.py` (folds, calibration, cost) |
| Shipped bundle | `models/demo-bundle/`, trained on simulated fleet 1 (512 GPUs, 60 days, 101 failures). |

## Data

**Simulated.** There is no public dataset of per-GPU health telemetry with labelled
failures. The simulator (`src/gpufleet/simulator.py`) generates DCGM-style scrapes every 5
minutes. It includes:

- five failure modes: memory, thermal, NVLink, power, and sudden with no warning;
- three kinds of stragglers;
- benign look-alikes, such as healthy GPUs with noisy ECC counters;
- messy collection: maintenance that resets counters, exporter outages, lost scrapes and
  clock skew.

The failure rate is raised (about 100 failures per 512 GPUs in 60 days) so there is enough
to learn from. False alerts per GPU-day don't depend on the failure rate.

Real DCGM exporter or Prometheus data loads into the same schema (`gpufleet ingest`,
[real-data.md](real-data.md)). **The model has not been trained or tested on real data.**

## Evaluation

Three independent simulated fleets (seeds 1 to 3). Each is trained on its first 43 days
and tested on its last 14, with a 72-hour gap between them. An alert is true if the GPU
fails within 72 hours after it, and each GPU alerts at most once per 24 hours.

| Policy | Predictable failures caught | False alerts / 1k GPU-days | Median warning | Alert precision |
| --- | --- | --- | --- | --- |
| **Two-tier (shipped)** | **93% (68 / 73)** | **1.3 (0.6-1.8)** | **27 h (22-35)** | **83%** |
| LightGBM 24h only | 93% | 2.1 | 10 h | 60% |
| Rules baseline (week 4, 24h) | 92% | 54.7 | 11 h | 6% |

16 of the 89 test failures are sudden (no warning signs by design). None are caught, and
they are excluded from "predictable".

Calibration on the test period (24h model): Brier score 0.0026, expected calibration error
0.0007 to 0.0021, against a base rate of 0.43% of GPU-hours. See
[figure](figures/week6-calibration.png).

**What it relies on** (ablations, [week 8](weeks/week-08.md)): each failure mode leans on one feature family.
Without NVLink features NVLink recall drops to 0%, without PCIe power recall drops to 0%,
and without temperature half the thermal failures are missed. The top reason given with an
alert names the failing part in 77% of true alerts, and one of the top three does in
99.7% (fleets the model never saw).

## Limits and risks

- **Simulated data only.** The failure signatures were written by hand from public
  reports (ECC error storms before memory failures, NVLink CRC bursts, thermal creep).
  Real failures will be messier and more varied. Expect lower recall and different
  feature importance on real fleets. Retrain and re-pick the thresholds on real labels
  before relying on it.
- **Sudden failures can't be predicted.** About 15% of simulated failures have no warning
  signs by design, and real fleets have them too. They are left out of "predictable recall"
  and listed separately.
- **Power failures give little warning** (2 to 12 hours), so about half are caught late or
  missed.
- **Thresholds rest on few failures.** About 40 out-of-fold failures per fleet set them.
  The rolling folds roughly double that compared with one validation window, but the
  operating point can still shift by a few points between fleets.
- **Calibration is a step function.** Isotonic regression gives a handful of distinct
  probability levels. Use the probabilities to rank and to pick thresholds, not as exact
  odds.
- **A new GPU or a restarted scorer has no history.** The scorer holds alerts until it has
  24 hours of data for a GPU, and since-install totals are only as good as the history it
  was given.
- **Feedback loops.** Once alerts lead to GPUs being swapped early, fewer failures are
  observed. Retraining on that data makes the model look less necessary than it is. Keep
  a record of what was swapped because of an alert, and why.

## Straggler detector

Precision 0.94 to 0.95 and recall 1.00 per busy scrape. Cause right 95 to 98% of the time,
on the same three fleets ([week 5](weeks/week-05.md)). About 70% of its "false" flags are on
GPUs that fail within the following week.

## Reproduce

```bash
./scripts/benchmark.sh                                  # weeks 3-4: fleets, features, baselines
gpufleet policies data/bench-seed1/features.parquet     # week 6 table for one fleet
gpufleet ablations data/bench-seed1/features.parquet    # week 8 ablations
gpufleet train-bundle data/bench-seed1/features.parquet --out models/demo-bundle
```
