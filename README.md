# GPU Fleet Failure Prediction

Predict failing GPUs and training-job stragglers from fleet telemetry (temperatures, power,
clocks, ECC and NVLink errors, Xid codes, step times).

The goal is a system that warns 12 to 72 hours before a GPU fails, keeps false alarms
under 5 per 1,000 GPU-days, and points out GPUs that slow down their training jobs, with
the likely cause. See the [8-week plan](docs/plan.md).

## Status

| Week | Focus | Status |
| --- | --- | --- |
| 1 | Package, telemetry schema, fleet simulator | Done ([notes](docs/weeks/week-01.md)) |
| 2 | Data realism, real-data loader, exploration | Done ([notes](docs/weeks/week-02.md)) |
| 3 | Labels and features | Done ([notes](docs/weeks/week-03.md)) |
| 4 | Baseline models and evaluation | Done ([notes](docs/weeks/week-04.md)) |
| 5 | Straggler detection | Next |
| 6 | Sequence models and calibration | |
| 7 | Serving and dashboard | |
| 8 | Explanations, model card, write-up | |

## Results so far

Test period of three simulated 512-GPU fleets, 24-hour horizon, thresholds picked on
validation for at most 5 false alerts per 1,000 GPU-days. Full table and caveats in the
[week 4 notes](docs/weeks/week-04.md).

| Model | PR-AUC | Predictable failures caught | False alerts / 1k GPU-days | Median warning |
| --- | --- | --- | --- | --- |
| Rules (fixed operator thresholds) | 0.24 | 92% | 54.7 | 11 h |
| Logistic regression | 0.46 | 83% | 2.5 | 9 h |
| LightGBM | 0.52 | 92% | 2.7 | 10 h |

LightGBM catches as many failures as the rules with about 20 times fewer false pages.
Warning time is still short of the 12-hour target; that is the main job for week 6.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,models]"

gpufleet simulate --out data/sim          # 256 GPUs, 30 days, ~6 seconds
gpufleet summarize data/sim               # validate, print counts and data quality
gpufleet plots data/sim                   # exploration figures into docs/figures
gpufleet features data/sim                # hourly labelled feature table
gpufleet baselines data/sim/features.parquet --out reports/my-run
pytest -q
```

Options: `--nodes`, `--days`, `--step-minutes`, `--seed`, `--failure-rate`, or
`--config my.json` with any field of `gpufleet.config.SimConfig`.

To explore interactively, install `pip install -e ".[notebook]"` and open
[notebooks/01_explore.ipynb](notebooks/01_explore.ipynb).

## Real fleet data

`gpufleet ingest` loads NVIDIA DCGM exporter output, from raw scrapes or a Prometheus
range query, into the same schema. See [docs/real-data.md](docs/real-data.md).

```bash
gpufleet ingest --prometheus-json dump.json --out data/real
gpufleet summarize data/real --step 5min
```

## Real-data check (NCSA Delta)

Do earlier Xid errors predict critical ones on a real fleet? `gpufleet delta` downloads the
public NCSA Delta A100 / H100 Xid logs (Cui et al., SC'25, Zenodo 15851780) and tests it.
On this release, earlier Xids do **not** predict hard failures such as a GPU falling off the
bus (out-of-sample ROC-AUC 0.55, not significant). They do predict GSP firmware errors
(0.74), but mostly because both track slow changes in the fleet rather than as a short-term
warning. The release has no node, GPU or time of day, so the per-GPU, hours-ahead question
stays open. Details: [docs/delta-real-data.md](docs/delta-real-data.md).

```bash
gpufleet delta --out reports/delta
```

## Simulated data

There is no public dataset of per-GPU health telemetry with labelled failures, so the
project starts with a simulator. It writes:

| File | Contents |
| --- | --- |
| `telemetry.parquet` | One row per GPU per scrape, DCGM-style fields ([schema](docs/telemetry-schema.md)) |
| `failures.parquet` | Ground truth: failure mode, onset time, failure time, repair time |
| `stragglers.parquet` | Ground truth: slow-GPU intervals and their cause |
| `inventory.parquet` | Every physical GPU and when it was installed or removed |
| `maintenance.parquet` | Ground truth: reboots, driver upgrades, drains and GPU swaps per node |
| `config.json` | Settings used, for reproducing the run |

The simulator injects five failure modes (memory, thermal, NVLink, power, and sudden
failures with no warning), three kinds of stragglers, and benign look-alikes such as
healthy GPUs with noisy ECC counters, so that no single metric gives the answer away.
It also reproduces messy collection: maintenance that resets counters, exporter outages,
lost scrapes and node clock skew. Details are in the [week 1](docs/weeks/week-01.md) and
[week 2](docs/weeks/week-02.md) notes.

![Failure signatures](docs/figures/failure-signatures.png)

## Layout

```
src/gpufleet/
  config.py      simulator settings and failure-mode definitions
  schema.py      telemetry schema and validator
  simulator.py   synthetic fleet generator
  clean.py       clock alignment, counter resets, gaps, data-quality report
  ingest.py      DCGM exporter / Prometheus loader
  explore.py     exploration signals and figures
  features.py    hourly feature table (no look-ahead)
  labels.py      24h / 72h labels, censoring, time split with embargo
  models.py      rules baseline, logistic regression, LightGBM
  evaluate.py    alert-level metrics: recall, false alerts, lead time
  experiment.py  train / validate / test protocol
  delta.py       real-data check on the public NCSA Delta Xid logs
  cli.py         command line
notebooks/       exploration notebook
scripts/         benchmark.sh reproduces the results table
reports/         saved benchmark results
tests/           pytest suite
docs/            plan, schema, weekly notes
```
