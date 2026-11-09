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
| 2 | Data realism, real-data loader, exploration | Next |
| 3 | Labels and features | |
| 4 | Baseline models and evaluation | |
| 5 | Straggler detection | |
| 6 | Sequence models and calibration | |
| 7 | Serving and dashboard | |
| 8 | Explanations, model card, write-up | |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

gpufleet simulate --out data/sim          # 256 GPUs, 30 days, ~6 seconds
gpufleet summarize data/sim               # validate and print counts
pytest -q
```

Options: `--nodes`, `--days`, `--step-minutes`, `--seed`, `--failure-rate`, or
`--config my.json` with any field of `gpufleet.config.SimConfig`.

## Data

There is no public dataset of per-GPU health telemetry with labelled failures, so the
project starts with a simulator. It writes:

| File | Contents |
| --- | --- |
| `telemetry.parquet` | One row per GPU per scrape, DCGM-style fields ([schema](docs/telemetry-schema.md)) |
| `failures.parquet` | Ground truth: failure mode, onset time, failure time, repair time |
| `stragglers.parquet` | Ground truth: slow-GPU intervals and their cause |
| `inventory.parquet` | Every physical GPU and when it was installed or removed |
| `config.json` | Settings used, for reproducing the run |

The simulator injects five failure modes (memory, thermal, NVLink, power, and sudden
failures with no warning), three kinds of stragglers, and benign look-alikes such as
healthy GPUs with noisy ECC counters, so that no single metric gives the answer away.
Details are in the [week 1 notes](docs/weeks/week-01.md).

## Layout

```
src/gpufleet/
  config.py      simulator settings and failure-mode definitions
  schema.py      telemetry schema and validator
  simulator.py   synthetic fleet generator
  cli.py         command line
tests/           pytest suite
docs/            plan, schema, weekly notes
```
