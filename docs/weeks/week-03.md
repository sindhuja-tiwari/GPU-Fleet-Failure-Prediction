# Week 3: labels and features

Exit check from the plan: the feature table builds from raw telemetry with one command,
and a leakage test passes. Both are met.

```bash
gpufleet features data/sim        # writes data/sim/features.parquet
```

On the benchmark fleet (512 GPUs, 60 days, 8.7 million scrapes) this takes about 25
seconds and produces 733,000 rows (one per GPU per hour) with 55 features.

## Prediction setup

- **When:** one prediction per GPU per hour, at the top of the hour.
- **From what:** only scrapes before that hour. Bucket `time = H` holds scrapes in
  `[H - 1h, H)`, and every window looks backwards from there.
- **Labels:** `label_24h` is 1 if the GPU fails within the next 24 hours, else 0.
  `label_72h` is the same for 72 hours.
- **Rows dropped:** anything after a GPU's failure.
- **Rows left unlabelled (censored):** healthy GPUs whose horizon runs past the end of the
  data. We do not know whether they would have failed, so they are neither positive nor
  negative.
- **Sudden failures** are labelled like any other. Evaluation reports them separately.

## Features

All rates come from reset-safe counter increments, divided by the time those increments
actually cover, so missing scrapes neither hide errors nor double-count them.

| Family | Columns | Why |
| --- | --- | --- |
| Error rates per hour over 6h, 24h, 72h | `sbe_rate_*`, `dbe_rate_*`, `retired_rate_*`, `nvlink_rate_*`, `pcie_rate_*` | ECC, NVLink and PCIe error rates separate failing GPUs; raw totals do not |
| Acceleration | `sbe_accel`, `nvlink_accel`, `pcie_accel` (log 6h rate minus log 72h rate) | Failing GPUs speed up; noisy healthy ones stay flat |
| Errors since install | `sbe_since_install`, `retired_since_install`, `nvlink_since_install`, `pcie_since_install` | Long-run history, unaffected by counter resets |
| Peer comparisons over 6h, 24h, 72h | `temp_vs_node_*`, `clock_vs_node_*`, `step_vs_node_*`, `power_vs_node_std_*` | Removes workload and rack effects, leaving the GPU's own drift |
| Temperature trend | `temp_trend_24h`, `temp_trend_72h` (degrees per hour vs peers) | Thermal failures drift for days |
| Throttling | `throttle_thermal_frac_*`, `throttle_power_frac_*`, `temp_max_1h` | Direct symptoms of cooling or power trouble |
| Xids | `xid63_*` (row remaps), `xid_benign_24h` | A burst of row remaps is a memory warning; benign Xids help the model ignore noise |
| Context | `busy_frac_*`, `coverage_*`, `resets_72h` | Tells the model how busy the GPU was and how much data a row rests on |

Fatal Xids are never features: they arrive in the failure row itself, which no feature
window can reach.

## Split

Time-based, never random: the last 14 days are test, the 10 days before that are
validation, and the rest is training. A 72-hour embargo before each boundary drops rows
whose label window would reach into the next period.

## Leakage checks

- `test_no_lookahead` deletes all telemetry from time *t* onwards, rebuilds the features,
  and checks every row up to *t* is bit-for-bit identical to the full build. It does this
  at three different cut points on a fleet with failures, maintenance and outages.
- `test_labels` checks positives sit strictly within the horizon before each failure, that
  no row comes after a failure, and that censored rows stay unlabelled.
- `test_rates_ignore_missing_scrapes` drops every other scrape and checks error rates hold.
