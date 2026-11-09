# Week 1: foundations and simulator

## What shipped

- `gpufleet` Python package with a CLI (`gpufleet simulate`, `gpufleet summarize`).
- Telemetry schema based on DCGM fields, with a validator ([telemetry-schema.md](../telemetry-schema.md)).
- Fleet simulator:
  - GPU lifecycles: degrade, fail with a fatal Xid, sit offline for repair, get replaced.
  - Five failure modes with different warning signs and warning windows:

    | Mode | Warning signs | Warning window | Fatal Xid |
    | --- | --- | --- | --- |
    | memory | corrected ECC errors climb fast, pages retired (Xid 63), then a double-bit error | 2 to 7 days | 48 |
    | thermal | temperature drifts up 10 to 25 C, thermal throttling, lower clocks, slower steps | 3 to 10 days | 79 |
    | nvlink | NVLink CRC errors in growing bursts | 12 to 72 hours | 74 |
    | power | noisy power draw, PCIe replays | 2 to 12 hours | 79 |
    | sudden | none | none | 119 |

  - Stragglers: chronic (lower clocks for the GPU's whole life), transient (host or network
    hiccups that slow steps while clocks look normal), and thermal (from the failing cooler).
  - Benign look-alikes so no single signal is a giveaway: 3% of healthy GPUs log a steady
    trickle of corrected ECC errors, 3% have noisy NVLinks, and jobs throw application
    Xids (13, 31, 43).
  - Jobs crash when one of their GPUs fails; rack inlet temperature follows a daily cycle;
    0.1% of scrapes go missing.
- 20 tests and GitHub Actions CI (ruff, pytest, CLI smoke test on Python 3.10 and 3.12).

## Default run

`gpufleet simulate` (32 nodes, 256 GPU slots, 30 days, 5-minute scrapes, seed 0) takes
about 6 seconds and writes 2.2 million rows. It produces 16 failures inside the window
(5 power, 4 nvlink, 3 memory, 2 thermal, 2 sudden), 2 more GPUs still degrading at the end,
and 82 straggler intervals.

The default failure rate (0.2% of GPUs start failing per day) is far above a real fleet's
1 to 3% per year, so a month of data has enough failures to learn from. It is one setting
(`--failure-rate`), and evaluation will also report results at realistic rates.

## Choices worth knowing

- **Ground truth stays separate.** Failure onsets and straggler intervals live in their own
  tables. Telemetry only has what a real exporter would give you.
- **The fatal row is kept.** The last row of a failed GPU carries the fatal Xid. Week 3
  labels must predict from rows strictly before that, or models will just learn "Xid 79
  means failure".
- **Rates, not totals.** Healthy noisy GPUs can reach ECC and NVLink totals similar to
  failing ones over a month. What separates them is how fast the counters are rising
  right now, which is what week 3 features will measure.

## Next week

Exploration notebook with plots of each failure mode against healthy GPUs, more collection
noise (longer gaps, clock skew, counter resets on driver reload), maintenance events, and a
loader that maps real DCGM exporter or Prometheus data onto the same schema.
