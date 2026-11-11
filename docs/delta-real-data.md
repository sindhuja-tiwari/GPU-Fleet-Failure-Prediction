# Real-data check: do earlier Xid errors predict critical ones on NCSA Delta?

The models in this repo are trained on simulated telemetry, because there is no public
per-GPU health dataset with labelled failures. This note checks the one assumption the
simulator leans on hardest, that **critical GPU errors are preceded by milder ones**,
against the only large public log of real datacenter GPU errors we found.

**Short answer: not in a way this data can show.** On the public release, earlier Xid
errors do not predict a GPU dropping out (fell off the bus, uncorrectable ECC). They do
predict GSP firmware errors, but mostly because both rise and fall with the same slow
changes in the fleet, not as a short-term warning. The released data is too coarse to
test the per-GPU, hours-ahead warning the project is built around, so this neither
confirms nor rules that out. The simulator stays as the training data for the main model.

Reproduce with `gpufleet delta` (downloads about 9 MB, about 7 minutes on 4 cores). Full
tables: [reports/delta/report.md](../reports/delta/report.md).

## Data

Cui et al., *Story of Two GPUs: Characterizing the Resilience of Hopper H100 and Ampere
A100 GPUs*, SC'25 ([arXiv:2503.11901](https://arxiv.org/abs/2503.11901)). Artifact on
[Zenodo, record 15851780](https://zenodo.org/records/15851780), CC BY 4.0.

* **A100 partition**: 106 nodes, 2022-10-03 to 2025-03-15 (895 days).
* **H100 partition**: 2024-10-21 to 2025-03-15 (146 days).

What the public release actually contains matters more than anything else here:

| File | What is in it | What is not |
| --- | --- | --- |
| `*_allfileregexbookkeeping_*.pkl` | For each day, every Xid event on the partition (5 s de-duplicated) | Node, GPU, time of day |
| `*_cond_prob_forward_*.pkl` | For each Xid event, the closest later Xid on the same GPU (or same node) within 5 s, and the delay | Anything beyond 5 s, absolute times, IDs |

The raw logs (300 GB+) were held back for privacy. The authors said they were working
with NCSA on a fuller release; as of 2026-11-11 the Zenodo record has no newer version.
So the finest grain available is **one partition per day**, or **5 seconds on one GPU**.

The pickles are loaded with a restricted unpickler that only allows plain containers, so
a tampered download cannot run code, and each file is checked against its SHA-256.

## What counts as critical

* **Hard failure**: Xid 48 (double-bit ECC), 64 (row-remap failure), 79 (fell off the
  bus), 94 / 95 (contained / uncontained ECC error). The GPU is out until reset or
  replaced. 23 A100 days and 8 H100 days have one.
* **GSP failure**: Xid 119 / 120 (GSP firmware timeout / error), 122 / 123 (SPI PMU RPC
  failure). Usually needs a GPU reset. 198 A100 days have one, too few on H100 to test.

Everything else (13, 31, 43, 45, 63, 74, 109, and the generic NVRM line on H100) is a
possible precursor.

## Test 1: partition-day forecasting

For each day *t*, features are the counts of each precursor Xid over days *t-7 .. t-1*
(log scale), and the label is whether a critical Xid appears anywhere on the partition on
day *t* (or within 3 days). Logistic regression, scored with 8 contiguous time blocks and
a 10-day gap. Significance comes from 300 circular shifts of the label series, which keep
each series' own autocorrelation.

To separate a short-term warning from "both happen in the same era", there is a
**bursts** variant: each precursor's last 7 days relative to its own previous 90 days.
A real precursor should survive that; a shared slow trend should not.

| Target | Model | ROC-AUC | p |
| --- | --- | --- | --- |
| A100 hard, next day | history only (critical Xids, last 30 d) | 0.37 | 0.79 |
| A100 hard, next day | precursor Xids, last 7 d | 0.55 | 0.30 |
| A100 hard, next day | precursor bursts + history | 0.54 | 0.24 |
| A100 GSP, next day | history only | 0.67 | 0.02 |
| A100 GSP, next day | precursor Xids, last 7 d | **0.74** | 0.003 |
| A100 GSP, next day | precursor bursts + history | 0.68 | 0.003 |
| H100 hard, next day | precursor Xids, last 7 d | 0.51 | 0.35 |

(0.5 is chance. 3-day horizons give the same picture; see the full report.)

What this says:

* **Hard failures are not predictable from earlier Xids at this grain.** Out-of-sample
  ROC-AUC is 0.54 to 0.56 and not distinguishable from shuffled labels. Looking at single
  Xids one at a time, a few do score 0.60 to 0.69 with a small false-discovery rate (MMU
  faults, Xid 31; GSP errors, 119 / 120; context-switch timeouts, 109), but they do not
  hold up when a model is trained on earlier blocks and tested on later ones. With 23
  positive days, a handful of coincidences is enough to make a single-Xid AUC look good.
  Xid 31 is the clearest warning sign: its 7-day average is the same before critical days
  and other days (65 vs 64); it is just present on more of them, which fits "the fleet
  was busy" better than "this GPU is about to fail".
* **GSP failures are predictable, but mostly from the era.** Precursors lift ROC-AUC from
  0.67 (history alone) to 0.74. Most of that comes from NVLink errors (Xid 74) and
  preemptive cleanup (Xid 45), which were near zero in early 2023 and ran at hundreds per
  quarter through 2024, when GSP timeouts were also at their highest. Once each precursor is measured
  against its own recent level, the lift almost disappears (0.68 vs 0.67). So the
  signal is "the fleet is in a phase where GSP errors are common" (driver, firmware,
  workload mix), not "a burst of errors this week warns of a GSP failure tomorrow".
* **H100 has too little data.** 8 days with a hard failure in 146; every model is at or
  below chance.

## Test 2: same-GPU cascades within 5 seconds

| Source Xid | Events | A critical Xid on the same GPU within 5 s |
| --- | --- | --- |
| A100 31 (MMU fault) | 8,469 | 0.1% |
| A100 74 (NVLink) | 1,854 | 0% |
| A100 79 (fell off the bus) | 10 | only another 79 |
| A100 120 (GSP error) | 136 | 99% (Xid 119) |
| H100 48 (double-bit ECC) | 17 | 59% (Xid 64, 95) |
| H100 63 / 64 (row remap) | 16 / 8 | 38% / 100% (Xid 94) |

On the same GPU, critical Xids mostly arrive with no milder error just before them. The
cascades that do exist (48 to 64 / 95, 64 to 94, 120 to 119) happen within milliseconds:
they are one incident logged as several codes, which matches the paper's propagation
analysis. No useful warning time there.

One cross-GPU pattern stands out: on A100 nodes, NVLink errors on one GPU are followed
within 0.7 s by a GPU on the same node falling off the bus (18 pairs) or a contained ECC
error (14). With only 10 Xid 79 events in total, several NVLink errors point at the same
failure, and they come almost at the same moment, so these are most likely peers noticing
the link drop as the GPU dies, again not a warning.

## What this means for the project

* The Xid feature is a **weak, coarse** signal on real data. The models here also use
  temperatures, ECC and retired-page counters, PCIe replays and NVLink CRC counts, which
  this release does not contain, and in the simulator those counters carry most of the
  early warning. Whether they do on real hardware is still untested.
* Our simulator assumes memory failures are preceded by rising correctable errors and
  page retirement. Delta does not contradict that (row-remap events, Xid 63, did not show
  up as a reliable precursor, but there are only 34 of them in 2.5 years), and it cannot
  test it either, because per-GPU counters are not in the release.
* A real test needs per-GPU data with timestamps. Options, roughly from easiest:
  1. The full Delta logs if NCSA releases them (the authors said they planned to).
  2. Our own DCGM exporter history from any fleet we can access, through `gpufleet ingest`.
  3. Other public traces with per-node GPU errors (none we found has hardware counters
     plus failure labels).

## Limits of this check

* "Critical" is a choice. Counting GSP errors as failures, or leaving Xid 64 out of the
  hard set, changes numbers but not conclusions.
* Partition-day is a much easier and much less useful question than per-GPU: a day with
  hundreds of GPUs busy has more of every error. A model that predicted partition-days
  well would still not tell an operator which GPU to drain.
* Tables show p-values of 0.000 when no shuffle beat the real labels (p < 0.001 for the
  single-Xid tests, which use 3,000 shuffles).
