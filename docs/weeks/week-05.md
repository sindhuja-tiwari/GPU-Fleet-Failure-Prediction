# Week 5: straggler detection

Exit check from the plan: precision and recall above 0.9 on injected stragglers. Met on all
three fleets.

| Fleet | Precision | Recall | F1 | Cause right | Injected stragglers found | Median delay |
| --- | --- | --- | --- | --- | --- | --- |
| seed 1 | 0.946 | 0.998 | 0.971 | 97.7% | 301 / 305 | 10 min |
| seed 2 | 0.946 | 0.997 | 0.971 | 95.2% | 311 / 312 | 10 min |
| seed 3 | 0.944 | 0.997 | 0.969 | 97.2% | 308 / 309 | 10 min |

Precision and recall are per busy scrape (a GPU running a job at a 5-minute scrape).
"Injected stragglers found" counts the simulator's straggler intervals that overlap busy
time and got at least one flag. Each fleet is 512 GPUs over 60 days, the same fleets as
week 4.

```bash
gpufleet stragglers data/bench-seed1      # about 50 s; prints the scores and writes intervals
```

## How it works (`src/gpufleet/stragglers.py`)

In data-parallel training every step waits for the slowest GPU, so the signal is each
GPU's step time divided by the median of its node peers at the same scrape. That cancels
out the job's own speed and leaves only "slower than its neighbours".

1. **Peer z-score.** `z = log(step / node median)`, scaled by a robust spread (the
   median absolute deviation over healthy-looking GPUs), so one bad GPU can't inflate it.
2. **Smoothed slowdown.** A 30-minute rolling median of the step ratio. Above 1.05
   (5% slower) means straggling. This catches the large, steady cases.
3. **CUSUM.** `S = max(0, S + z - k)` builds up during small but steady slowdowns that
   the 5% line misses, and ignores single slow steps. The cap on `S` (16) lets it drop
   back within an hour after a large transient. Without the cap, precision fell to 0.66.
4. **Cause.** From what else the GPU shows while flagged:
   - `thermal`: clocks lower than its peers and running hot or thermally throttled.
   - `clock`: clocks lower but temperature normal (degraded silicon, a power cap).
   - `off_gpu`: clocks normal, so the slowdown comes from the host, network or input
     pipeline. Swapping the GPU won't help, and the dashboard says so.

Recall by straggler kind (mean of 3 fleets): thermal 99.7%, chronic 100%, transient 95.7%.
Transient stragglers are the hardest. Some last only a few scrapes, and the 30-minute
median needs about 3 slow scrapes before it moves.

## The "false" positives are mostly failing GPUs

About 5% of flagged scrapes have no injected straggler behind them. On fleet 1 they form
651 short stretches (median 35 minutes) across 184 GPUs. **454 of them (70%) sit in the
week before that same GPU fails**, mostly memory failures (403) and thermal failures (49).
In the simulator, a degrading GPU slows down too. That is a real slowdown, but the straggler
truth table doesn't list it. The remaining stretches work out to about 6 per 1,000
GPU-days.

So the straggler detector doubles as an early failure signal. The failure model already
picks this up through `step_vs_node_6h` and `clock_vs_node_*`.

## Intervals

`gpufleet stragglers` also writes one row per flagged stretch (start, end, scrapes, median
slowdown, cause). A chronic straggler is split into several stretches, because there is no
step time to compare while the GPU sits idle between jobs. That is why fleet 1 shows 2,051
stretches for 305 injected intervals.

## Tests

`tests/test_stragglers.py` checks the end-to-end scores on the test fleet. It also checks
that CUSUM resets per GPU and respects its cap, that a healthy node raises nothing, and that
a 30%-slow GPU with normal clocks gets flagged as `off_gpu` while its peers stay clean.
