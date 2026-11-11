# Week 7: serving

Exit check from the plan: `docker compose up` replays telemetry with live risk scores.
`docker compose up` was run end to end: the trainer step, the API, the replayer and
Prometheus with the alert rules all came up, and live scores flowed. CI now builds the
image and replays a short fleet through it on every PR. The latency target is met.

| Check | Result |
| --- | --- |
| Streamed features equal batch features | yes, within float32 rounding (largest gap 3e-5) |
| Scoring latency, 512 GPUs, per scrape round | median 47 ms, 95th percentile 205 ms, worst 0.44 s (target under 1 s) |
| Same latency over HTTP (JSON in, scores out) | median 66 ms, 95th percentile 240 ms, worst 0.45 s |
| Live risk scores, alerts with reasons, stragglers | yes, on the dashboard, in the API and in Prometheus metrics |

## The pieces

```
telemetry ──► POST /v1/telemetry (JSON rows) ─┐
DCGM text ──► POST /v1/scrape  (exporter text)┤
                                              ▼
                     StreamingScorer (src/gpufleet/stream.py)
                     ├─ per-scrape signals (counter increments, peer ratios)
                     ├─ hourly buckets, last 80 h kept per GPU
                     ├─ on each new hour: features ─► model bundle ─► p24, p72, level
                     ├─ alerts (24 h cooldown, 24 h warm-up) with top reasons
                     └─ stragglers from the last 2 h of scrapes
                                              │
      GET /  dashboard · /v1/risk · /v1/alerts · /v1/stragglers · /v1/gpu/{uuid} · /metrics
                                              │
                         Prometheus ─► alert rules (deploy/prometheus/alerts.yml)
```

**Model bundle** (`src/gpufleet/bundle.py`, `gpufleet train-bundle`). Both LightGBM models,
their isotonic calibration, the two alert thresholds, and the feature list, saved as two
LightGBM text files plus `bundle.json`. It is trained the week 6 way on everything
labelled: rolling out-of-fold scores, calibration, then the cost-optimal two-tier
thresholds within the false-alert budget. Training on fleet 1 (733k rows, 101 failures)
takes about two minutes on 4 cores.

**Streaming scorer** (`src/gpufleet/stream.py`). It keeps a small amount of state per GPU:

- the last scrape, to turn the next scrape's counters into increments;
- the scrapes of the hour still open;
- the last 80 hourly buckets, plus running totals of whatever was trimmed off, so
  "since install" features stay right after trimming;
- the last 2 hours of per-scrape signals, for stragglers.

When the first scrape of a new hour arrives, the finished hour is bucketed and every GPU in
it is scored. It runs **the same feature code as training** (`features_from_buckets`), so
the two can't drift apart. A test checks streamed features against the batch features at
the last hour of a 4-day replay, after trimming has kicked in.

Two rules keep the pages useful:

- **Cooldown:** one alert per GPU per 24 hours, as in the evaluation.
- **Warm-up:** no alerts for a GPU until the scorer has seen 24 hours of its data. Before
  that, the 24h and 72h windows are mostly empty, and a freshly started scorer raised
  false alerts in its first hours.

Scores are still computed during warm-up and show on the dashboard.

**API** (`src/gpufleet/api.py`, `gpufleet serve`). FastAPI. Ingest takes either JSON rows in
the telemetry schema or raw DCGM exporter text, so a scrape can be forwarded as is. The
exporter's Xid field holds the *last* Xid seen, so the API keeps the previous value per GPU
and only counts changes. Read endpoints return the latest score per GPU, recent alerts with
reasons, current stragglers, and one GPU's 72-hour score history. One lock serialises
ingest and reads, which is plenty at one scrape round per 5 minutes.

**Alert rules** (`deploy/prometheus/alerts.yml`). `/metrics` exports the probability and
alert tier per GPU, straggler slowdown with its cause, alert counters and ingest health.
The rules page on urgent (2), open a ticket on warning (1), flag stragglers that stay above
10% slower for 15 minutes, and warn if telemetry stops arriving.

**Dashboard** (`GET /`). A single static page that polls `/v1/summary` every 2 seconds and
shows:

- counts of urgent and warning GPUs, stragglers and alerts;
- the highest-risk GPUs;
- the selected GPU's 72-hour risk history against the two thresholds;
- recent alerts with their reasons;
- the stragglers table, with the cause in plain words.

## Running it

With Docker:

```bash
docker compose up --build
# dashboard: http://localhost:8000   Prometheus alerts: http://localhost:9090/alerts
```

`trainer` copies the pretrained bundle shipped in `models/demo-bundle` into place, so the
demo starts in seconds. With `TRAIN=1` it simulates a 512-GPU, 60-day fleet and trains a
new one instead (about 10 minutes on 4 cores). `api` serves it. `replayer` simulates a different 7-day fleet (another seed) and streams it in
at 20 scrape rounds a second, so one simulated day takes about 15 seconds. `prometheus`
scrapes `/metrics` and evaluates the rules.

Without Docker:

```bash
pip install -e ".[serve]"
gpufleet serve --bundle models/demo-bundle &                  # http://127.0.0.1:8000
gpufleet simulate --out data/live --nodes 16 --days 7 --seed 42 --failure-rate 0.01
gpufleet replay data/live --rate 20
```

## Latency

Measured in-process on a 512-GPU fleet (64 nodes), 4 days, one scrape round per call (512
rows):

- Most rounds only add signals: about 47 ms.
- The first round of each hour also buckets the hour and scores all 512 GPUs: 0.2 to 0.44 s.

Over HTTP (`gpufleet serve` on the same machine, JSON rows posted by the replay client), the
median is 66 ms and the worst is 0.45 s. Building and parsing JSON for 512 rows adds about
20 ms per round.

## Not done / caveats

- **Docker test environment.** Where this was built, the build container couldn't reach
  the Debian mirror, so the local test copied `libgomp` in by hand instead of running
  the `apt-get` line. The CI job builds the Dockerfile exactly as committed.
- **One process, state in memory.** A restart loses the open windows. It needs 72 hours
  of telemetry (replayed from Prometheus, for example) to warm back up fully. Sharding by
  node would be the next step for fleets much larger than a few thousand GPUs.
- **No auth** on the API. Put it behind the cluster's usual ingress.
