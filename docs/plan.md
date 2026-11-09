# 8-week plan

Goal: flag GPUs likely to fail in the next 24 to 72 hours, and spot stragglers in training
jobs, from per-GPU telemetry. Each week ends with a merged PR and a note in `docs/weeks/`.

| Week | Focus | Deliverables | Done when |
| --- | --- | --- | --- |
| 1 | Foundations and simulator | Package and CLI, telemetry schema with validation, fleet simulator (5 failure modes, 3 straggler types, benign noise), tests, CI | `gpufleet simulate` writes a 30-day, 256-GPU fleet in under a minute and every row passes schema checks |
| 2 | Data realism and exploration | Scrape gaps and clock skew, maintenance events, loader for real DCGM / Prometheus exports, exploration notebook | Each failure mode is visibly different from healthy GPUs, and real exports load into the same schema |
| 3 | Labels and features | 24h / 72h horizon labels with censoring, rolling-window and peer-relative features, time-based splits | Feature table builds with one command, and a leakage test passes |
| 4 | Baselines | Rules baseline, logistic regression, LightGBM, evaluation module | Results table in the README; LightGBM beats rules on PR-AUC and false alarms |
| 5 | Straggler detection | Robust peer z-scores, CUSUM change points, cause attribution | Precision and recall above 0.9 on injected stragglers |
| 6 | Sequence models and calibration | GRU or temporal CNN, survival model, calibration, cost-based alert threshold | Calibrated model with a justified alert threshold |
| 7 | Serving | Streaming scorer, FastAPI, alert rules, dashboard, Docker Compose demo | `docker compose up` replays telemetry with live risk scores |
| 8 | Explain and ship | SHAP per alert, ablations, model card, final README, demo video | A reader can understand the results and run the demo from the README |

## Success metrics (held-out final week)

| Metric | Target |
| --- | --- |
| Recall of predictable failures, 24h horizon | at least 70% |
| False alarms | under 5 per 1,000 GPU-days |
| Median warning lead time | at least 12 hours |
| PR-AUC vs rules baseline | at least 2x |
| Straggler precision / recall | at least 0.9 / 0.9 |
| Scoring latency, 512 GPUs | under 1 second per scrape |

Failures with no warning (`sudden`) are reported separately.
