"""Command line entry point.

``simulate`` and ``ingest`` produce telemetry, ``summarize`` checks it, ``plots`` explores it,
``features`` builds the labelled hourly table, and ``baselines`` trains and scores models.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .clean import quality_report
from .config import SimConfig
from .schema import validate_telemetry
from .simulator import simulate


def _simulate(args: argparse.Namespace) -> None:
    cfg = SimConfig.from_json(args.config) if args.config else SimConfig()
    for field, value in (
        ("n_nodes", args.nodes),
        ("days", args.days),
        ("step_minutes", args.step_minutes),
        ("seed", args.seed),
        ("failure_rate_per_gpu_day", args.failure_rate),
    ):
        if value is not None:
            setattr(cfg, field, value)

    t0 = time.perf_counter()
    result = simulate(cfg)
    validate_telemetry(result.telemetry)
    out = result.write(args.out)
    took = time.perf_counter() - t0
    print(
        f"wrote {len(result.telemetry):,} rows for {cfg.n_gpus} GPU slots over {cfg.days:g} days "
        f"to {out} in {took:.1f}s"
    )
    _print_summary(result.telemetry, result.failures, result.stragglers)
    m = result.maintenance
    if len(m):
        print("\nMaintenance events:\n" + m["kind"].value_counts().to_string())
    _print_quality(result.telemetry, f"{cfg.step_minutes}min")


def _print_quality(tel: pd.DataFrame, step: str) -> None:
    q = quality_report(tel, step)
    print(
        f"\nData quality: {q['completeness']:.1%} of expected scrapes present, "
        f"{q['gaps']} gaps ({q['gaps_over_1h']} over an hour), "
        f"{q['counter_resets']} counter resets, max node clock skew {q['max_node_skew_s']}s"
    )


def _print_summary(tel: pd.DataFrame, failures: pd.DataFrame, stragglers: pd.DataFrame) -> None:
    observed = failures[failures["observed"]]
    print(f"\nGPUs seen: {tel['gpu_uuid'].nunique()}, "
          f"window: {tel['timestamp'].min()} to {tel['timestamp'].max()}")
    print(f"Failures in window: {len(observed)} "
          f"(plus {len(failures) - len(observed)} still degrading at the end)")
    if len(observed):
        print(observed["mode"].value_counts().to_string())
    print(f"\nStraggler intervals: {len(stragglers)}")
    if len(stragglers):
        print(stragglers["kind"].value_counts().to_string())


def _summarize(args: argparse.Namespace) -> None:
    d = Path(args.data)
    tel = pd.read_parquet(d / "telemetry.parquet")
    validate_telemetry(tel)
    print(f"{len(tel):,} telemetry rows, schema OK")
    if (d / "failures.parquet").exists():
        _print_summary(tel, pd.read_parquet(d / "failures.parquet"),
                       pd.read_parquet(d / "stragglers.parquet"))
    step = "5min"
    if (d / "config.json").exists():
        step = f"{json.loads((d / 'config.json').read_text())['step_minutes']}min"
    _print_quality(tel, args.step or step)


def _ingest(args: argparse.Namespace) -> None:
    from .ingest import load_exporter_scrapes, load_prometheus_range, to_telemetry

    job_label = args.job_label or None
    if args.prometheus_json:
        long = pd.concat(
            [load_prometheus_range(p, job_label) for p in args.prometheus_json], ignore_index=True
        )
    else:
        files = sorted(Path(args.exporter_dir).glob("*.prom"))
        if not files:
            raise SystemExit(f"no .prom files in {args.exporter_dir}")
        long = load_exporter_scrapes(files, job_label)
    rack_map = json.loads(Path(args.rack_map).read_text()) if args.rack_map else None
    tel = to_telemetry(long, rack_map=rack_map)
    validate_telemetry(tel)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tel.to_parquet(out / "telemetry.parquet", index=False)
    print(f"wrote {len(tel):,} rows for {tel['gpu_uuid'].nunique()} GPUs to {out}")
    _print_quality(tel, args.step)


def _step_of(d: Path, default: str = "5min") -> str:
    if (d / "config.json").exists():
        return f"{json.loads((d / 'config.json').read_text())['step_minutes']}min"
    return default


def _features(args: argparse.Namespace) -> None:
    from .features import build_features, feature_columns
    from .labels import add_labels

    d = Path(args.data)
    t0 = time.perf_counter()
    tel = pd.read_parquet(d / "telemetry.parquet")
    step = args.step or _step_of(d)
    feats = build_features(tel, step)
    if (d / "failures.parquet").exists():
        end = tel["timestamp"].max().round(step)
        feats = add_labels(feats, pd.read_parquet(d / "failures.parquet"), end)
    out = Path(args.out) if args.out else d / "features.parquet"
    feats.to_parquet(out, index=False)
    msg = (f"wrote {len(feats):,} rows x {len(feature_columns(feats))} features to {out} "
           f"in {time.perf_counter() - t0:.1f}s")
    if "label_24h" in feats:
        msg += (f"\npositives: {int(feats['label_24h'].sum()):,} (24h), "
                f"{int(feats['label_72h'].sum()):,} (72h); "
                f"censored rows: {int(feats['label_24h'].isna().sum()):,}")
    print(msg)


def _baselines(args: argparse.Namespace) -> None:
    from .experiment import run_baselines

    feats = pd.read_parquet(args.features)
    res = run_baselines(feats, max_false_per_1k=args.budget, val_days=args.val_days,
                        test_days=args.test_days, seed=args.seed)
    out = res.write(args.out)
    print(res.markdown())
    print(f"wrote {out}")


def _plots(args: argparse.Namespace) -> None:
    from .explore import save_all_figures

    paths = save_all_figures(args.data, args.out)
    for p in paths:
        print(f"wrote {p}")


def _stragglers(args: argparse.Namespace) -> None:
    from .stragglers import StragglerConfig, run

    d = Path(args.data)
    tel = pd.read_parquet(d / "telemetry.parquet")
    truth = pd.read_parquet(d / "stragglers.parquet") if (d / "stragglers.parquet").exists() \
        else None
    t0 = time.perf_counter()
    flags, iv, metrics = run(tel, truth, args.step or _step_of(d), StragglerConfig())
    out = Path(args.out) if args.out else d / "straggler_intervals.csv"
    iv.to_csv(out, index=False)
    print(f"{int(flags['straggler'].sum()):,} straggling scrapes, {len(iv)} intervals "
          f"in {time.perf_counter() - t0:.1f}s; wrote {out}")
    if metrics:
        print(json.dumps(metrics, indent=2, default=str))


def _policies(args: argparse.Namespace) -> None:
    from .experiment import run_policies

    feats = pd.read_parquet(args.features)
    res = run_policies(feats, max_false_per_1k=args.budget, seed=args.seed,
                       with_sequence=not args.no_sequence)
    out = res.write(args.out)
    cols = ["policy", "eval_horizon", "pr_auc", "caught", "failures", "predictable_recall",
            "false_alerts_per_1k_gpu_days", "median_lead_hours", "cost_per_1k_gpu_days"]
    print(res.results[cols].to_string(index=False))
    print(f"wrote {out}")


def _ablations(args: argparse.Namespace) -> None:
    from .experiment import run_ablations

    res = run_ablations(pd.read_parquet(args.features), max_false_per_1k=args.budget,
                        seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    cols = ["dropped", "features_dropped", "pr_auc", "predictable_recall",
            "false_alerts_per_1k_gpu_days", "median_lead_hours"]
    print(res[cols].to_string(index=False))
    print(f"wrote {args.out}")


def _train_bundle(args: argparse.Namespace) -> None:
    from .bundle import train_bundle

    t0 = time.perf_counter()
    feats = pd.read_parquet(args.features)
    b = train_bundle(feats, max_false_per_1k=args.budget, seed=args.seed)
    out = b.save(args.out)
    print(f"trained on {b.meta['trained_rows']:,} rows ({b.meta['failures']} failures) in "
          f"{time.perf_counter() - t0:.0f}s; thresholds p24>={b.t24:.3f} (urgent), "
          f"p72>={b.t72:.3f} (warning); wrote {out}")


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .api import create_app

    rack_map = json.loads(Path(args.rack_map).read_text()) if args.rack_map else None
    app = create_app(args.bundle, step=args.step, rack_map=rack_map)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def _replay(args: argparse.Namespace) -> None:
    import urllib.request

    from .client import replay_to_url

    d = Path(args.data)
    tel = pd.read_parquet(d / "telemetry.parquet")
    if args.days:
        tel = tel[tel["timestamp"] < tel["timestamp"].min() + pd.Timedelta(days=args.days)]
    url = args.url.rstrip("/")
    for _ in range(args.wait):  # the API may still be starting
        try:
            urllib.request.urlopen(url + "/healthz", timeout=5).close()
            break
        except OSError:
            time.sleep(1)
    n = replay_to_url(tel, url, args.step or _step_of(d), args.rate)
    print(f"replay finished: {n} alerts")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gpufleet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="generate a synthetic fleet dataset")
    s.add_argument("--out", default="data/sim", help="output directory")
    s.add_argument("--config", help="JSON file with SimConfig fields")
    s.add_argument("--nodes", type=int, help="number of 8-GPU nodes")
    s.add_argument("--days", type=float, help="length of the window in days")
    s.add_argument("--step-minutes", type=int, help="minutes between scrapes")
    s.add_argument("--seed", type=int, help="random seed")
    s.add_argument("--failure-rate", type=float, help="failure onsets per GPU per day")
    s.set_defaults(func=_simulate)

    m = sub.add_parser("summarize", help="validate and summarise a dataset directory")
    m.add_argument("data", help="directory written by `gpufleet simulate`")
    m.set_defaults(func=_summarize)

    m.add_argument("--step", help="scrape interval, e.g. 5min (default: from config.json)")

    i = sub.add_parser("ingest", help="load real DCGM exporter or Prometheus data")
    src = i.add_mutually_exclusive_group(required=True)
    src.add_argument("--prometheus-json", nargs="+", help="saved query_range responses")
    src.add_argument("--exporter-dir", help="directory of scrape files named by timestamp")
    i.add_argument("--out", default="data/real", help="output directory")
    i.add_argument("--job-label", default="pod", help="label naming the job ('' for none)")
    i.add_argument("--rack-map", help="JSON file mapping node name to rack")
    i.add_argument("--step", default="5min", help="scrape interval, for the quality report")
    i.set_defaults(func=_ingest)

    pl = sub.add_parser("plots", help="save the exploration figures for a simulated dataset")
    pl.add_argument("data", help="directory written by `gpufleet simulate`")
    pl.add_argument("--out", default="docs/figures", help="where to write PNGs")
    pl.set_defaults(func=_plots)

    f = sub.add_parser("features", help="build the labelled hourly feature table")
    f.add_argument("data", help="directory with telemetry.parquet (and failures.parquet)")
    f.add_argument("--out", help="output file (default: <data>/features.parquet)")
    f.add_argument("--step", help="scrape interval (default: from config.json, else 5min)")
    f.set_defaults(func=_features)

    b = sub.add_parser("baselines", help="train and evaluate rules, logistic and LightGBM")
    b.add_argument("features", help="labelled feature table from `gpufleet features`")
    b.add_argument("--out", default="reports/baselines", help="output directory")
    b.add_argument("--budget", type=float, default=5.0,
                   help="false alerts per 1,000 GPU-days allowed when picking thresholds")
    b.add_argument("--val-days", type=float, default=10)
    b.add_argument("--test-days", type=float, default=14)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=_baselines)

    st = sub.add_parser("stragglers", help="detect stragglers (and score them if truth exists)")
    st.add_argument("data", help="directory with telemetry.parquet")
    st.add_argument("--out", help="CSV of straggler intervals (default: in the data dir)")
    st.add_argument("--step", help="scrape interval (default: from config.json, else 5min)")
    st.set_defaults(func=_stragglers)

    po = sub.add_parser("policies", help="week 6: calibration, cost thresholds, two-tier alerts")
    po.add_argument("features", help="labelled feature table")
    po.add_argument("--out", default="reports/week6", help="output directory")
    po.add_argument("--budget", type=float, default=5.0)
    po.add_argument("--seed", type=int, default=0)
    po.add_argument("--no-sequence", action="store_true", help="skip the sequence model")
    po.set_defaults(func=_policies)

    ab = sub.add_parser("ablations", help="retrain without each feature family")
    ab.add_argument("features", help="labelled feature table")
    ab.add_argument("--out", default="reports/ablations.csv")
    ab.add_argument("--budget", type=float, default=5.0)
    ab.add_argument("--seed", type=int, default=0)
    ab.set_defaults(func=_ablations)

    tb = sub.add_parser("train-bundle", help="train the deployable model bundle")
    tb.add_argument("features", help="labelled feature table")
    tb.add_argument("--out", default="models/bundle")
    tb.add_argument("--budget", type=float, default=5.0)
    tb.add_argument("--seed", type=int, default=0)
    tb.set_defaults(func=_train_bundle)

    sv = sub.add_parser("serve", help="run the scoring API and dashboard")
    sv.add_argument("--bundle", default="models/bundle")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--step", default="5min", help="scrape interval")
    sv.add_argument("--rack-map", help="JSON file mapping node name to rack")
    sv.set_defaults(func=_serve)

    rp = sub.add_parser("replay", help="send a dataset to a running API, scrape by scrape")
    rp.add_argument("data", help="directory with telemetry.parquet")
    rp.add_argument("--url", default="http://127.0.0.1:8000")
    rp.add_argument("--rate", type=float, default=0,
                    help="scrape rounds per second (0: as fast as the API takes them)")
    rp.add_argument("--days", type=float, help="only the first N days")
    rp.add_argument("--step", help="scrape interval (default: from config.json, else 5min)")
    rp.add_argument("--wait", type=int, default=60, help="seconds to wait for the API")
    rp.set_defaults(func=_replay)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
