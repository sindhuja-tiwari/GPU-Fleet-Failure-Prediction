"""Command line entry point: ``gpufleet simulate``, ``summarize``, ``ingest`` and ``plots``."""

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


def _plots(args: argparse.Namespace) -> None:
    from .explore import save_all_figures

    paths = save_all_figures(args.data, args.out)
    for p in paths:
        print(f"wrote {p}")


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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
