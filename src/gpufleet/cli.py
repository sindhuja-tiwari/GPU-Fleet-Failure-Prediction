"""Command line entry point: ``gpufleet simulate`` and ``gpufleet summarize``."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

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
    _print_summary(tel, pd.read_parquet(d / "failures.parquet"),
                   pd.read_parquet(d / "stragglers.parquet"))


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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
