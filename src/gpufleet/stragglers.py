"""Straggler detection: which GPUs slow their training jobs down, and why.

In data-parallel training every step waits for the slowest GPU, so one slow GPU costs the
whole job. Detection works on each GPU's step time relative to its peers on the same node
at the same moment, which cancels out the job's own speed:

1. **Peer z-score.** ``z = log(step / node median) / scale``, where ``scale`` is a robust
   estimate (MAD) of how much healthy GPUs scatter around their peers.
2. **CUSUM.** A one-sided cumulative sum ``S = max(0, S + z - k)`` per GPU builds up
   during a sustained slowdown and ignores single noisy steps. A GPU is a straggler while
   its smoothed slowdown is above ``min_slowdown``, or while ``S`` is above ``h`` and it is
   still at least ``cusum_floor`` slow (this catches smaller but steady slowdowns).
3. **Cause.** Each flagged interval is labelled from what else the GPU shows:
   ``thermal`` (low clocks and running hot or throttled), ``clock`` (low clocks at normal
   temperature, e.g. degraded silicon or a power cap), or ``off_gpu`` (clocks normal, so the
   slowdown comes from the host, network or data loading).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import scrape_signals

CAUSES = ("thermal", "clock", "off_gpu")


@dataclass
class StragglerConfig:
    k: float = 1.0  # CUSUM slack, in robust z units
    h: float = 8.0  # CUSUM alarm level
    min_slowdown: float = 1.05  # smoothed step-time ratio that counts as straggling
    cusum_floor: float = 1.02  # CUSUM alarms only count while still at least this slow
    cusum_cap: float = 16.0  # cap on S, so it recovers within about an hour
    smooth_scrapes: int = 6  # rolling median window (30 min at 5-minute scrapes)
    min_scrapes: int = 3  # shortest interval reported
    low_clock: float = 0.97  # clock / node median below this means the GPU itself is slow
    hot_c: float = 5.0  # degrees above node median that count as running hot


def detect(sig: pd.DataFrame, cfg: StragglerConfig | None = None) -> pd.DataFrame:
    """Flag straggling scrapes. ``sig`` is the output of ``features.scrape_signals``.

    Adds ``step_z``, ``cusum``, ``slowdown`` (smoothed step ratio), ``straggler`` (bool) and
    ``cause`` to the busy rows and returns them.
    """
    cfg = cfg or StragglerConfig()
    d = sig[sig["step_vs_node"].notna()].sort_values(["gpu_uuid", "timestamp"]).copy()
    logr = np.log(d["step_vs_node"].clip(lower=1e-3))
    centred = logr - logr.median()
    scale = 1.4826 * centred.abs().median()
    d["step_z"] = centred / (scale if scale > 0 else 1.0)

    g = d.groupby("gpu_uuid", observed=True, sort=False)

    def rolling_median(col):
        r = g[col].rolling(cfg.smooth_scrapes, min_periods=1).median()
        return r.reset_index(level=0, drop=True).reindex(d.index)

    d["slowdown"] = rolling_median("step_vs_node")
    d["cusum"] = _cusum_by_group(d["gpu_uuid"].to_numpy(), d["step_z"].to_numpy(), cfg.k,
                                 cfg.cusum_cap)
    d["straggler"] = (d["slowdown"] > cfg.min_slowdown) | (
        (d["cusum"] > cfg.h) & (d["slowdown"] > cfg.cusum_floor))

    clock = rolling_median("clock_vs_node")
    temp = rolling_median("temp_vs_node")
    hot = (temp > cfg.hot_c) | d["throttle_thermal"]
    low = clock < cfg.low_clock
    d["cause"] = np.select([low & hot, low], ["thermal", "clock"], default="off_gpu")
    d.loc[~d["straggler"], "cause"] = ""
    return d


def _cusum_by_group(groups: np.ndarray, z: np.ndarray, k: float, cap: float) -> np.ndarray:
    """CUSUM over rows sorted by group, restarting at each new group."""
    out = np.empty(len(z), dtype=float)
    s, prev = 0.0, None
    zz = np.nan_to_num(z.astype(float), nan=0.0).tolist()
    for i, (grp, v) in enumerate(zip(groups.tolist(), zz, strict=True)):
        if grp != prev:
            s, prev = 0.0, grp
        s = min(cap, max(0.0, s + v - k))
        out[i] = s
    return out


def intervals(flags: pd.DataFrame, step: str = "5min", min_scrapes: int = 3) -> pd.DataFrame:
    """Merge consecutive straggling scrapes into intervals with a majority cause."""
    step = pd.Timedelta(step)
    f = flags.sort_values(["gpu_uuid", "timestamp"])
    rows = []
    for uuid, g in f.groupby("gpu_uuid", observed=True, sort=False):
        on = g["straggler"].to_numpy()
        ts = g["timestamp"].to_numpy()
        # New run when the flag flips or there is a gap of more than 2 scrapes.
        brk = np.r_[True, (on[1:] != on[:-1]) | (np.diff(ts) > 2 * step)]
        run = np.cumsum(brk)
        for _, r in g[on].groupby(run[on]):
            if len(r) < min_scrapes:
                continue
            rows.append({
                "gpu_uuid": str(uuid), "node_id": str(r["node_id"].iloc[0]),
                "start": r["timestamp"].iloc[0], "end": r["timestamp"].iloc[-1] + step,
                "scrapes": len(r), "slowdown": float(r["slowdown"].median()),
                "cause": r["cause"].mode().iloc[0],
            })
    return pd.DataFrame(rows, columns=["gpu_uuid", "node_id", "start", "end", "scrapes",
                                       "slowdown", "cause"])


TRUE_CAUSE = {"thermal": "thermal", "chronic": "clock", "transient": "off_gpu"}


def evaluate_stragglers(flags: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Compare flags with injected straggler intervals (simulator ground truth).

    Scrape level: precision and recall of the straggler flag on busy scrapes.
    Interval level: share of injected intervals detected, median detection delay, and how
    often the cause is right on correctly flagged scrapes.
    """
    f = flags[["gpu_uuid", "timestamp", "straggler", "cause"]].copy()
    f["gpu_uuid"] = f["gpu_uuid"].astype(str)
    f["true_kind"] = ""
    t = truth.copy()
    t["gpu_uuid"] = t["gpu_uuid"].astype(str)
    by_gpu = {u: g for u, g in t.groupby("gpu_uuid")}
    for uuid, idx in f.groupby("gpu_uuid").groups.items():
        if uuid not in by_gpu:
            continue
        ts = f.loc[idx, "timestamp"]
        for _, iv in by_gpu[uuid].iterrows():
            m = (ts >= iv["start"]) & (ts < iv["end"])
            f.loc[m[m].index, "true_kind"] = iv["kind"]
    actual = f["true_kind"] != ""
    tp = int((f["straggler"] & actual).sum())
    precision = tp / max(int(f["straggler"].sum()), 1)
    recall = tp / max(int(actual.sum()), 1)

    hit = f[f["straggler"] & actual]
    cause_ok = (hit["cause"] == hit["true_kind"].map(TRUE_CAUSE)).mean() if len(hit) else np.nan

    detected, delays = 0, []
    busy_intervals = 0
    f_by_gpu = {u: g for u, g in f.groupby("gpu_uuid")}
    for _, iv in t.iterrows():
        fg = f_by_gpu.get(iv["gpu_uuid"])
        if fg is None:
            continue
        g = fg[(fg["timestamp"] >= iv["start"]) & (fg["timestamp"] < iv["end"])]
        if g.empty:
            continue  # GPU was idle the whole interval: nothing to detect
        busy_intervals += 1
        on = g[g["straggler"]]
        if len(on):
            detected += 1
            delays.append((on["timestamp"].min() - g["timestamp"].min()) / pd.Timedelta("1min"))
    per_kind = {}
    for kind in TRUE_CAUSE:
        k = f[f["true_kind"] == kind]
        per_kind[kind] = round(float(k["straggler"].mean()), 3) if len(k) else None
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 3),
        "cause_accuracy": round(float(cause_ok), 3),
        "intervals_with_busy_time": busy_intervals,
        "intervals_detected": detected,
        "median_delay_minutes": float(np.median(delays)) if delays else np.nan,
        "recall_by_kind": per_kind,
    }


def run(tel: pd.DataFrame, truth: pd.DataFrame | None = None, step: str = "5min",
        cfg: StragglerConfig | None = None):
    """Detect on raw telemetry. Returns (flags, intervals, metrics or None)."""
    flags = detect(scrape_signals(tel, step), cfg)
    iv = intervals(flags, step, (cfg or StragglerConfig()).min_scrapes)
    metrics = evaluate_stragglers(flags, truth) if truth is not None else None
    return flags, iv, metrics
