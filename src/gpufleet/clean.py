"""Helpers for messy telemetry: clock skew, gaps and counter resets.

These work the same on simulated and real data, and every later stage (features,
models, serving) should go through them rather than reading raw counters or timestamps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import COUNTER_COLUMNS


def align_timestamps(df: pd.DataFrame, step: str | pd.Timedelta = "5min") -> pd.DataFrame:
    """Snap each scrape to the nearest step boundary.

    Node clocks drift by seconds, so raw timestamps from different nodes never match
    exactly. After alignment, rows from the same scrape round share a timestamp and can
    be compared across nodes. If two scrapes of one GPU land on the same step, the later
    one wins. The original time is kept in ``raw_timestamp``.
    """
    out = df.copy()
    out["raw_timestamp"] = out["timestamp"]
    out["timestamp"] = out["timestamp"].dt.round(pd.Timedelta(step))
    out = out.sort_values(["gpu_uuid", "raw_timestamp"])
    out = out.drop_duplicates(["gpu_uuid", "timestamp"], keep="last")
    return out.sort_values(["timestamp", "node_id", "gpu_index"]).reset_index(drop=True)


def counter_increments(
    df: pd.DataFrame, columns: list[str] | None = None, suffix: str = "_inc"
) -> pd.DataFrame:
    """Add per-interval counts for cumulative counters, handling resets.

    For each GPU, the increment is the difference from its previous row. When a counter
    goes down (reboot or driver reload), the new value is taken as the count since the
    reset. The first row of each GPU gets 0, since nothing before it is known.
    Adds ``<column><suffix>`` for each counter and a boolean ``counter_reset`` column.
    """
    columns = columns or COUNTER_COLUMNS
    out = df.sort_values(["gpu_uuid", "timestamp"]).copy()
    by_gpu = out.groupby("gpu_uuid", observed=True)
    any_reset = np.zeros(len(out), dtype=bool)
    for col in columns:
        prev = by_gpu[col].shift()
        diff = out[col] - prev
        reset = (diff < 0).to_numpy()
        any_reset |= reset
        inc = np.where(reset, out[col], diff)
        out[col + suffix] = np.nan_to_num(inc, nan=0).astype(np.int64)
    out["counter_reset"] = any_reset
    return out.sort_index()


def find_gaps(
    df: pd.DataFrame, step: str | pd.Timedelta = "5min", min_missing: int = 1
) -> pd.DataFrame:
    """List gaps in each GPU's data where at least ``min_missing`` scrapes are missing.

    Returns one row per gap: gpu_uuid, node_id, start (last row before), end (first row
    after), missing (number of scrapes lost).
    """
    step = pd.Timedelta(step)
    d = df.sort_values(["gpu_uuid", "timestamp"])[["gpu_uuid", "node_id", "timestamp"]]
    prev = d.groupby("gpu_uuid", observed=True)["timestamp"].shift()
    missing = ((d["timestamp"] - prev) / step).round() - 1
    mask = missing >= min_missing
    return pd.DataFrame(
        {
            "gpu_uuid": d.loc[mask, "gpu_uuid"].astype(str).to_numpy(),
            "node_id": d.loc[mask, "node_id"].astype(str).to_numpy(),
            "start": prev[mask].to_numpy(),
            "end": d.loc[mask, "timestamp"].to_numpy(),
            "missing": missing[mask].astype(int).to_numpy(),
        }
    )


def quality_report(df: pd.DataFrame, step: str | pd.Timedelta = "5min") -> dict:
    """Headline data-quality numbers for a telemetry frame."""
    step = pd.Timedelta(step)
    aligned = align_timestamps(df, step)
    span = aligned.groupby("gpu_uuid", observed=True)["timestamp"].agg(["min", "max"])
    expected = int((((span["max"] - span["min"]) / step).round() + 1).sum())
    gaps = find_gaps(aligned, step)
    inc = counter_increments(aligned)
    offset = (aligned["raw_timestamp"] - aligned["timestamp"]).dt.total_seconds()
    node_skew = offset.groupby(aligned["node_id"].astype(str)).median()
    return {
        "rows": len(df),
        "gpus": int(df["gpu_uuid"].nunique()),
        "completeness": round(len(aligned) / max(expected, 1), 4),
        "gaps": len(gaps),
        "gaps_over_1h": int((gaps["missing"] * step > pd.Timedelta("1h")).sum()),
        "counter_resets": int(inc["counter_reset"].sum()),
        "max_node_skew_s": round(float(node_skew.abs().max()), 1) if len(node_skew) else 0.0,
        "duplicate_scrapes": len(df) - len(aligned),
    }
