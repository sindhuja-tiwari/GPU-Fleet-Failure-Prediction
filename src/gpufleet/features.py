"""Feature table: one row per GPU per hour, built only from data before that hour.

Pipeline (``build_features``):

1. Snap scrapes to the grid and turn counters into reset-safe increments (``gpufleet.clean``).
2. Per-scrape signals relative to the GPU's node peers at the same moment.
3. Hourly buckets. Bucket ``time = H`` holds scrapes in ``[H - 1h, H)``, so a prediction
   made at ``H`` only sees the past.
4. Each GPU's hourly series is completed (missing hours become empty buckets, not zeros),
   then summarised over trailing 6, 24 and 72 hour windows.

Counter rates are normalised by the time the increments actually cover, so missing
scrapes neither hide errors nor double-count them. ``coverage_24h`` tells the model how
much data a row is based on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import align_timestamps, counter_increments

WINDOWS = (6, 24, 72)  # hours
SCRAPES_PER_HOUR_DEFAULT = 12

COUNTERS = {
    "sbe": "ecc_sbe_total_inc",
    "dbe": "ecc_dbe_total_inc",
    "retired": "retired_pages_total_inc",
    "nvlink": "nvlink_crc_total_inc",
    "pcie": "pcie_replay_total_inc",
}
BENIGN_XIDS = (13, 31, 43)
XID_ROW_REMAP = 63
KEY_COLUMNS = ["gpu_uuid", "node_id", "rack_id", "gpu_index", "time"]


def scrape_signals(tel: pd.DataFrame, step: str = "5min") -> pd.DataFrame:
    """Per-scrape signals: counter increments and comparisons with the GPU's node peers."""
    df = counter_increments(align_timestamps(tel, step))
    # How many scrape intervals each row's increments cover (more than 1 after a gap).
    prev = df.groupby("gpu_uuid", observed=True)["timestamp"].shift()
    df["interval"] = ((df["timestamp"] - prev) / pd.Timedelta(step)).fillna(1).clip(lower=1)
    peers = df.groupby(["timestamp", "node_id"], observed=True)
    busy = df["gpu_util"] > 50
    df["temp_vs_node"] = df["temp_gpu_c"] - peers["temp_gpu_c"].transform("median")
    df.loc[~busy, "temp_vs_node"] = np.nan  # idle GPUs on a busy node are naturally cooler
    df["clock_vs_node"] = df["sm_clock_mhz"] / peers["sm_clock_mhz"].transform("median")
    df.loc[~busy, "clock_vs_node"] = np.nan
    df["step_vs_node"] = df["step_compute_ms"] / peers["step_compute_ms"].transform("median")
    df["power_vs_node"] = df["power_w"] - peers["power_w"].transform("median")
    return df


def hourly_buckets(sig: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-scrape signals into ``[H-1h, H)`` buckets labelled ``time = H``."""
    s = sig.assign(
        time=sig["timestamp"].dt.floor("h") + pd.Timedelta("1h"),
        busy=sig["gpu_util"] > 50,
        xid63=sig["xid"] == XID_ROW_REMAP,
        xid_benign=sig["xid"].isin(BENIGN_XIDS),
    )
    g = s.groupby(["gpu_uuid", "time"], observed=True, sort=False)
    out = pd.DataFrame(
        {
            "node_id": g["node_id"].first(),
            "rack_id": g["rack_id"].first(),
            "gpu_index": g["gpu_index"].first(),
            "scrapes": g.size(),
            "covered": g["interval"].sum(),
            "busy": g["busy"].sum(),
            **{f"{name}_n": g[col].sum() for name, col in COUNTERS.items()},
            "xid63_n": g["xid63"].sum(),
            "xid_benign_n": g["xid_benign"].sum(),
            "reset_n": g["counter_reset"].sum(),
            "temp_vs_node_sum": g["temp_vs_node"].sum(),
            "temp_vs_node_cnt": g["temp_vs_node"].count(),
            "temp_max": g["temp_gpu_c"].max(),
            "clock_vs_node_sum": g["clock_vs_node"].sum(),
            "clock_vs_node_cnt": g["clock_vs_node"].count(),
            "step_vs_node_sum": g["step_vs_node"].sum(),
            "step_vs_node_cnt": g["step_vs_node"].count(),
            "power_vs_node_sum": g["power_vs_node"].sum(),
            "power_vs_node_sq": (s["power_vs_node"] ** 2).groupby(
                [s["gpu_uuid"], s["time"]], observed=True, sort=False).sum(),
            "throttle_thermal_n": g["throttle_thermal"].sum(),
            "throttle_power_n": g["throttle_power"].sum(),
        }
    )
    return out.reset_index()


def _complete_grid(h: pd.DataFrame) -> pd.DataFrame:
    """Give every GPU one row per hour from its first to its last bucket."""
    h = h.sort_values(["gpu_uuid", "time"])
    span = h.groupby("gpu_uuid", observed=True)["time"].agg(["min", "max"])
    n = ((span["max"] - span["min"]) / pd.Timedelta("1h")).astype(int) + 1
    uuids = np.repeat(span.index.to_numpy(), n.to_numpy())
    offs = np.concatenate([np.arange(k) for k in n.to_numpy()])
    times = np.repeat(span["min"].to_numpy(), n.to_numpy()) + offs * np.timedelta64(1, "h")
    grid = pd.DataFrame({"gpu_uuid": uuids, "time": times, "hour_idx": offs})
    h = h.assign(gpu_uuid=h["gpu_uuid"].astype(str))
    out = grid.merge(h, on=["gpu_uuid", "time"], how="left")
    for col in ("node_id", "rack_id", "gpu_index"):
        out[col] = out.groupby("gpu_uuid", sort=False)[col].ffill()
    num = [c for c in h.columns if c not in ("gpu_uuid", "time", "node_id", "rack_id",
                                             "gpu_index")]
    out[num] = out[num].fillna(0)
    return out


def _roll(df: pd.DataFrame, col: str, w: int) -> np.ndarray:
    """Trailing sum over ``w`` hourly rows within each GPU (grid must be complete, sorted)."""
    c = df.groupby("gpu_uuid", sort=False)[col].cumsum().to_numpy(dtype=float)
    idx = df["hour_idx"].to_numpy()
    prev = np.where(idx >= w, np.roll(c, w), 0.0)
    return c - prev


def build_features(
    tel: pd.DataFrame, step: str = "5min", windows: tuple[int, ...] = WINDOWS
) -> pd.DataFrame:
    """Build the hourly feature table from raw telemetry."""
    return features_from_buckets(hourly_buckets(scrape_signals(tel, step)), step, windows)


SINCE_INSTALL = ("sbe", "retired", "nvlink", "pcie")


def features_from_buckets(
    buckets: pd.DataFrame,
    step: str = "5min",
    windows: tuple[int, ...] = WINDOWS,
    offsets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Features from hourly buckets (output of ``hourly_buckets``).

    ``offsets`` (indexed by gpu_uuid) carries history that is no longer in ``buckets``, for
    the streaming scorer: ``hours`` already seen and ``<name>_n`` error counts for the
    since-install features. Windows only need the last ``max(windows)`` hours.
    """
    scrapes_per_hour = pd.Timedelta("1h") / pd.Timedelta(step)
    h = _complete_grid(buckets)
    f = h[["gpu_uuid", "node_id", "rack_id", "gpu_index", "time"]].copy()
    f["gpu_index"] = f["gpu_index"].astype(int)
    off = None
    if offsets is not None and len(offsets):
        off = offsets.reindex(h["gpu_uuid"]).fillna(0).reset_index(drop=True)
        off.index = h.index
    f["hours_seen"] = h["hour_idx"] + 1 + (off["hours"] if off is not None else 0)

    roll = {}
    for w in windows:
        for col in ["scrapes", "covered", "busy", *[f"{n}_n" for n in COUNTERS], "xid63_n",
                    "xid_benign_n", "reset_n", "temp_vs_node_sum", "temp_vs_node_cnt",
                    "clock_vs_node_sum", "clock_vs_node_cnt", "step_vs_node_sum",
                    "step_vs_node_cnt", "power_vs_node_sum", "power_vs_node_sq",
                    "throttle_thermal_n", "throttle_power_n"]:
            roll[(col, w)] = _roll(h, col, w)

    with np.errstate(divide="ignore", invalid="ignore"):
        for w in windows:
            scr = roll[("scrapes", w)]
            f[f"coverage_{w}h"] = scr / (w * scrapes_per_hour)
            cov = roll[("covered", w)]
            per_hour = scrapes_per_hour / np.where(cov > 0, cov, np.nan)
            for name in COUNTERS:
                f[f"{name}_rate_{w}h"] = roll[(f"{name}_n", w)] * per_hour  # events per hour
            f[f"xid63_{w}h"] = roll[("xid63_n", w)]
            f[f"busy_frac_{w}h"] = roll[("busy", w)] / np.where(scr > 0, scr, np.nan)
            for sig in ("temp_vs_node", "clock_vs_node", "step_vs_node"):
                cnt = roll[(f"{sig}_cnt", w)]
                f[f"{sig}_{w}h"] = roll[(f"{sig}_sum", w)] / np.where(cnt > 0, cnt, np.nan)
            mean = roll[("power_vs_node_sum", w)] / np.where(scr > 0, scr, np.nan)
            var = roll[("power_vs_node_sq", w)] / np.where(scr > 0, scr, np.nan) - mean**2
            f[f"power_vs_node_std_{w}h"] = np.sqrt(np.clip(var, 0, None))
            f[f"throttle_thermal_frac_{w}h"] = roll[("throttle_thermal_n", w)] / np.where(
                scr > 0, scr, np.nan)
            f[f"throttle_power_frac_{w}h"] = roll[("throttle_power_n", w)] / np.where(
                scr > 0, scr, np.nan)
        f["xid_benign_24h"] = roll[("xid_benign_n", 24)]
        f["resets_72h"] = roll[("reset_n", 72)]

        # Acceleration: is the recent error rate above the longer-run rate?
        for name in ("sbe", "nvlink", "pcie"):
            f[f"{name}_accel"] = np.log1p(f[f"{name}_rate_6h"]) - np.log1p(f[f"{name}_rate_72h"])
        # Temperature trend vs peers (degrees per hour) over 24h and 72h.
        for w in (24, 72):
            f[f"temp_trend_{w}h"] = _trend(h, "temp_vs_node_sum", "temp_vs_node_cnt", w)
        f["temp_max_1h"] = h["temp_max"].where(h["scrapes"] > 0)

    # Raw values of this one hour, for sequence models (not used by the tabular models).
    with np.errstate(divide="ignore", invalid="ignore"):
        cov1 = np.where(h["covered"] > 0, h["covered"], np.nan)
        for name in ("sbe", "nvlink", "pcie", "retired"):
            f[f"seq_{name}"] = h[f"{name}_n"] * scrapes_per_hour / cov1
        for sig in ("temp_vs_node", "clock_vs_node", "step_vs_node"):
            cnt = h[f"{sig}_cnt"]
            f[f"seq_{sig}"] = h[f"{sig}_sum"] / cnt.where(cnt > 0)
        f["seq_throttle_thermal"] = h["throttle_thermal_n"] / h["scrapes"].where(h["scrapes"] > 0)
        f["seq_coverage"] = h["scrapes"] / scrapes_per_hour

    # Errors since this GPU was first seen (survives counter resets).
    for name in SINCE_INSTALL:
        f[f"{name}_since_install"] = h.groupby("gpu_uuid", sort=False)[f"{name}_n"].cumsum()
        if off is not None:
            f[f"{name}_since_install"] += off[f"{name}_n"]
    f["time"] = pd.to_datetime(f["time"]).dt.as_unit("ms")
    for c in ("gpu_uuid", "node_id", "rack_id"):
        f[c] = f[c].astype(str)
    return f.reset_index(drop=True)


def _trend(h: pd.DataFrame, sum_col: str, cnt_col: str, w: int) -> np.ndarray:
    """Least-squares slope of the hourly mean over the trailing ``w`` hours."""
    has = (h[cnt_col] > 0).astype(float)
    y = (h[sum_col] / h[cnt_col].where(h[cnt_col] > 0)).fillna(0)
    x = h["hour_idx"].astype(float)
    tmp = pd.DataFrame({"gpu_uuid": h["gpu_uuid"], "hour_idx": h["hour_idx"], "n": has,
                        "sx": x * has, "sy": y * has, "sxy": x * y * has, "sxx": x * x * has})
    n, sx, sy, sxy, sxx = (_roll(tmp, c, w) for c in ("n", "sx", "sy", "sxy", "sxx"))
    den = n * sxx - sx**2
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * sxy - sx * sy) / den
    return np.where((n >= w / 4) & (den > 0), slope, np.nan)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: everything except keys, labels and analysis-only columns."""
    skip = set(KEY_COLUMNS) | {"hours_seen", "score", "part"}
    return [c for c in df.columns
            if c not in skip
            and not c.startswith(("label_", "ttf_", "fail_", "mode", "seq_"))
            and pd.api.types.is_numeric_dtype(df[c])]
