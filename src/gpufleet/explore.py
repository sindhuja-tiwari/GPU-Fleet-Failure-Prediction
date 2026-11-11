"""Exploration helpers and figures for a simulated fleet.

Used by ``notebooks/01_explore.ipynb`` and ``gpufleet plots``. The signal helpers
(``scrape_signals``, ``hourly_signals``, ``epochs``) are the first draft of the week 3
features, so they only use data available at each point in time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .clean import align_timestamps, find_gaps
from .features import scrape_signals  # noqa: F401  (re-exported for the notebook)

# Reference categorical palette (fixed order) plus neutrals.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, GRID, HEALTHY = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"

MODE_SIGNAL = {
    # mode: (signal column, axis label, log scale)
    "memory": ("sbe_per_h", "Corrected ECC errors per hour", True),
    "thermal": ("temp_vs_node", "Temperature above node median (C)", False),
    "nvlink": ("nvlink_per_h", "NVLink CRC errors per hour", True),
    "power": ("pcie_per_h", "PCIe replays per hour", True),
    "sudden": ("temp_vs_node", "Temperature above node median (C)", False),
}


@dataclass
class Dataset:
    telemetry: pd.DataFrame
    failures: pd.DataFrame
    stragglers: pd.DataFrame
    maintenance: pd.DataFrame
    step: str

    @classmethod
    def load(cls, data_dir: str | Path) -> Dataset:
        import json

        d = Path(data_dir)
        step = f"{json.loads((d / 'config.json').read_text())['step_minutes']}min"
        return cls(
            pd.read_parquet(d / "telemetry.parquet"),
            pd.read_parquet(d / "failures.parquet"),
            pd.read_parquet(d / "stragglers.parquet"),
            pd.read_parquet(d / "maintenance.parquet"),
            step,
        )

    @classmethod
    def from_result(cls, result) -> Dataset:
        return cls(result.telemetry, result.failures, result.stragglers, result.maintenance,
                   f"{result.config.step_minutes}min")


def hourly_signals(sig: pd.DataFrame) -> pd.DataFrame:
    """Roll per-scrape signals up to one row per GPU per hour."""
    h = sig.assign(hour=sig["timestamp"].dt.floor("h"))
    g = h.groupby(["gpu_uuid", "hour"], observed=True)
    out = pd.DataFrame(
        {
            "node_id": g["node_id"].first(),
            "sbe_per_h": g["ecc_sbe_total_inc"].sum(),
            "nvlink_per_h": g["nvlink_crc_total_inc"].sum(),
            "pcie_per_h": g["pcie_replay_total_inc"].sum(),
            "retired_per_h": g["retired_pages_total_inc"].sum(),
            "temp_vs_node": g["temp_vs_node"].mean(),
            "power_vs_node_std": g["power_vs_node"].std(),
            "clock_vs_node": g["clock_vs_node"].mean(),
            "step_vs_node": g["step_vs_node"].mean(),
            "scrapes": g.size(),
        }
    )
    return out.reset_index()


def epochs(hourly: pd.DataFrame, failures: pd.DataFrame, signal: str, mode: str,
           hours: int = 168) -> pd.DataFrame:
    """One column per failed GPU of ``mode``: ``signal`` by whole hours before failure."""
    obs = failures[failures["observed"] & (failures["mode"] == mode)]
    cols = {}
    for _, f in obs.iterrows():
        g = hourly[hourly["gpu_uuid"].astype(str) == f["gpu_uuid"]]
        rel = ((g["hour"] - f["failure_time"].floor("h")) / pd.Timedelta("1h")).astype(int)
        s = pd.Series(g[signal].to_numpy(), index=rel.to_numpy())
        cols[f["gpu_uuid"]] = s[(s.index > -hours) & (s.index < 0)]  # drop the failure hour
    return pd.DataFrame(cols).sort_index()


def healthy_hours(hourly: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """Hourly rows of GPUs that never started failing in the window."""
    bad = set(failures["gpu_uuid"])
    return hourly[~hourly["gpu_uuid"].astype(str).isin(bad)]


def warning_time(hourly: pd.DataFrame, failures: pd.DataFrame, q: float = 0.999,
                 hours: int = 240, sustain: int = 2) -> pd.DataFrame:
    """How early each failure's mode signal leaves the healthy range.

    The threshold is the ``q`` quantile of the signal over healthy GPU-hours. Warning time
    is the hours before failure when the signal first stays above it for ``sustain``
    hours in a row (NaN if it never does). This uses the true failure mode to pick the
    signal, so it is an upper bound on what a single-signal rule can do, not a model.
    """
    healthy = healthy_hours(hourly, failures)
    rows = []
    for mode, (signal, _, _) in MODE_SIGNAL.items():
        thresh = float(np.nanquantile(healthy[signal], q))
        ep = epochs(hourly, failures, signal, mode, hours)
        for uuid in ep.columns:
            above = (ep[uuid] > thresh).astype(int)
            run = above.groupby((above != above.shift()).cumsum()).cumsum()
            hits = run[run >= sustain].index
            first = -(hits.min() - sustain + 1) if len(hits) else np.nan
            rows.append({"gpu_uuid": uuid, "mode": mode, "signal": signal,
                         "threshold": round(thresh, 2), "warning_hours": first})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------


def _style(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK_2)


def fig_failure_signatures(hourly: pd.DataFrame, failures: pd.DataFrame, hours: int = 168):
    """Each failure mode's warning signal in the week before failure, against healthy GPUs."""
    import matplotlib.pyplot as plt

    modes = list(MODE_SIGNAL)
    fig, axes = plt.subplots(1, len(modes), figsize=(4 * len(modes), 3.4), sharex=True)
    healthy = healthy_hours(hourly, failures)
    for ax, mode in zip(axes, modes, strict=True):
        signal, label, log = MODE_SIGNAL[mode]
        ep = epochs(hourly, failures, signal, mode, hours)
        h = healthy[signal].dropna()
        lo, mid, hi = np.percentile(h, [1, 50, 99])
        ax.axhspan(lo, hi, color=HEALTHY, alpha=0.18, linewidth=0)
        ax.axhline(mid, color=HEALTHY, linewidth=1.5)
        for c in ep.columns:
            ax.plot(ep.index, ep[c], color=ORANGE, alpha=0.25, linewidth=1)
        if ep.shape[1]:
            ax.plot(ep.index, ep.median(axis=1).rolling(3, center=True, min_periods=1).median(),
                    color=ORANGE, linewidth=2)
        if log:
            ax.set_yscale("symlog", linthresh=1)
        n = ep.shape[1]
        _style(ax, f"{mode} ({n} failure{'s' if n != 1 else ''})", "Hours before failure", label)
    axes[0].text(0.02, 0.97, "healthy GPUs: median and 1st-99th percentile band",
                 transform=axes[0].transAxes, fontsize=7, color=INK_2, va="top")
    axes[0].text(0.02, 0.88, "failing GPUs: each one, and their median",
                 transform=axes[0].transAxes, fontsize=7, color=ORANGE, va="top")
    fig.tight_layout()
    return fig


def fig_lookalikes(hourly: pd.DataFrame, sig: pd.DataFrame, failures: pd.DataFrame):
    """Why single signals are not enough: noisy healthy GPUs and benign Xids."""
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))

    # Left: worst 24h corrected-ECC count of each healthy GPU vs. the last 24h of memory failures
    healthy = healthy_hours(hourly, failures)
    roll = (healthy.sort_values("hour").groupby("gpu_uuid", observed=True)["sbe_per_h"]
            .rolling(24, min_periods=1).sum())
    worst = roll.groupby(level=0, observed=True).max()
    mem = epochs(hourly, failures, "sbe_per_h", "memory", hours=25).sum()
    edges = [0, 1, 3, 10, 30, 100, 300, np.inf]
    names = ["0", "1-2", "3-9", "10-29", "30-99", "100-299", "300+"]
    pos = np.arange(len(names))
    for vals, color, label, shift in [
        (worst, HEALTHY, f"healthy GPUs, their worst day (n={len(worst)})", -0.2),
        (mem, ORANGE, f"memory failures, last day before failing (n={len(mem)})", 0.2),
    ]:
        share = pd.cut(vals, edges, right=False, labels=names).value_counts(normalize=True)
        a.bar(pos + shift, share.reindex(names).fillna(0) * 100, width=0.38, color=color,
              label=label)
    a.set_xticks(pos, names)
    a.legend(fontsize=7, frameon=False)
    _style(a, "Corrected ECC errors in 24 hours", "Errors", "% of GPUs")

    # Right: what each Xid code turns out to mean
    x = sig.loc[sig["xid"] != 0, ["gpu_uuid", "xid", "timestamp"]].copy()
    fatal = failures.loc[failures["observed"], ["gpu_uuid", "fatal_xid"]]
    x = x.merge(fatal, how="left", left_on=x["gpu_uuid"].astype(str), right_on="gpu_uuid",
                suffixes=("", "_f"))
    last = x["xid"] == x["fatal_xid"]
    counts = pd.DataFrame({
        "fatal (GPU failed)": x[last].groupby("xid").size(),
        "not fatal": x[~last].groupby("xid").size(),
    }).fillna(0).astype(int).sort_index()
    labels = [str(c) for c in counts.index]
    b.bar(labels, counts["not fatal"], color=HEALTHY, label="did not end the GPU")
    b.bar(labels, counts["fatal (GPU failed)"], bottom=counts["not fatal"], color=ORANGE,
          label="last thing the GPU logged")
    b.legend(fontsize=7, frameon=False)
    _style(b, "Xid codes seen", "Xid code", "Events")
    fig.tight_layout()
    return fig


def fig_stragglers(sig: pd.DataFrame, stragglers: pd.DataFrame, max_points: int = 4000,
                   seed: int = 0):
    """Step time vs. SM clock, both relative to node peers, by straggler cause."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    d = sig.dropna(subset=["step_vs_node", "clock_vs_node"])[
        ["gpu_uuid", "timestamp", "step_vs_node", "clock_vs_node"]
    ]
    d = d.assign(kind="healthy")
    for _, s in stragglers.iterrows():
        m = ((d["gpu_uuid"].astype(str) == s["gpu_uuid"]) & (d["timestamp"] >= s["start"])
             & (d["timestamp"] < s["end"]))
        d.loc[m, "kind"] = s["kind"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    style = [("healthy", HEALTHY, 6), ("chronic", BLUE, 10), ("transient", ORANGE, 10),
             ("thermal", AQUA, 10)]
    for kind, color, size in style:
        k = d[d["kind"] == kind]
        if len(k) > max_points:
            k = k.iloc[rng.choice(len(k), max_points, replace=False)]
        ax.scatter(k["clock_vs_node"], k["step_vs_node"], s=size, color=color, alpha=0.5,
                   linewidths=0, label=f"{kind} ({len(d[d['kind'] == kind]):,} scrapes)")
    ax.legend(fontsize=7, frameon=False, loc="upper right", markerscale=1.5)
    _style(ax, "Stragglers: slow steps, and whether clocks explain them",
           "SM clock / node median", "Step time / node median")
    fig.tight_layout()
    return fig


def fig_collection(ds: Dataset, nodes: int = 16):
    """Timeline of what interrupts the data: maintenance, exporter outages, failures."""
    import matplotlib.pyplot as plt

    tel = align_timestamps(ds.telemetry, ds.step)
    gaps = find_gaps(tel, ds.step, min_missing=3)
    names = sorted(tel["node_id"].astype(str).unique())[:nodes]
    fig, ax = plt.subplots(figsize=(11, 0.28 * len(names) + 1.2))
    m = ds.maintenance
    colors = {"reboot": BLUE, "driver_upgrade": BLUE, "gpu_swap": BLUE, "drain": AQUA}
    for i, node in enumerate(names):
        g = gaps[gaps["node_id"] == node].drop_duplicates(["start", "end"])
        for _, r in g.iterrows():
            ax.barh(i, r["end"] - r["start"], left=r["start"], height=0.5, color=HEALTHY)
        for _, r in m[m["node_id"] == node].iterrows():
            ax.barh(i, r["end"] - r["start"], left=r["start"], height=0.5,
                    color=colors[r["kind"]], alpha=0.85)
        f = ds.failures[(ds.failures["node_id"] == node) & ds.failures["observed"]]
        ax.scatter(f["failure_time"], [i] * len(f), marker="x", color=ORANGE, s=30, zorder=3)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ax.legend(handles=[
        Patch(color=HEALTHY, label="missing data (3+ scrapes)"),
        Patch(color=BLUE, label="reboot / driver upgrade / GPU swap (counters reset)"),
        Patch(color=AQUA, label="drain (no jobs, telemetry continues)"),
        Line2D([], [], marker="x", color=ORANGE, linestyle="", label="GPU failure"),
    ], fontsize=7, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    _style(ax, f"Collection interruptions, first {len(names)} nodes")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def save_all_figures(data_dir: str | Path, out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    ds = Dataset.load(data_dir)
    sig = scrape_signals(ds.telemetry, ds.step)
    hourly = hourly_signals(sig)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = {
        "failure-signatures.png": fig_failure_signatures(hourly, ds.failures),
        "lookalikes.png": fig_lookalikes(hourly, sig, ds.failures),
        "stragglers.png": fig_stragglers(sig, ds.stragglers),
        "collection.png": fig_collection(ds),
    }
    paths = []
    for name, fig in figs.items():
        fig.savefig(out / name, dpi=110, facecolor="white")
        paths.append(out / name)
    return paths
