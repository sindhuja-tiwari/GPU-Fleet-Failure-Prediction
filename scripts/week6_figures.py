"""Figures for the week 6 note, from reports/week6/seed*/."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
REP, OUT = ROOT / "reports" / "week6", ROOT / "docs" / "figures"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
PCT = FuncFormatter(lambda v, _: f"{100 * v:g}%")
SHORT = {
    "lightgbm 24h, rolling-validation threshold": "LightGBM 24h, rolling threshold",
    "lightgbm 72h, rolling-validation threshold": "LightGBM 72h, rolling threshold",
    "lightgbm 24h, cost-optimal threshold": "LightGBM 24h, cost-optimal",
    "two-tier (72h warning + 24h urgent)": "Two-tier (72h warning + 24h urgent)",
    "sequence MLP 24h (week 4 protocol)": "Sequence MLP 24h (week 4 split)",
    "lightgbm 24h (week 4 protocol)": "LightGBM 24h (week 4 split)",
}


def policies():
    runs = pd.concat([pd.read_csv(p).assign(seed=p.parent.name)
                      for p in sorted(REP.glob("seed*/policies.csv"))])
    order = list(SHORT)[::-1]
    g = runs.groupby("policy")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, col, title, target, tlabel in [
        (axes[0], "median_lead_hours", "Median warning time (hours)", 12, "12 h target"),
        (axes[1], "false_alerts_per_1k_gpu_days", "False alerts per 1,000 GPU-days", 5,
         "budget"),
    ]:
        mean, lo, hi = (g[col].agg(f).reindex(order) for f in ("mean", "min", "max"))
        y = range(len(order))
        ax.hlines(y, lo, hi, color=AXIS, lw=2)
        ax.scatter(mean, y, s=40, color=SERIES[0], zorder=3, edgecolor="#fcfcfb", lw=1.5)
        for yi, v in zip(y, mean, strict=True):
            ax.annotate(f"{v:.0f}" if v >= 10 else f"{v:.1f}", (v, yi), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.axvline(target, color=MUTED, ls=(0, (4, 3)), lw=1)
        ax.annotate(tlabel, (target, -0.75), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=MUTED, va="center")
        ax.set_ylim(-1, len(order) - 0.4)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(left=0)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(order)), [SHORT[p] for p in order], color=INK2)
    fig.text(0.01, 0.01, "Test period, 3 fleets: dot = mean, line = range.", fontsize=9,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "week6-policies.png", dpi=200)


def reliability():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
    for ax, key, title in [(axes[0], "lightgbm_24h_test", "Fails within 24h"),
                           (axes[1], "lightgbm_72h_test", "Fails within 72h")]:
        ax.plot([1e-4, 1], [1e-4, 1], color=AXIS, lw=1, ls=(0, (4, 3)))
        ax.annotate("perfect calibration", (1.5e-4, 1.5e-4), xytext=(10, -12),
                    textcoords="offset points", ha="left", fontsize=9, color=MUTED, rotation=36,
                    rotation_mode="anchor")
        for i, p in enumerate(sorted(REP.glob("seed*/calibration.json"))):
            rel = pd.DataFrame(json.loads(p.read_text())[key]["reliability"])
            rel = rel[rel["observed_rate"] > 0]
            ax.plot(rel["mean_predicted"], rel["observed_rate"], "-o", color=SERIES[i], lw=2,
                    ms=6, mec="#fcfcfb", mew=1.5, label=f"fleet {p.parent.name[-1]}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Predicted probability")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Observed failure rate")
    axes[1].legend(frameon=False, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "week6-calibration.png", dpi=200)


if __name__ == "__main__":
    policies()
    reliability()
