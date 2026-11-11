"""Combine per-seed baseline results into one table (mean and range across seeds)."""

import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/baselines")
frames = []
for d in sorted(root.glob("seed*")):
    r = pd.read_csv(d / "results.csv")
    r["seed"] = d.name.removeprefix("seed")
    frames.append(r)
allr = pd.concat(frames, ignore_index=True)
allr.to_csv(root / "all_seeds.csv", index=False)

cols = {
    "pr_auc": ("PR-AUC", "{:.2f}"),
    "predictable_recall": ("Predictable failures caught", "{:.0%}"),
    "false_alerts_per_1k_gpu_days": ("False alerts / 1k GPU-days", "{:.1f}"),
    "median_lead_hours": ("Median lead (h)", "{:.0f}"),
    "precision": ("Alert precision", "{:.0%}"),
}
lines = ["| Horizon | Model | " + " | ".join(v[0] for v in cols.values()) + " |",
         "| --- | --- |" + " --- |" * len(cols)]
for (h, model), g in allr.groupby(["horizon", "model"], sort=False):
    cells = []
    for c, (_, fmt) in cols.items():
        mean, lo, hi = g[c].mean(), g[c].min(), g[c].max()
        cells.append(f"{fmt.format(mean)} ({fmt.format(lo)}-{fmt.format(hi)})")
    lines.append(f"| {h}h | {model} | " + " | ".join(cells) + " |")
table = "\n".join(lines) + "\n"
(root / "summary.md").write_text(
    f"Mean (min-max) over {allr['seed'].nunique()} simulated fleets.\n\n" + table)
print(table)
