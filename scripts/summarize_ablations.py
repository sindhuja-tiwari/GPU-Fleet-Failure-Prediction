"""Average reports/ablations/seed*.csv into reports/ablations/summary.md."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "reports" / "ablations"


def main() -> None:
    runs = pd.concat([pd.read_csv(p) for p in sorted(ROOT.glob("seed*.csv"))])
    modes = sorted(c for c in runs.columns if c.startswith("recall_") and c != "recall_")
    cols = ["features_dropped", "pr_auc", "predictable_recall",
            "false_alerts_per_1k_gpu_days", "median_lead_hours", *modes]
    m = runs.groupby("dropped", sort=False)[cols].mean()
    full = m.iloc[0]
    lines = [
        "| Features removed | Count | PR-AUC | Change | Predictable caught "
        "| False alerts / 1k | Median lead (h) | "
        + " | ".join(c.removeprefix("recall_") for c in modes) + " |",
        "| --- " * (7 + len(modes)) + "|",
    ]
    for name, r in m.iterrows():
        delta = "" if name == m.index[0] else f"{r['pr_auc'] - full['pr_auc']:+.3f}"
        lines.append(
            f"| {name} | {r['features_dropped']:.0f} | {r['pr_auc']:.3f} | {delta} | "
            f"{r['predictable_recall']:.0%} | {r['false_alerts_per_1k_gpu_days']:.1f} | "
            f"{r['median_lead_hours']:.0f} | "
            + " | ".join(f"{r[c]:.0%}" for c in modes) + " |")
    n = runs["dropped"].value_counts().iloc[0]
    out = "\n".join([f"Mean of {n} fleets, LightGBM 24h, week 4 protocol. Last columns: "
                     "share of each failure mode caught.", "", *lines, ""])
    (ROOT / "summary.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
