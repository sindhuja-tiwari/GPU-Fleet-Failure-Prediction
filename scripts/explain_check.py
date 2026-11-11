"""Do alert explanations point at the right failure mode?

Scores unseen fleets with a trained bundle, raises two-tier alerts (24h cooldown), keeps
the true ones, and checks whether the top reason's feature family matches the failure
mode the simulator injected. Writes reports/explanations.csv and prints a summary.

    python scripts/explain_check.py models/demo-bundle data/bench-seed2 data/bench-seed3
"""

import sys
from pathlib import Path

import pandas as pd

from gpufleet.bundle import Bundle
from gpufleet.evaluate import alerts, failure_table
from gpufleet.policy import two_tier_score

FAMILY_MODE = [
    (("sbe_", "dbe_", "retired_", "xid63"), "memory"),
    (("nvlink_",), "nvlink"),
    (("temp_", "throttle_thermal"), "thermal"),
    (("pcie_", "power_", "throttle_power"), "power"),
]


def mode_of(feature: str) -> str:
    for prefixes, mode in FAMILY_MODE:
        if feature.startswith(prefixes):
            return mode
    return "other"


def main(bundle_dir: str, *data_dirs: str) -> None:
    b = Bundle.load(bundle_dir)
    rows = []
    for d in data_dirs:
        f = pd.read_parquet(Path(d) / "features.parquet")
        p = b.probabilities(f)
        f = f.assign(score=two_tier_score(p["p24"], p["p72"], b.t24, b.t72))
        al = alerts(f, 1.0, 24)
        fails = failure_table(f)
        al = al.reset_index().merge(fails, on="gpu_uuid", how="inner").set_index("index")
        lead = (al["failure_time"] - al["time"]) / pd.Timedelta("1h")
        al = al[(lead > 0) & (lead <= 72)].assign(lead_hours=lead)
        why = b.explain(f.loc[al.index])
        for (_, a), reasons in zip(al.iterrows(), why, strict=True):
            top = reasons[0]["feature"] if reasons else ""
            rows.append({"fleet": Path(d).name, "gpu_uuid": a["gpu_uuid"], "mode": a["mode"],
                         "lead_hours": round(a["lead_hours"], 1), "top_feature": top,
                         "top_mode": mode_of(top),
                         "any_top3": a["mode"] in {mode_of(r["feature"]) for r in reasons},
                         "reasons": "; ".join(r["label"] for r in reasons)})
    out = pd.DataFrame(rows)
    out.to_csv(Path("reports") / "explanations.csv", index=False)
    out["match"] = out["top_mode"] == out["mode"]
    print(f"{len(out)} true alerts; the top reason matches the failure mode in "
          f"{out['match'].mean():.1%}, one of the top three in {out['any_top3'].mean():.1%}")
    print(out.groupby("mode")[["match", "any_top3"]].agg(["size", "mean"]).round(2).to_string())
    print(out.groupby(["mode", "top_mode"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
