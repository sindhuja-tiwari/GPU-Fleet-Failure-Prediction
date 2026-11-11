"""Train and compare the baselines on one labelled feature table.

Protocol: fit on train, early-stop and pick the alert threshold on validation (lowest
threshold within the false-alert budget), then report on test with that threshold
unchanged. The rules baseline has a fixed threshold (any rule firing).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import evaluate, threshold_for_budget
from .labels import time_split
from .models import LightGBMModel, LogisticBaseline, RulesBaseline


@dataclass
class ExperimentResult:
    results: pd.DataFrame  # one row per model x horizon
    by_mode: pd.DataFrame  # recall and lead time per failure mode, per model x horizon
    importance: dict[str, pd.Series] = field(default_factory=dict)
    split: dict = field(default_factory=dict)

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(out / "results.csv", index=False)
        self.by_mode.to_csv(out / "by_mode.csv", index=False)
        (out / "split.json").write_text(json.dumps(self.split, indent=2, default=str))
        for name, imp in self.importance.items():
            imp.rename("gain_share").to_csv(out / f"importance_{name}.csv")
        (out / "report.md").write_text(self.markdown())
        return out

    def markdown(self) -> str:
        r = self.results
        lines = [
            "| Horizon | Model | PR-AUC | Failures caught | Predictable caught "
            "| False alerts / 1k GPU-days | Median lead (h) | Precision |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, x in r.iterrows():
            lines.append(
                f"| {x['horizon']}h | {x['model']} | {x['pr_auc']:.3f} | "
                f"{x['caught']}/{x['failures']} ({x['recall']:.0%}) | "
                f"{x['predictable_caught']}/{x['predictable_failures']} "
                f"({x['predictable_recall']:.0%}) | {x['false_alerts_per_1k_gpu_days']:.1f} | "
                f"{_fmt(x['median_lead_hours'])} | {_fmt_pct(x['precision'])} |"
            )
        return "\n".join(lines) + "\n"


def _fmt(v):
    return "-" if pd.isna(v) else f"{v:.0f}"


def _fmt_pct(v):
    return "-" if pd.isna(v) else f"{v:.0%}"


def run_baselines(
    feats: pd.DataFrame,
    horizons: tuple[int, ...] = (24, 72),
    max_false_per_1k: float = 5.0,
    val_days: float = 10,
    test_days: float = 14,
    models: tuple[str, ...] = ("rules", "logistic", "lightgbm"),
    seed: int = 0,
) -> ExperimentResult:
    split = time_split(feats, val_days, test_days, embargo_hours=max(horizons))
    part = split.assign(feats)
    train, val, test = (feats[part == p].copy() for p in ("train", "val", "test"))

    rows, mode_rows, importance = [], [], {}
    for h in horizons:
        label = f"label_{h}h"
        for name in models:
            if name == "rules":
                model = RulesBaseline()
            elif name == "logistic":
                model = LogisticBaseline().fit(train, val, label)
            elif name == "lightgbm":
                model = LightGBMModel(seed=seed).fit(train, val, label)
                importance[f"lightgbm_{h}h"] = model.importance()
            else:
                raise ValueError(name)
            val["score"] = model.score(val)
            test["score"] = model.score(test)
            if name == "rules":
                thr = RulesBaseline.default_threshold
            else:
                thr = threshold_for_budget(val, max_false_per_1k, h, cooldown_hours=24)
            rep, fails = evaluate(test, thr, horizon_hours=h, cooldown_hours=24, label=label)
            rows.append({"horizon": h, "model": name, **rep.as_dict()})
            for mode, g in fails.groupby("mode"):
                mode_rows.append({
                    "horizon": h, "model": name, "mode": mode, "failures": len(g),
                    "caught": int(g["caught"].sum()),
                    "median_lead_hours": float(g["lead_hours"].median())
                    if g["caught"].any() else np.nan,
                })
    info = {
        "train_end": split.train_end, "val_end": split.val_end,
        "embargo_hours": int(split.embargo / pd.Timedelta("1h")),
        "rows": {p: int((part == p).sum()) for p in ("train", "val", "test", "gap")},
        "max_false_per_1k": max_false_per_1k,
    }
    return ExperimentResult(pd.DataFrame(rows), pd.DataFrame(mode_rows), importance, info)
