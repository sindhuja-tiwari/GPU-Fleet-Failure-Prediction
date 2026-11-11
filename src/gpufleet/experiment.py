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


# ---------------------------------------------------------------------------------------
# Week 6: steadier thresholds, calibration, cost and two-tier alerts
# ---------------------------------------------------------------------------------------


@dataclass
class PolicyResult:
    results: pd.DataFrame
    calibration: dict
    cost_curve: pd.DataFrame
    oof: pd.DataFrame
    info: dict = field(default_factory=dict)

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(out / "policies.csv", index=False)
        self.cost_curve.to_csv(out / "cost_curve.csv", index=False)
        (out / "calibration.json").write_text(json.dumps(self.calibration, indent=2))
        (out / "policy_info.json").write_text(json.dumps(self.info, indent=2, default=str))
        return out


def run_policies(
    feats: pd.DataFrame,
    cost=None,
    max_false_per_1k: float = 5.0,
    val_days: float = 10,
    test_days: float = 14,
    seed: int = 0,
    with_sequence: bool = True,
) -> PolicyResult:
    from .policy import (
        Calibrator,
        CostModel,
        calibration_report,
        oof_scores,
        pick_threshold_by_cost,
        pick_two_tier,
        two_tier_score,
    )

    cost = cost or CostModel()
    split = time_split(feats, val_days, test_days, embargo_hours=72)
    part = split.assign(feats)
    dev = feats[feats["time"] < split.val_end - split.embargo].copy()
    test = feats[part == "test"].copy()

    p_test, oof = {}, pd.DataFrame(index=dev.index)
    for h in (24, 72):
        label = f"label_{h}h"
        oof[f"p{h}"] = oof_scores(dev, lambda: LightGBMModel(seed=seed), label)
        cut = dev["time"].max() - pd.Timedelta(days=5)
        final = LightGBMModel(seed=seed).fit(dev[dev["time"] < cut], dev[dev["time"] >= cut],
                                             label)
        raw = oof[f"p{h}"]
        cal = Calibrator().fit(raw.to_numpy(), dev[label].to_numpy())
        p_test[h] = cal(final.score(test))
        oof[f"p{h}"] = pd.Series(cal(raw.fillna(0).to_numpy()), index=raw.index).where(
            raw.notna())
    ooff = dev.join(oof).dropna(subset=["p24"])
    test = test.assign(p24=p_test[24], p72=p_test[72])

    rows = []

    def add(policy, scored, thr, horizon, label=None):
        rep, fails = evaluate(scored, thr, horizon_hours=horizon, label=label)
        d = rep.as_dict()
        d["cost_per_1k_gpu_days"] = round(
            1000 * cost.total(rep.false_alerts, fails) / rep.gpu_days, 2)
        rows.append({"policy": policy, "eval_horizon": horizon, **d})

    # 1. LightGBM 24h, budget threshold from out-of-fold scores (vs. one window in week 4)
    s = ooff.assign(score=ooff["p24"])
    thr24 = threshold_for_budget(s, max_false_per_1k, 24)
    add("lightgbm 24h, rolling-validation threshold", test.assign(score=test["p24"]), thr24, 24)
    # 2. LightGBM 72h, same
    s = ooff.assign(score=ooff["p72"])
    thr72 = threshold_for_budget(s, max_false_per_1k, 72)
    add("lightgbm 72h, rolling-validation threshold", test.assign(score=test["p72"]), thr72, 72)
    # 3. LightGBM 24h, cost-optimal threshold within the budget
    thr_c, curve = pick_threshold_by_cost(ooff.assign(score=ooff["p24"]), cost, 24,
                                          max_false_per_1k=max_false_per_1k)
    add("lightgbm 24h, cost-optimal threshold", test.assign(score=test["p24"]), thr_c, 24)
    # 4. Two tiers: 72h warning + 24h urgent, cost-optimal within the budget
    t24, t72, _ = pick_two_tier(ooff, cost, 72, max_false_per_1k=max_false_per_1k)
    tt = test.assign(score=two_tier_score(test["p24"], test["p72"], t24, t72))
    add("two-tier (72h warning + 24h urgent)", tt, 1.0, 72, label="label_72h")

    # 5. Sequence model under the week 4 protocol (train / validation / test)
    seq_info = {}
    if with_sequence:
        from .seqmodel import SequenceMLP

        train, val = feats[part == "train"], feats[part == "val"]
        seq = SequenceMLP(feats, seed=seed).fit(train, val, "label_24h")
        v = val.assign(score=seq.score(val))
        thr = threshold_for_budget(v, max_false_per_1k, 24)
        add("sequence MLP 24h (week 4 protocol)", test.assign(score=seq.score(test)), thr, 24)
        lgb = LightGBMModel(seed=seed).fit(train, val, "label_24h")
        v = val.assign(score=lgb.score(val))
        thr = threshold_for_budget(v, max_false_per_1k, 24)
        add("lightgbm 24h (week 4 protocol)", test.assign(score=lgb.score(test)), thr, 24)
        seq_info = {"lookback_hours": seq.lookback}

    calib = {
        "lightgbm_24h_test": calibration_report(test["p24"], test["label_24h"]),
        "lightgbm_72h_test": calibration_report(test["p72"], test["label_72h"]),
    }
    info = {
        "oof_days": 20, "oof_failures": int(ooff.loc[ooff["label_24h"] == 1, "gpu_uuid"].nunique()),
        "val_window_failures": int(feats.loc[(part == "val") & (feats["label_24h"] == 1),
                                             "gpu_uuid"].nunique()),
        "thresholds": {"lightgbm_24h_budget": thr24, "lightgbm_72h_budget": thr72,
                       "lightgbm_24h_cost": thr_c, "two_tier_t24": t24, "two_tier_t72": t72},
        "cost_model": cost.as_dict(), **seq_info,
    }
    return PolicyResult(pd.DataFrame(rows), calib, curve, ooff[["gpu_uuid", "time", "p24",
                                                                  "p72", "label_24h"]], info)


# ---------------------------------------------------------------------------------------
# Week 8: ablations
# ---------------------------------------------------------------------------------------

#: Feature families, by column-name test. A column can belong to more than one family.
FAMILIES = {
    "ECC and retired pages": lambda c: c.startswith(("sbe_", "dbe_", "retired_", "xid63")),
    "NVLink": lambda c: c.startswith("nvlink_"),
    "PCIe": lambda c: c.startswith("pcie_"),
    "temperature and thermal throttling": lambda c: c.startswith(("temp_", "throttle_thermal")),
    "power": lambda c: c.startswith(("power_", "throttle_power")),
    "clock and step time": lambda c: c.startswith(("clock_", "step_")),
    "all peer-relative (vs node)": lambda c: "_vs_node" in c,
    "since-install totals": lambda c: c.endswith("_since_install"),
    "acceleration and trends": lambda c: c.endswith("_accel") or "_trend_" in c,
    "6h windows": lambda c: c.endswith("_6h"),
    "72h windows": lambda c: c.endswith("_72h"),
}


def run_ablations(feats: pd.DataFrame, max_false_per_1k: float = 5.0, val_days: float = 10,
                  test_days: float = 14, seed: int = 0, horizon: int = 24,
                  families: dict | None = None) -> pd.DataFrame:
    """Retrain LightGBM without each feature family (week 4 protocol) and compare."""
    from .features import feature_columns

    families = families or FAMILIES
    split = time_split(feats, val_days, test_days, embargo_hours=72)
    part = split.assign(feats)
    train, val, test = (feats[part == p] for p in ("train", "val", "test"))
    label = f"label_{horizon}h"
    cols = feature_columns(feats)
    rows = []
    for name, test_fn in [("none (full model)", lambda c: False), *families.items()]:
        drop = [c for c in cols if test_fn(c)]
        tr, va, te = (d.drop(columns=drop) for d in (train, val, test))
        model = LightGBMModel(seed=seed).fit(tr, va, label)
        v = va.assign(score=model.score(va))
        thr = threshold_for_budget(v, max_false_per_1k, horizon)
        rep, fails = evaluate(te.assign(score=model.score(te)), thr, horizon_hours=horizon)
        by_mode = fails[fails["mode"] != "sudden"].groupby("mode")["caught"].mean()
        rows.append({"dropped": name, "features_dropped": len(drop), **rep.as_dict(),
                     **{f"recall_{m}": round(float(v), 3) for m, v in by_mode.items()}})
    return pd.DataFrame(rows)
