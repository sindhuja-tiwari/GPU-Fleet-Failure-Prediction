"""A trained, deployable model bundle: two LightGBM models, calibration and thresholds.

``train_bundle`` follows the week 6 recipe on a labelled feature table: rolling-origin
out-of-fold scores, isotonic calibration, and two-tier thresholds picked by cost within the
false-alert budget. ``Bundle`` loads it back and scores feature rows, with per-row feature
contributions (SHAP values, computed natively by LightGBM) for explaining alerts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import feature_columns

# Plain-language names for features without a time window. Windowed features
# (``<family>_<N>h``) are named from FAMILY_LABELS below, e.g. "PCIe replays (last 6h)".
FEATURE_LABELS = {
    "sbe_accel": "corrected ECC errors accelerating",
    "nvlink_accel": "NVLink CRC errors accelerating",
    "pcie_accel": "PCIe replays accelerating",
    "sbe_since_install": "corrected ECC errors since install",
    "retired_since_install": "memory pages retired since install",
    "nvlink_since_install": "NVLink CRC errors since install",
    "pcie_since_install": "PCIe replays since install",
    "temp_max_1h": "peak temperature in the last hour",
}


# Fallback names by feature family, for features not listed above.
FAMILY_LABELS = {
    "sbe_rate": "corrected ECC errors", "dbe_rate": "uncorrectable ECC errors",
    "retired_rate": "memory pages retired", "nvlink_rate": "NVLink CRC errors",
    "pcie_rate": "PCIe replays", "xid63": "row-remap Xid 63 events",
    "temp_vs_node": "running hotter than its node peers",
    "clock_vs_node": "lower clocks than its node peers",
    "step_vs_node": "slower training steps than its peers",
    "power_vs_node_std": "unstable power draw",
    "throttle_thermal_frac": "thermal throttling", "temp_trend": "temperature rising",
    "throttle_power_frac": "power throttling", "busy_frac": "share of time busy",
    "coverage": "telemetry coverage", "xid_benign": "benign Xid events",
    "resets": "counter resets from reboots",
}


def label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    base, _, window = feature.rpartition("_")
    if base in FAMILY_LABELS and window.endswith("h") and window[:-1].isdigit():
        return f"{FAMILY_LABELS[base]} (last {window})"
    return feature.replace("_", " ")


@dataclass
class Bundle:
    boosters: dict  # horizon -> lightgbm.Booster
    calibration: dict  # horizon -> (x thresholds, y values) of the isotonic fit
    features: list[str]
    t24: float
    t72: float
    meta: dict

    # -- scoring ------------------------------------------------------------------------
    def probabilities(self, feats: pd.DataFrame) -> pd.DataFrame:
        x = feats[self.features].to_numpy(dtype=np.float32)
        out = {}
        for h, booster in self.boosters.items():
            raw = booster.predict(x)
            cx, cy = self.calibration[h]
            out[f"p{h}"] = np.interp(raw, cx, cy)
        return pd.DataFrame(out, index=feats.index)

    def alert_level(self, p: pd.DataFrame) -> pd.Series:
        """'urgent' (24h tier), 'warning' (72h tier) or '' for each row."""
        lvl = np.where(p["p24"] >= self.t24, "urgent",
                       np.where(p["p72"] >= self.t72, "warning", ""))
        return pd.Series(lvl, index=p.index)

    def explain(self, feats: pd.DataFrame, horizon: int = 72, top: int = 3) -> list[list[dict]]:
        """Top features pushing each row's risk up (SHAP contributions, log-odds units)."""
        x = feats[self.features].to_numpy(dtype=np.float32)
        contrib = self.boosters[horizon].predict(x, pred_contrib=True)[:, :-1]
        out = []
        for row, vals in zip(feats.itertuples(index=False), contrib, strict=True):
            order = np.argsort(-vals)[:top]
            out.append([
                {"feature": self.features[i], "label": label(self.features[i]),
                 "value": _num(getattr(row, self.features[i])),
                 "contribution": round(float(vals[i]), 3)}
                for i in order if vals[i] > 0
            ])
        return out

    # -- persistence --------------------------------------------------------------------
    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for h, booster in self.boosters.items():
            booster.save_model(str(out / f"lightgbm_{h}h.txt"))
        meta = {
            **self.meta,
            "features": self.features,
            "thresholds": {"p24": self.t24, "p72": self.t72},
            "calibration": {str(h): {"x": list(map(float, cx)), "y": list(map(float, cy))}
                            for h, (cx, cy) in self.calibration.items()},
        }
        (out / "bundle.json").write_text(json.dumps(meta, indent=2, default=str))
        return out

    @classmethod
    def load(cls, path: str | Path) -> Bundle:
        import lightgbm as lgb

        p = Path(path)
        meta = json.loads((p / "bundle.json").read_text())
        boosters = {h: lgb.Booster(model_file=str(p / f"lightgbm_{h}h.txt")) for h in (24, 72)}
        cal = {int(h): (np.array(v["x"]), np.array(v["y"]))
               for h, v in meta.pop("calibration").items()}
        th = meta.pop("thresholds")
        feats = meta.pop("features")
        return cls(boosters, cal, feats, th["p24"], th["p72"], meta)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, 3)


def train_bundle(feats: pd.DataFrame, max_false_per_1k: float = 5.0, seed: int = 0,
                 cost=None) -> Bundle:
    """Train on all labelled rows of ``feats`` with the week 6 recipe."""
    from .models import LightGBMModel
    from .policy import Calibrator, CostModel, oof_scores, pick_two_tier

    cost = cost or CostModel()
    dev = feats.copy()
    boosters, calibration, oof = {}, {}, pd.DataFrame(index=dev.index)
    for h in (24, 72):
        label_col = f"label_{h}h"
        raw = oof_scores(dev, lambda: LightGBMModel(seed=seed), label_col)
        cal = Calibrator().fit(raw.to_numpy(), dev[label_col].to_numpy())
        cut = dev["time"].max() - pd.Timedelta(days=5)
        final = LightGBMModel(seed=seed).fit(dev[dev["time"] < cut], dev[dev["time"] >= cut],
                                             label_col)
        boosters[h] = final.booster
        calibration[h] = (cal.iso.X_thresholds_, cal.iso.y_thresholds_)
        oof[f"p{h}"] = pd.Series(cal(raw.fillna(0).to_numpy()), index=raw.index).where(
            raw.notna())
    scored = dev.join(oof).dropna(subset=["p24"])
    t24, t72, _ = pick_two_tier(scored, cost, 72, max_false_per_1k=max_false_per_1k)
    meta = {
        "trained_rows": int(len(dev)),
        "trained_from": str(dev["time"].min()), "trained_to": str(dev["time"].max()),
        "failures": int(dev.loc[dev["label_24h"] == 1, "gpu_uuid"].nunique()),
        "max_false_per_1k": max_false_per_1k, "cost_model": cost.as_dict(), "seed": seed,
    }
    return Bundle(boosters, calibration, feature_columns(dev), t24, t72, meta)
