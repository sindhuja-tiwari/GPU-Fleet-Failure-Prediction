"""Baseline failure predictors: operator rules, logistic regression and LightGBM.

Every model has the same interface: ``fit(train, val, label)`` and ``score(df)``, where the
frames are labelled feature tables and higher scores mean more likely to fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import feature_columns

# Fixed thresholds in the spirit of common fleet health checks (NVIDIA's guidance on
# retired pages and row remapping, plus typical SRE alerts). Deliberately not tuned on
# this data: this is what a team would run before having a model.
RULES: dict[str, tuple[str, float]] = {
    "uncorrectable ECC in 24h": ("dbe_rate_24h", 0.0),
    "page retired / row remapped in 24h": ("retired_rate_24h", 0.0),
    "corrected ECC > 2 per hour over 6h": ("sbe_rate_6h", 2.0),
    "NVLink CRC > 2 per hour over 6h": ("nvlink_rate_6h", 2.0),
    "PCIe replays > 1 per hour over 6h": ("pcie_rate_6h", 1.0),
    "thermally throttled > 25% of the last 6h": ("throttle_thermal_frac_6h", 0.25),
    "over 85 C in the last hour": ("temp_max_1h", 85.0),
}


class RulesBaseline:
    """Score = number of rules firing. Any rule firing is an alert (threshold 1)."""

    name = "rules"
    default_threshold = 1.0

    def fit(self, train=None, val=None, label=None):
        return self

    def fired(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: df[col].fillna(0) > thr
                             for name, (col, thr) in RULES.items()})

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.fired(df).sum(axis=1).to_numpy(dtype=float)


def _model_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=np.float32)


class LogisticBaseline:
    """Logistic regression on log-scaled, standardised features."""

    name = "logistic"

    def __init__(self, C: float = 0.1):
        self.C = C

    def _prep(self, df):
        x = df[self.cols].astype(float).copy()
        for c in self.log_cols:
            x[c] = np.sign(x[c]) * np.log1p(np.abs(x[c]))
        return x.fillna(self.fill).to_numpy()

    def fit(self, train, val=None, label="label_24h"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        train = train[train[label].notna()]
        self.cols = feature_columns(train)
        self.log_cols = [c for c in self.cols
                         if "rate" in c or "since_install" in c or c.startswith("xid")]
        self.fill = train[self.cols].median()
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000),
        )
        self.model.fit(self._prep(train), train[label].to_numpy())
        return self

    def score(self, df):
        return self.model.predict_proba(self._prep(df))[:, 1]


class LightGBMModel:
    """Gradient-boosted trees, early-stopped on validation PR-AUC."""

    name = "lightgbm"

    def __init__(self, seed: int = 0, **params):
        self.params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "metric": "average_precision",
            "verbosity": -1,
            "seed": seed,
            "num_threads": 4,
            "deterministic": True,
            "force_row_wise": True,
            **params,
        }

    def fit(self, train, val=None, label="label_24h"):
        import lightgbm as lgb

        train = train[train[label].notna()]
        self.cols = feature_columns(train)
        dtrain = lgb.Dataset(_model_matrix(train, self.cols), train[label].to_numpy(),
                             feature_name=self.cols, free_raw_data=False)
        callbacks = [lgb.log_evaluation(0)]
        valid = []
        if val is not None:
            val = val[val[label].notna()]
            valid = [lgb.Dataset(_model_matrix(val, self.cols), val[label].to_numpy(),
                                 reference=dtrain)]
            callbacks.append(lgb.early_stopping(50, verbose=False))
        self.booster = lgb.train(self.params, dtrain, num_boost_round=1000,
                                 valid_sets=valid, callbacks=callbacks)
        return self

    def score(self, df):
        return self.booster.predict(_model_matrix(df, self.cols),
                                    num_iteration=self.booster.best_iteration or None)

    def importance(self, top: int = 15) -> pd.Series:
        imp = pd.Series(self.booster.feature_importance("gain"), index=self.cols)
        return (imp / imp.sum()).sort_values(ascending=False).head(top)
