"""Turning scores into an alerting policy: steadier thresholds, calibration and cost.

Week 4 picked each threshold from one 10-day validation window with about 20 failures,
and recall swung between fleets. Here:

* **Rolling-origin out-of-fold scores.** The pre-test period is cut into several windows.
  For each, a model trained only on data before it (minus an embargo) scores it. Stitched
  together, these out-of-fold scores cover far more failures than one validation window.
* **Calibration.** Isotonic regression on the out-of-fold scores maps raw model output to
  a probability of failing within the horizon.
* **Cost.** Thresholds are picked to minimise an explicit cost, so early warnings are
  worth more than late ones (see ``CostModel``).
* **Two tiers.** A 72h model raises early "warning" alerts and a 24h model raises "urgent"
  ones. A GPU alerts when either fires.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .evaluate import evaluate


@dataclass
class CostModel:
    """Relative costs, in units of "one false alert".

    Defaults, for a training fleet: a false alert costs a node drain and a health check (1).
    A missed failure crashes a job, loses work since the last checkpoint and needs an
    emergency swap (20). A caught failure still costs a planned swap (2), plus up to 8 more
    the less warning there was: with 12 or more hours the job can be moved at a checkpoint.
    """

    false_alert: float = 1.0
    missed: float = 20.0
    caught_base: float = 2.0
    late_penalty: float = 8.0
    plan_hours: float = 12.0

    def total(self, false_alerts: int, fails: pd.DataFrame) -> float:
        caught = fails[fails["caught"]]
        lead = caught["lead_hours"].clip(upper=self.plan_hours)
        late = (1 - lead / self.plan_hours).sum()
        n_missed = int((~fails["caught"]).sum())
        return float(self.false_alert * false_alerts + self.missed * n_missed
                     + self.caught_base * len(caught) + self.late_penalty * late)

    def as_dict(self):
        return asdict(self)


def rolling_folds(dev: pd.DataFrame, n_folds: int = 4, fold_days: float = 5,
                  embargo_hours: int = 72):
    """Yield (train_index, score_index) for expanding-window folds at the end of ``dev``."""
    end = dev["time"].max() + pd.Timedelta("1h")
    for j in range(n_folds, 0, -1):
        start = end - pd.Timedelta(days=fold_days * j)
        stop = start + pd.Timedelta(days=fold_days)
        train = dev.index[dev["time"] < start - pd.Timedelta(hours=embargo_hours)]
        score = dev.index[(dev["time"] >= start) & (dev["time"] < stop)]
        if len(train) and len(score):
            yield train, score


def oof_scores(dev: pd.DataFrame, make_model, label: str, n_folds: int = 4,
               fold_days: float = 5, embargo_hours: int = 72, es_days: float = 5) -> pd.Series:
    """Out-of-fold scores for the last ``n_folds * fold_days`` days of ``dev``.

    ``make_model()`` returns an unfitted model with ``fit(train, val, label)`` and
    ``score(df)``. The last ``es_days`` of each fold's training data are used for early
    stopping.
    """
    out = pd.Series(np.nan, index=dev.index)
    used = 0
    for tr_idx, sc_idx in rolling_folds(dev, n_folds, fold_days, embargo_hours):
        tr = dev.loc[tr_idx]
        cut = tr["time"].max() - pd.Timedelta(days=es_days)
        fit = tr[(tr["time"] < cut) & tr[label].notna()]
        if fit.empty or fit[label].sum() == 0:
            continue  # too little history before this fold to train on
        model = make_model().fit(tr[tr["time"] < cut], tr[tr["time"] >= cut], label)
        out.loc[sc_idx] = model.score(dev.loc[sc_idx])
        used += 1
    if not used:
        days = (dev["time"].max() - dev["time"].min()) / pd.Timedelta("1D")
        raise ValueError(
            f"not enough history for rolling validation: {days:.0f} days with labels, and "
            f"each fold needs failures in the {es_days:g}+ days before it. Use a longer "
            "dataset (for simulated data, --days 60).")
    return out


class Calibrator:
    """Isotonic regression from raw score to probability."""

    def fit(self, scores, labels):
        from sklearn.isotonic import IsotonicRegression

        ok = ~np.isnan(scores) & ~np.isnan(labels)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        self.iso.fit(np.asarray(scores)[ok], np.asarray(labels)[ok])
        return self

    def __call__(self, scores):
        return self.iso.predict(np.asarray(scores, dtype=float))


def calibration_report(prob, labels, bins: int = 10) -> dict:
    """Brier score, expected calibration error, and a reliability table (quantile bins)."""
    prob, labels = np.asarray(prob, dtype=float), np.asarray(labels, dtype=float)
    ok = ~np.isnan(prob) & ~np.isnan(labels)
    p, y = prob[ok], labels[ok]
    # Most rows are near zero, so bin by the top of the score distribution.
    edges = np.unique(np.quantile(p, [0, 0.9, 0.99, 0.995, 0.999, 0.9995, 1]))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows, ece = [], 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "rows": int(m.sum()), "mean_predicted": float(p[m].mean()),
                         "observed_rate": float(y[m].mean())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece),
            "base_rate": float(y.mean()), "reliability": rows}


def tail_thresholds(scores: pd.Series, n: int = 25, max_n: int = 40) -> np.ndarray:
    """Candidate thresholds in the top 5% of scores.

    Quantiles spaced on a log scale, plus every distinct value in the tail: calibrated
    scores are step functions, and quantiles alone can jump over whole steps.
    """
    v = scores.dropna().to_numpy()
    levels = 1 - np.logspace(-5, -1.3, n)
    cands = np.unique(np.concatenate([np.quantile(v, levels),
                                      np.unique(v[v >= np.quantile(v, 0.95)])]))
    if len(cands) > max_n:
        cands = cands[np.unique(np.linspace(0, len(cands) - 1, max_n).round().astype(int))]
    return cands


def pick_threshold_by_cost(scored: pd.DataFrame, cost: CostModel, horizon_hours: int,
                           candidates=None, max_false_per_1k: float | None = None):
    """Threshold on ``score`` that minimises cost (optionally within a false-alert budget)."""
    candidates = tail_thresholds(scored["score"]) if candidates is None else candidates
    best, curve = None, []
    for thr in candidates:
        rep, fails = evaluate(scored, thr, horizon_hours=horizon_hours)
        c = cost.total(rep.false_alerts, fails)
        curve.append({"threshold": float(thr), "cost": c, "recall": rep.recall,
                      "false_per_1k": rep.false_alerts_per_1k_gpu_days,
                      "median_lead": rep.median_lead_hours})
        within = max_false_per_1k is None or rep.false_alerts_per_1k_gpu_days <= max_false_per_1k
        if within and (best is None or c < best[1]):
            best = (float(thr), c)
    if best is None:  # nothing within budget: take the strictest threshold
        best = (float(max(candidates)), np.nan)
    return best[0], pd.DataFrame(curve)


def two_tier_score(p24: np.ndarray, p72: np.ndarray, t24: float, t72: float) -> np.ndarray:
    """>= 1 when either tier fires. Scaled so one threshold (1.0) covers both tiers."""
    return np.maximum(np.asarray(p24) / t24, np.asarray(p72) / t72)


def pick_two_tier(scored: pd.DataFrame, cost: CostModel, horizon_hours: int = 72,
                  n: int = 12, max_false_per_1k: float | None = None, tie_tol: float = 0.02):
    """Grid-search (t24, t72) on columns ``p24`` and ``p72`` to minimise cost.

    Calibrated scores are step functions, so neighbouring grid points often cost almost
    the same. Among pairs within ``tie_tol`` (relative) of the lowest cost, the one with the
    longest median lead time wins: same cost, more warning.
    """
    c24, c72 = tail_thresholds(scored["p24"], n), tail_thresholds(scored["p72"], n)
    grid = []
    for t24 in c24:
        for t72 in c72:
            s = scored.assign(score=two_tier_score(scored["p24"], scored["p72"], t24, t72))
            rep, fails = evaluate(s, 1.0, horizon_hours=horizon_hours)
            if max_false_per_1k is not None and rep.false_alerts_per_1k_gpu_days > max_false_per_1k:
                continue
            lead = rep.median_lead_hours if not np.isnan(rep.median_lead_hours) else 0.0
            grid.append((float(t24), float(t72), cost.total(rep.false_alerts, fails), lead))
    if not grid:
        return float(c24.max()), float(c72.max()), np.nan
    lowest = min(g[2] for g in grid)
    near = [g for g in grid if g[2] <= lowest * (1 + tie_tol)]
    t24, t72, c, _ = max(near, key=lambda g: (g[3], -g[2]))
    return t24, t72, c
