"""Evaluation the way an operator would see it.

Row-level scores are turned into **alerts**: the first hour a GPU's score crosses the
threshold raises an alert, and that GPU stays quiet for ``cooldown_hours`` afterwards
(nobody wants the same page every hour). Then:

* an alert is **true** if the GPU fails within ``horizon_hours`` after it, otherwise **false**;
* a failure is **caught** if at least one true alert came before it;
* **lead time** is the time from the first true alert to the failure.

Headline numbers: PR-AUC (row level, threshold-free), recall of failures, false alerts per
1,000 GPU-days, and median lead time. Sudden failures (no warning signs, by design) are
reported separately so they can neither hide nor inflate model quality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

UNPREDICTABLE = ("sudden",)


def alerts(scored: pd.DataFrame, threshold: float, cooldown_hours: int = 24) -> pd.DataFrame:
    """Alerts raised by ``score >= threshold`` with a per-GPU cooldown.

    ``scored`` needs gpu_uuid, time and score. Returns one row per alert.
    """
    hot = scored.loc[scored["score"] >= threshold, ["gpu_uuid", "time"]]
    hot = hot.sort_values(["gpu_uuid", "time"])
    cooldown = pd.Timedelta(hours=cooldown_hours)
    keep = []
    for _, times in hot.groupby("gpu_uuid", sort=False)["time"]:
        last = None
        for idx, t in times.items():
            if last is None or t - last >= cooldown:
                keep.append(idx)
                last = t
    return hot.loc[keep]


@dataclass
class Report:
    threshold: float
    pr_auc: float
    failures: int
    caught: int
    recall: float
    predictable_failures: int
    predictable_caught: int
    predictable_recall: float
    alerts: int
    false_alerts: int
    gpu_days: float
    false_alerts_per_1k_gpu_days: float
    median_lead_hours: float
    precision: float

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def failure_table(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per failure in the scored period: gpu_uuid, failure_time, mode."""
    pos = scored[scored["ttf_hours"].notna()]
    ft = pos["time"] + pd.to_timedelta(pos["ttf_hours"], unit="h")
    out = pd.DataFrame({"gpu_uuid": pos["gpu_uuid"], "failure_time": ft,
                        "mode": pos["fail_mode"].astype(str)})
    return out.drop_duplicates("gpu_uuid").reset_index(drop=True)


def evaluate(
    scored: pd.DataFrame,
    threshold: float,
    horizon_hours: int = 24,
    cooldown_hours: int = 24,
    label: str | None = None,
) -> tuple[Report, pd.DataFrame]:
    """Score one period at one threshold. Returns the report and a per-failure table.

    ``scored`` is a labelled feature table (see ``gpufleet.labels``) restricted to one
    period, with a ``score`` column. Failures counted are those whose failure time falls
    inside the period's time range plus one hour.
    """
    label = label or f"label_{horizon_hours}h"
    lab = scored[scored[label].notna()]
    pr_auc = np.nan
    if lab[label].any():
        pr_auc = float(average_precision_score(lab[label], lab["score"]))

    t0, t1 = scored["time"].min(), scored["time"].max() + pd.Timedelta("1h")
    fails = failure_table(scored)
    fails = fails[(fails["failure_time"] > t0) & (fails["failure_time"] <= t1)]

    al = alerts(scored, threshold, cooldown_hours)
    al = al.merge(fails[["gpu_uuid", "failure_time"]], on="gpu_uuid", how="left")
    lead = (al["failure_time"] - al["time"]) / pd.Timedelta("1h")
    al["true"] = (lead > 0) & (lead <= horizon_hours)
    al["lead_hours"] = lead.where(al["true"])

    first = al[al["true"]].groupby("gpu_uuid")["lead_hours"].max()
    fails = fails.assign(caught=fails["gpu_uuid"].isin(first.index),
                         lead_hours=fails["gpu_uuid"].map(first))
    predictable = fails[~fails["mode"].isin(UNPREDICTABLE)]
    gpu_days = len(scored) / 24
    false_alerts = int((~al["true"]).sum())

    def ratio(a, b):
        return float(a / b) if b else float("nan")

    report = Report(
        threshold=float(threshold),
        pr_auc=pr_auc,
        failures=len(fails),
        caught=int(fails["caught"].sum()),
        recall=ratio(fails["caught"].sum(), len(fails)),
        predictable_failures=len(predictable),
        predictable_caught=int(predictable["caught"].sum()),
        predictable_recall=ratio(predictable["caught"].sum(), len(predictable)),
        alerts=len(al),
        false_alerts=false_alerts,
        gpu_days=gpu_days,
        false_alerts_per_1k_gpu_days=ratio(1000 * false_alerts, gpu_days),
        median_lead_hours=float(fails["lead_hours"].median()) if fails["caught"].any() else np.nan,
        precision=ratio(al["true"].sum(), len(al)),
    )
    return report, fails


def threshold_for_budget(
    scored: pd.DataFrame,
    max_false_per_1k: float = 5.0,
    horizon_hours: int = 24,
    cooldown_hours: int = 24,
    candidates: int = 60,
) -> float:
    """Lowest threshold whose false alerts stay within the budget on ``scored``.

    Pick this on validation data, then apply it unchanged to test.
    """
    pos_scores = scored["score"].to_numpy()
    # Alerts are rare, so search the top tail of the score distribution finely.
    levels = 1 - np.logspace(-6, -0.7, candidates)
    qs = np.unique(np.quantile(pos_scores, levels))
    best = float(np.nextafter(pos_scores.max(), np.inf))
    for thr in sorted(qs, reverse=True):
        rep, _ = evaluate(scored, thr, horizon_hours, cooldown_hours)
        if rep.false_alerts_per_1k_gpu_days <= max_false_per_1k:
            best = float(thr)
        else:
            break
    return best
