import numpy as np
import pandas as pd
import pytest

from gpufleet.evaluate import alerts, evaluate, threshold_for_budget

T0 = pd.Timestamp("2026-01-01")


def _gpu(uuid, hours, scores, fail_after=None, mode="memory"):
    t = T0 + pd.to_timedelta(np.arange(hours), unit="h")
    ttf = None if fail_after is None else fail_after - np.arange(hours)
    df = pd.DataFrame({"gpu_uuid": uuid, "time": t, "score": scores})
    df["ttf_hours"] = ttf if ttf is not None else np.nan
    df["fail_mode"] = mode if fail_after is not None else None
    df["label_24h"] = ((df["ttf_hours"] > 0) & (df["ttf_hours"] <= 24)).astype(float)
    return df


def test_cooldown_suppresses_repeat_alerts():
    df = _gpu("a", 72, np.ones(72))
    al = alerts(df, 0.5, cooldown_hours=24)
    assert al["time"].tolist() == [T0, T0 + pd.Timedelta("24h"), T0 + pd.Timedelta("48h")]


def test_caught_failure_and_lead_time():
    scores = np.zeros(100)
    scores[80:] = 1.0  # starts alerting 20h before failure at hour 100
    fail = _gpu("f", 100, scores, fail_after=100)
    healthy = _gpu("h", 100, np.zeros(100))
    rep, fails = evaluate(pd.concat([fail, healthy], ignore_index=True), 0.5)
    assert rep.caught == 1 and rep.false_alerts == 0
    assert rep.median_lead_hours == pytest.approx(20)
    assert rep.pr_auc > 0.8


def test_early_alert_counts_as_false():
    scores = np.zeros(100)
    scores[50] = 1.0  # 50h before failure: outside the 24h horizon
    df = _gpu("f", 100, scores, fail_after=100)
    rep, _ = evaluate(df, 0.5)
    assert rep.caught == 0 and rep.false_alerts == 1


def test_sudden_failures_are_reported_separately():
    df = pd.concat([
        _gpu("m", 50, np.r_[np.zeros(40), np.ones(10)], fail_after=50, mode="memory"),
        _gpu("s", 50, np.zeros(50), fail_after=50, mode="sudden"),
    ], ignore_index=True)
    rep, _ = evaluate(df, 0.5)
    assert rep.failures == 2 and rep.caught == 1
    assert rep.predictable_failures == 1 and rep.predictable_recall == 1.0


def test_threshold_respects_budget():
    rng = np.random.default_rng(0)
    gpus = [_gpu(f"h{i}", 240, rng.random(240) * 0.5) for i in range(40)]
    gpus[0].loc[100, "score"] = 0.9  # one noisy healthy GPU
    df = pd.concat(gpus, ignore_index=True)
    thr = threshold_for_budget(df, max_false_per_1k=5.0)
    rep, _ = evaluate(df, thr)
    assert rep.false_alerts_per_1k_gpu_days <= 5.0
