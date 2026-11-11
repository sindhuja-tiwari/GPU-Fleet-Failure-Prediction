import numpy as np
import pandas as pd

from gpufleet.policy import (
    Calibrator,
    CostModel,
    calibration_report,
    pick_two_tier,
    rolling_folds,
    two_tier_score,
)
from gpufleet.seqmodel import SEQ_COLUMNS, windows


def test_cost_rewards_early_warning():
    cost = CostModel()
    early = pd.DataFrame({"caught": [True], "lead_hours": [30.0]})
    late = pd.DataFrame({"caught": [True], "lead_hours": [1.0]})
    missed = pd.DataFrame({"caught": [False], "lead_hours": [np.nan]})
    assert cost.total(0, early) == cost.caught_base
    assert cost.total(0, early) < cost.total(0, late) < cost.total(0, missed)
    assert cost.total(3, early) == cost.caught_base + 3


def test_rolling_folds_respect_embargo():
    dev = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=30 * 24, freq="h")})
    folds = list(rolling_folds(dev, n_folds=4, fold_days=5, embargo_hours=72))
    assert len(folds) == 4
    scored = []
    for tr, sc in folds:
        gap = dev.loc[sc, "time"].min() - dev.loc[tr, "time"].max()
        assert gap > pd.Timedelta("72h")
        scored.append(set(sc))
    assert all(not (a & b) for i, a in enumerate(scored) for b in scored[i + 1:])


def test_calibrator_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    s = rng.random(5000)
    y = (rng.random(5000) < s ** 3).astype(float)
    cal = Calibrator().fit(s, y)
    p = cal(np.linspace(-1, 2, 50))
    assert (np.diff(p) >= 0).all() and p.min() >= 0 and p.max() <= 1
    rep = calibration_report(cal(s), y)
    assert rep["ece"] < 0.05 and 0 < rep["brier"] < 0.25


def test_two_tier_score_fires_on_either_tier():
    s = two_tier_score(np.array([0.6, 0.1, 0.1]), np.array([0.1, 0.5, 0.1]), 0.5, 0.4)
    assert (s >= 1).tolist() == [True, True, False]


def test_pick_two_tier_stays_within_budget():
    rng = np.random.default_rng(1)
    rows = []
    for g in range(40):
        fails = g < 8
        for h in range(200):
            t = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=h)
            ttf = 199 - h + 0.5 if fails else np.nan
            near = fails and ttf <= 72
            rows.append({"gpu_uuid": f"g{g}", "time": t, "ttf_hours": ttf,
                         "fail_mode": "memory" if fails else None,
                         "label_72h": float(near), "label_24h": float(fails and ttf <= 24),
                         "p24": rng.random() * 0.2 + (0.7 if fails and ttf <= 24 else 0),
                         "p72": rng.random() * 0.2 + (0.6 if near else 0)})
    scored = pd.DataFrame(rows)
    t24, t72, cost = pick_two_tier(scored, CostModel(), 72, n=8, max_false_per_1k=50)
    assert 0 < t24 <= 1 and 0 < t72 <= 1 and np.isfinite(cost)


def test_sequence_windows_do_not_cross_gpus():
    n = 5
    df = pd.DataFrame({"gpu_uuid": ["a"] * n + ["b"] * n,
                       "time": list(pd.date_range("2026-01-01", periods=n, freq="h")) * 2})
    for c in SEQ_COLUMNS:
        df[c] = 0.0
    df["seq_coverage"] = [1.0] * n + [2.0] * n
    w = windows(df, lookback=3).reshape(2 * n, 3, len(SEQ_COLUMNS))
    cov = w[:, :, SEQ_COLUMNS.index("seq_coverage")]
    assert cov[0].tolist() == [0, 0, 1]  # first hour of GPU a: no history yet
    assert cov[n].tolist() == [0, 0, 2]  # first hour of GPU b: nothing from GPU a
    assert cov[n + 2].tolist() == [2, 2, 2]
