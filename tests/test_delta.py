import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from gpufleet.delta import (
    bh_qvalues,
    cascade_table,
    daily_counts,
    day_table,
    safe_load,
)


def _ev(xid: int) -> dict:
    return {"TemplateID": 1000 + xid, "Tag": "", "Regex": "", "Category": "", "Group": ""}


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_safe_load_refuses_code(tmp_path):
    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps(_Evil()))
    with pytest.raises(pickle.UnpicklingError):
        safe_load(p)


def test_safe_load_reads_plain_containers(tmp_path):
    d = defaultdict(list)
    d[1031] = [0.5, 1.0]
    p = tmp_path / "ok.pkl"
    p.write_bytes(pickle.dumps({"a": [1, 2], "b": d}))
    out = safe_load(p)
    assert out["a"] == [1, 2] and out["b"][1031] == [0.5, 1.0]


def test_daily_counts_fills_gaps_and_labels_nvrm():
    data = {
        "20240101": [_ev(31), _ev(31), {"TemplateID": 368}],
        "20240104": [_ev(79)],
    }
    c = daily_counts(data)
    assert len(c) == 4
    assert list(c.columns) == ["31", "79", "nvrm"]
    assert c.loc["2024-01-01", "31"] == 2
    assert c.loc["2024-01-02"].sum() == 0


def test_day_table_uses_only_past_days():
    days = pd.date_range("2024-01-01", periods=60, freq="D")
    counts = pd.DataFrame(0, index=days, columns=["31", "79"])
    counts["31"] = 1  # present every day, so it passes min_days
    counts.loc[days[40], "79"] = 1
    counts.loc[days[40], "31"] = 100  # same-day burst must not leak into day 40's features
    X, y, pcols, _ = day_table(counts, {79}, lookback=7, horizon=1, level_days=10)
    assert pcols == ["xid31_last7d"]
    assert y.loc[days[40]] == 1 and y.sum() == 1
    assert X.loc[days[40], "xid31_last7d"] == pytest.approx(np.log1p(7))
    assert X.loc[days[41], "xid31_last7d"] == pytest.approx(np.log1p(106))


def test_day_table_horizon_marks_days_before_event():
    days = pd.date_range("2024-01-01", periods=60, freq="D")
    counts = pd.DataFrame({"31": 1, "79": 0}, index=days)
    counts.loc[days[45], "79"] = 1
    _, y, _, _ = day_table(counts, {79}, horizon=3, level_days=10)
    assert list(y[y == 1].index) == list(days[43:46])
    assert y.index.max() == days[-3]  # last days lack a full label window


def test_cascade_table():
    prop = {1031: {-1: [0] * 8, 1094: [0.1], 1031: [0.2]}}
    t = cascade_table(prop)
    row = t.iloc[0]
    assert row["source_xid"] == "31" and row["events"] == 10
    assert row["followed_within_5s"] == pytest.approx(0.2)
    assert row["critical_within_5s"] == pytest.approx(0.1)
    assert row["critical_followers"] == "94:1"


def test_bh_qvalues():
    q = bh_qvalues(np.array([0.01, 0.04, 0.03, 0.5]))
    assert q == pytest.approx([0.04, 0.0533333, 0.0533333, 0.5], rel=1e-4)
