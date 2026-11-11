import warnings

import numpy as np
import pandas as pd
import pytest

from gpufleet.clean import align_timestamps
from gpufleet.config import SimConfig
from gpufleet.features import build_features, feature_columns
from gpufleet.labels import add_labels, time_split
from gpufleet.simulator import simulate


@pytest.fixture(scope="module")
def fleet():
    cfg = SimConfig(n_nodes=4, days=8, seed=13, failure_rate_per_gpu_day=0.03,
                    maintenance_rate_per_node_day=0.3)
    res = simulate(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = build_features(res.telemetry)
    return res, feats


def test_one_row_per_gpu_hour(fleet):
    _, f = fleet
    assert not f.duplicated(["gpu_uuid", "time"]).any()
    step = f.sort_values(["gpu_uuid", "time"]).groupby("gpu_uuid")["time"].diff().dropna()
    assert (step == pd.Timedelta("1h")).all()  # gaps are filled with empty hours


def test_no_lookahead(fleet):
    """Features at time t must not change if everything from t onwards is deleted."""
    res, full = fleet
    aligned = align_timestamps(res.telemetry)
    keyed = full.set_index(["gpu_uuid", "time"])
    cols = feature_columns(full)
    start = full["time"].min()
    for hours in (30, 77, 150):
        t = start + pd.Timedelta(hours=hours)
        keep = aligned.loc[aligned["timestamp"] < t, "raw_timestamp"]
        trunc = res.telemetry[res.telemetry["timestamp"].isin(set(keep))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            part = build_features(trunc).set_index(["gpu_uuid", "time"])
        common = part.index[part.index.get_level_values("time") <= t]
        a = keyed.loc[common, cols].to_numpy(dtype=float)
        b = part.loc[common, cols].to_numpy(dtype=float)
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_rates_ignore_missing_scrapes():
    """A GPU with half its scrapes missing should show the same error rate, not half."""
    res = simulate(SimConfig(n_nodes=1, days=3, seed=4, noisy_ecc_frac=1.0,
                             failure_rate_per_gpu_day=0, maintenance_rate_per_node_day=0,
                             exporter_outage_rate_per_node_day=0, chronic_straggler_frac=0))
    tel = res.telemetry
    thinned = tel[tel["timestamp"].dt.minute % 10 < 5]  # drop every other scrape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = build_features(tel)
        b = build_features(thinned)
    ra = a.groupby("gpu_uuid")["sbe_rate_72h"].last().mean()
    rb = b.groupby("gpu_uuid")["sbe_rate_72h"].last().mean()
    assert rb == pytest.approx(ra, rel=0.35)
    assert b["coverage_24h"].iloc[-1] == pytest.approx(0.5, abs=0.05)


def test_labels(fleet):
    res, f = fleet
    end = res.telemetry["timestamp"].max().round("5min")
    lab = add_labels(f, res.failures, end)
    obs = res.failures[res.failures["observed"]]
    assert len(obs) > 0
    for _, fl in obs.iterrows():
        g = lab[lab["gpu_uuid"] == fl["gpu_uuid"]]
        assert g["time"].max() <= fl["failure_time"]  # nothing after the failure
        pos = g[g["label_24h"] == 1]
        assert ((fl["failure_time"] - pos["time"]) <= pd.Timedelta("24h")).all()
        assert ((fl["failure_time"] - pos["time"]) > pd.Timedelta(0)).all()
        near = g[(fl["failure_time"] - g["time"] <= pd.Timedelta("24h"))
                 & (g["time"] < fl["failure_time"])]
        assert (near["label_24h"] == 1).all()
        assert (g["label_72h"] >= g["label_24h"]).all()
    healthy = lab[lab["ttf_hours"].isna()]
    censored = healthy[healthy["time"] + pd.Timedelta("24h") > end]
    assert censored["label_24h"].isna().all()
    assert (healthy.loc[healthy["time"] + pd.Timedelta("72h") <= end, "label_72h"] == 0).all()


def test_split_has_embargo(fleet):
    _, f = fleet
    split = time_split(f, val_days=2, test_days=2, embargo_hours=24)
    part = split.assign(f)
    assert set(part.unique()) == {"train", "val", "test", "gap"}
    train_max = f.loc[part == "train", "time"].max()
    val_min = f.loc[part == "val", "time"].min()
    assert train_max + pd.Timedelta("24h") <= val_min
    assert f.loc[part == "val", "time"].max() < f.loc[part == "test", "time"].min()


def test_labels_with_no_failures(tmp_path):
    """A fleet with no failures (empty failures table, read back from parquet) still labels."""
    res = simulate(SimConfig(n_nodes=1, days=4, seed=1, failure_rate_per_gpu_day=0))
    res.write(tmp_path)
    failures = pd.read_parquet(tmp_path / "failures.parquet")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = build_features(res.telemetry)
    lab = add_labels(f, failures, res.telemetry["timestamp"].max().round("5min"))
    assert len(lab) == len(f)
    assert set(lab["label_24h"].dropna().unique()) == {0.0}
