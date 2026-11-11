import pandas as pd

from gpufleet.clean import align_timestamps, counter_increments, find_gaps, quality_report
from gpufleet.config import SimConfig
from gpufleet.simulator import simulate


def _frame(values, minutes, uuid="GPU-a"):
    ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(minutes, unit="min")
    return pd.DataFrame(
        {"timestamp": ts, "gpu_uuid": uuid, "node_id": "node-000", "gpu_index": 0,
         "ecc_sbe_total": values}
    )


def test_counter_increments_handle_reset():
    df = _frame([0, 2, 5, 1, 4], [0, 5, 10, 60, 65])  # reboot between 10 and 60 min
    out = counter_increments(df, ["ecc_sbe_total"])
    assert out["ecc_sbe_total_inc"].tolist() == [0, 2, 3, 1, 3]
    assert out["counter_reset"].tolist() == [False, False, False, True, False]


def test_counter_increments_are_per_gpu():
    df = pd.concat([_frame([0, 1, 2], [0, 5, 10], "GPU-a"), _frame([7, 7, 9], [0, 5, 10], "GPU-b")])
    out = counter_increments(df.reset_index(drop=True), ["ecc_sbe_total"])
    assert out["ecc_sbe_total_inc"].tolist() == [0, 1, 1, 0, 0, 2]


def test_align_snaps_skewed_scrapes():
    df = _frame([0, 1, 2], [0, 5, 10])
    df["timestamp"] += pd.to_timedelta([-17, 4, 21], unit="s")
    out = align_timestamps(df)
    expected = pd.Timestamp("2026-01-01") + pd.to_timedelta([0, 5, 10], unit="min")
    assert out["timestamp"].tolist() == list(expected)
    assert (out["raw_timestamp"] != out["timestamp"]).all()


def test_align_keeps_latest_duplicate():
    df = _frame([1, 2], [0, 0])
    df["timestamp"] += pd.to_timedelta([-10, 10], unit="s")
    out = align_timestamps(df)
    assert len(out) == 1 and out["ecc_sbe_total"].iloc[0] == 2


def test_find_gaps():
    df = _frame([0, 1, 2, 3], [0, 5, 30, 35])  # 4 scrapes missing between 5 and 30
    gaps = find_gaps(df)
    assert len(gaps) == 1 and gaps["missing"].iloc[0] == 4


def test_simulated_maintenance_shows_up_as_resets_and_gaps():
    cfg = SimConfig(n_nodes=4, days=10, seed=11, maintenance_rate_per_node_day=0.5,
                    noisy_ecc_frac=1.0, failure_rate_per_gpu_day=0)
    res = simulate(cfg)
    resets = res.maintenance[res.maintenance["resets_counters"]]
    assert len(resets) > 0
    report = quality_report(res.telemetry)
    assert report["counter_resets"] > 0
    assert report["gaps_over_1h"] > 0
    assert report["completeness"] < 1

    # Every detected counter reset sits right after a maintenance window on that node.
    inc = counter_increments(align_timestamps(res.telemetry))
    for _, row in inc[inc["counter_reset"]].iterrows():
        node = resets[resets["node_id"] == row["node_id"]]
        assert ((node["end"] <= row["timestamp"]) &
                (row["timestamp"] - node["end"] < pd.Timedelta("1h"))).any()


def test_drained_node_runs_no_jobs():
    cfg = SimConfig(n_nodes=4, days=10, seed=5, maintenance_rate_per_node_day=0.5,
                    maintenance_mix={"drain": 1.0})
    res = simulate(cfg)
    tel = align_timestamps(res.telemetry)
    drains = res.maintenance[res.maintenance["kind"] == "drain"]
    assert len(drains) > 0
    for _, d in drains.iterrows():
        rows = tel[(tel["node_id"] == d["node_id"]) & (tel["timestamp"] >= d["start"])
                   & (tel["timestamp"] < d["end"])]
        assert len(rows) > 0  # telemetry keeps flowing
        assert rows["job_id"].isna().all()
