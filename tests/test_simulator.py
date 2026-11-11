import numpy as np
import pandas as pd
import pytest

from gpufleet.clean import align_timestamps, counter_increments
from gpufleet.config import FAILURE_MODES, SimConfig
from gpufleet.simulator import simulate


def test_same_seed_gives_same_data(small_cfg):
    a = simulate(small_cfg)
    b = simulate(small_cfg)
    pd.testing.assert_frame_equal(a.telemetry, b.telemetry)
    pd.testing.assert_frame_equal(a.failures, b.failures)


def test_different_seed_gives_different_data(small_cfg):
    a = simulate(small_cfg)
    b = simulate(SimConfig(**{**small_cfg.__dict__, "seed": small_cfg.seed + 1}))
    assert not a.telemetry["power_w"].equals(b.telemetry["power_w"])


def test_row_count_and_time_grid(sim, aligned, small_cfg):
    tel = sim.telemetry
    full = small_cfg.n_steps * small_cfg.n_gpus
    assert 0.8 * full < len(tel) <= full  # repairs, maintenance and lost scrapes remove rows
    assert len(aligned) == len(tel)  # skew never pushes two scrapes onto one step
    step = pd.Timedelta(minutes=small_cfg.step_minutes)
    start = pd.Timestamp(small_cfg.start)
    assert ((aligned["timestamp"] - start) % step == pd.Timedelta(0)).all()


def test_clock_skew_is_per_node_and_bounded(aligned, small_cfg):
    offset = (aligned["raw_timestamp"] - aligned["timestamp"]).dt.total_seconds()
    assert offset.abs().max() < 4 * (small_cfg.clock_skew_s + small_cfg.scrape_jitter_s)
    assert offset.abs().max() > 1  # skew is actually there
    # GPUs on one node are scraped together, so they share a raw timestamp.
    per_scrape = aligned.groupby(["node_id", "timestamp"], observed=True)["raw_timestamp"]
    assert (per_scrape.nunique() == 1).all()


def test_every_observed_failure_ends_with_its_fatal_xid(sim, aligned):
    tel = aligned
    observed = sim.failures[sim.failures["observed"]]
    assert len(observed) > 0
    last = tel.sort_values("timestamp").groupby("gpu_uuid", observed=True).tail(1)
    last = last.set_index("gpu_uuid")
    for _, f in observed.iterrows():
        row = last.loc[f["gpu_uuid"]]
        assert row["timestamp"] == f["failure_time"]
        assert row["xid"] == FAILURE_MODES[f["mode"]]["fatal_xid"]


def test_failed_slot_is_offline_until_repaired(sim, aligned):
    tel = aligned
    observed = sim.failures[sim.failures["observed"]]
    for _, f in observed.iterrows():
        slot = tel[(tel["node_id"] == f["node_id"]) & (tel["gpu_index"] == f["gpu_index"])]
        ts = slot["timestamp"]
        assert slot[(ts > f["failure_time"]) & (ts < f["repaired_time"])].empty
        after = slot[slot["timestamp"] >= f["repaired_time"]]
        if not after.empty:
            assert f["gpu_uuid"] not in set(after["gpu_uuid"])  # replaced by a new GPU
            assert after["ecc_sbe_total"].iloc[0] <= 1  # counters start fresh


def test_onset_comes_before_failure(sim):
    f = sim.failures
    assert (f["onset_time"] <= f["failure_time"]).all()
    predictable = f[f["mode"] != "sudden"]
    assert (predictable["failure_time"] - predictable["onset_time"] >= pd.Timedelta(hours=1)).all()


def _rate_change(inc, failure, column):
    """Mean per-scrape increment before onset vs. in the last 12 hours before failure."""
    g = inc[inc["gpu_uuid"] == failure["gpu_uuid"]].set_index("timestamp")[column + "_inc"]
    end = failure["failure_time"]
    before = g[: failure["onset_time"]]
    late = g[end - pd.Timedelta(hours=12) : end - pd.Timedelta(minutes=5)]
    return (before.mean() if len(before) else 0.0), late.mean()


@pytest.mark.parametrize(
    "mode, column",
    [("memory", "ecc_sbe_total"), ("nvlink", "nvlink_crc_total"), ("power", "pcie_replay_total")],
)
def test_degradation_raises_error_rates(mode, column):
    cfg = SimConfig(n_nodes=8, days=20, seed=3, failure_rate_per_gpu_day=0.01,
                    mode_mix={mode: 1.0}, noisy_ecc_frac=0, noisy_nvlink_frac=0)
    res = simulate(cfg)
    observed = res.failures[res.failures["observed"]]
    assert len(observed) >= 3
    inc = counter_increments(align_timestamps(res.telemetry), [column])
    rates = [_rate_change(inc, f, column) for _, f in observed.iterrows()]
    assert np.mean([late > base for base, late in rates]) >= 0.8


def test_thermal_degradation_raises_temperature():
    cfg = SimConfig(n_nodes=8, days=20, seed=4, failure_rate_per_gpu_day=0.01,
                    mode_mix={"thermal": 1.0})
    res = simulate(cfg)
    tel = res.telemetry
    observed = res.failures[res.failures["observed"]]
    assert len(observed) >= 3
    for _, f in observed.iterrows():
        g = tel[(tel["gpu_uuid"] == f["gpu_uuid"]) & (tel["gpu_util"] > 50)].set_index("timestamp")
        late = g[f["failure_time"] - pd.Timedelta(hours=12) :]
        node = tel[(tel["node_id"] == f["node_id"]) & (tel["gpu_util"] > 50)
                   & (tel["gpu_uuid"] != f["gpu_uuid"])]
        if len(late) < 10:
            continue  # node was idle near the end
        assert late["temp_gpu_c"].mean() > node["temp_gpu_c"].mean() + 5


def test_chronic_stragglers_are_slower_than_peers(sim):
    tel = sim.telemetry[sim.telemetry["step_compute_ms"].notna()].copy()
    tel["rel"] = tel["step_compute_ms"] / tel.groupby(["timestamp", "node_id"], observed=True)[
        "step_compute_ms"
    ].transform("median")
    chronic = sim.stragglers[sim.stragglers["kind"] == "chronic"]
    assert len(chronic) > 0
    rel = tel.groupby("gpu_uuid", observed=True)["rel"].mean()
    # GPUs that also had a transient or thermal slowdown compound, so skip them here.
    other = set(sim.stragglers.loc[sim.stragglers["kind"] != "chronic", "gpu_uuid"])
    for _, s in chronic.iterrows():
        if s["gpu_uuid"] in rel.index and s["gpu_uuid"] not in other:
            assert rel[s["gpu_uuid"]] == pytest.approx(s["slowdown"], rel=0.03)
    healthy = rel.drop(sim.stragglers["gpu_uuid"], errors="ignore")
    assert healthy.between(0.97, 1.03).mean() > 0.95


def test_benign_lookalikes_exist(sim):
    """Healthy GPUs must sometimes show app Xids, so Xid != failure."""
    failed = set(sim.failures["gpu_uuid"])
    healthy = sim.telemetry[~sim.telemetry["gpu_uuid"].isin(failed)]
    assert healthy["xid"].isin([13, 31, 43]).any()


def test_write_round_trip(sim, tmp_path):
    out = sim.write(tmp_path / "run")
    for name in ("telemetry", "failures", "stragglers", "inventory"):
        assert (out / f"{name}.parquet").exists()
    back = pd.read_parquet(out / "telemetry.parquet")
    assert len(back) == len(sim.telemetry)
    assert SimConfig.from_json(out / "config.json") == sim.config


def test_bad_config_is_rejected():
    with pytest.raises(ValueError, match="unknown failure modes"):
        simulate(SimConfig(mode_mix={"cosmic_rays": 1.0}))
