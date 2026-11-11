import numpy as np
import pandas as pd

from gpufleet.stragglers import StragglerConfig, _cusum_by_group, detect, intervals, run


def test_detects_injected_stragglers(sim):
    flags, iv, m = run(sim.telemetry, sim.stragglers)
    assert m["precision"] >= 0.8
    assert m["recall"] >= 0.8
    assert m["cause_accuracy"] >= 0.8
    assert m["intervals_detected"] >= 0.8 * m["intervals_with_busy_time"]
    assert len(iv) > 0 and (iv["end"] > iv["start"]).all()
    assert set(flags.loc[flags["straggler"], "cause"]) <= {"thermal", "clock", "off_gpu"}


def test_cusum_resets_per_gpu_and_is_capped():
    groups = np.array([0, 0, 0, 1, 1])
    z = np.array([10.0, 10.0, 10.0, 0.0, 0.0])
    out = _cusum_by_group(groups, z, k=1.0, cap=12.0)
    assert out[:3].tolist() == [9.0, 12.0, 12.0]
    assert out[3:].tolist() == [0.0, 0.0]


def _signals(slow_gpu_factor):
    ts = pd.date_range("2026-01-01", periods=24, freq="5min")
    rows = []
    for t in ts:
        for g in range(8):
            ratio = slow_gpu_factor if g == 0 else 1.0
            rows.append({"gpu_uuid": f"g{g}", "node_id": "n0", "gpu_index": g, "timestamp": t,
                         "step_vs_node": ratio, "clock_vs_node": 1.0, "temp_vs_node": 0.0,
                         "throttle_thermal": False})
    return pd.DataFrame(rows)


def test_healthy_node_has_no_stragglers():
    flags = detect(_signals(1.0), StragglerConfig())
    assert not flags["straggler"].any()


def test_slow_gpu_flagged_as_off_gpu_when_clocks_normal():
    flags = detect(_signals(1.3), StragglerConfig())
    slow = flags[flags["gpu_uuid"] == "g0"]
    assert slow["straggler"].iloc[-1]
    assert slow["cause"].iloc[-1] == "off_gpu"
    assert not flags.loc[flags["gpu_uuid"] != "g0", "straggler"].any()
    assert len(intervals(flags)) == 1
