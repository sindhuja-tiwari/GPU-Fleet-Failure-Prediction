import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gpufleet.clean import align_timestamps
from gpufleet.config import SimConfig
from gpufleet.ingest import (
    load_exporter_scrapes,
    load_prometheus_range,
    parse_exporter_text,
    telemetry_to_exporter_text,
    telemetry_to_prometheus_range,
    to_telemetry,
)
from gpufleet.schema import validate_telemetry
from gpufleet.simulator import simulate

FIXTURE = Path(__file__).parent / "fixtures" / "dcgm_exporter_sample.prom"


def test_parse_real_exporter_output():
    long = parse_exporter_text(FIXTURE.read_text(), "2026-03-01T12:00:00")
    assert set(long["node_id"]) == {"gpu-node-17"}
    assert set(long["gpu_index"]) == {0, 1}
    assert "DCGM_FI_DEV_FB_USED" in set(long["metric"])  # unknown fields are kept in long form
    assert set(long["job_id"].dropna()) == {"llm-pretrain-7b-worker-3"}


def test_real_exporter_output_maps_to_schema():
    long = parse_exporter_text(FIXTURE.read_text(), "2026-03-01T12:00:00")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tel = to_telemetry(long, rack_map={"gpu-node-17": "rack-a3"})
    validate_telemetry(tel)
    missing = str(caught[0].message)
    assert "ecc_sbe_total" in missing and "nvlink_crc_total" in missing

    g0, g1 = tel.iloc[0], tel.iloc[1]
    assert g0["gpu_util"] == 97 and g0["power_w"] == pytest.approx(612.48, rel=1e-6)
    assert g0["job_id"] == "llm-pretrain-7b-worker-3" and pd.isna(g1["job_id"])
    assert g0["throttle_thermal"] and g0["throttle_power"]  # 36 = 0x20 + 0x04
    assert not g1["throttle_thermal"] and not g1["throttle_power"]  # 1 = GPU idle
    assert g1["retired_pages_total"] == 4  # correctable + uncorrectable remapped rows
    assert g1["xid"] == 63 and g0["xid"] == 0
    assert tel["rack_id"].astype(str).tolist() == ["rack-a3", "rack-a3"]
    assert tel["temp_mem_c"].isna().all()  # not published by this exporter config


def test_missing_required_field_is_an_error():
    long = parse_exporter_text(FIXTURE.read_text(), "2026-03-01T12:00:00")
    with pytest.raises(ValueError, match="power_w"):
        to_telemetry(long[long["metric"] != "DCGM_FI_DEV_POWER_USAGE"])


def test_xid_last_value_becomes_events():
    base = {"node_id": "n", "gpu_index": 0, "gpu_uuid": "GPU-x", "job_id": None}
    rows = []
    for i, (xid, util) in enumerate([(0, 50), (43, 50), (43, 50), (79, 50)]):
        ts = pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=5 * i)
        for metric, value in [
            ("DCGM_FI_DEV_XID_ERRORS", xid), ("DCGM_FI_DEV_GPU_UTIL", util),
            ("DCGM_FI_DEV_SM_CLOCK", 1900), ("DCGM_FI_DEV_POWER_USAGE", 400),
            ("DCGM_FI_DEV_GPU_TEMP", 60),
        ]:
            rows.append({"timestamp": ts, "metric": metric, "value": value, **base})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tel = to_telemetry(pd.DataFrame(rows))
    assert tel["xid"].tolist() == [0, 43, 0, 79]


def test_scrape_files_round_trip(tmp_path):
    res = simulate(SimConfig(n_nodes=1, days=0.25, seed=2))
    tel = res.telemetry
    for ts, scrape in tel.groupby("timestamp"):
        name = ts.strftime("%Y-%m-%dT%H-%M-%S") + ".prom"
        (tmp_path / name).write_text(telemetry_to_exporter_text(scrape))
    long = load_exporter_scrapes(sorted(tmp_path.glob("*.prom")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = to_telemetry(long, rack_map={"node-000": "rack-00"})
    validate_telemetry(back)
    assert len(back) == len(tel)
    # File names keep whole seconds only.
    assert (back["timestamp"] - tel["timestamp"].dt.floor("s")).abs().max() == pd.Timedelta(0)


def test_prometheus_range_round_trip():
    cfg = SimConfig(n_nodes=2, days=3, seed=9, failure_rate_per_gpu_day=0.05)
    sim = simulate(cfg).telemetry
    payload = telemetry_to_prometheus_range(sim)
    back = to_telemetry(load_prometheus_range(payload),
                        rack_map=dict(zip(sim["node_id"].astype(str), sim["rack_id"].astype(str),
                                          strict=False)))
    validate_telemetry(back)

    a = align_timestamps(sim).set_index(["gpu_uuid", "timestamp"]).sort_index()
    b = align_timestamps(back).set_index(["gpu_uuid", "timestamp"]).sort_index()
    a.index = a.index.set_levels(a.index.levels[0].astype(str), level=0)
    b.index = b.index.set_levels(b.index.levels[0].astype(str), level=0)
    assert a.index.equals(b.index)

    for col in ["gpu_util", "mem_util", "sm_clock_mhz", "power_w", "temp_gpu_c", "temp_mem_c",
                "ecc_sbe_total", "ecc_dbe_total", "retired_pages_total", "pcie_replay_total",
                "nvlink_crc_total", "throttle_thermal", "throttle_power"]:
        np.testing.assert_allclose(a[col].astype(float), b[col].astype(float), rtol=1e-4,
                                   err_msg=col)
    assert (a["job_id"].fillna("") == b["job_id"].fillna("")).all()
    assert (a["rack_id"].astype(str) == b["rack_id"].astype(str)).all()
    # Repeated identical Xids collapse into one event; every recovered event is real.
    got = b["xid"] != 0
    assert (b.loc[got, "xid"] == a.loc[got, "xid"]).all()
    assert got.sum() >= 0.8 * (a["xid"] != 0).sum()
    assert b["step_compute_ms"].isna().all()  # not a DCGM field
