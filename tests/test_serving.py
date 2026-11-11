"""Model bundle, streaming scorer and HTTP API."""

import warnings

import numpy as np
import pandas as pd
import pytest

from gpufleet.bundle import Bundle, label, train_bundle
from gpufleet.config import SimConfig
from gpufleet.features import build_features, feature_columns
from gpufleet.labels import add_labels
from gpufleet.simulator import simulate
from gpufleet.stream import StreamingScorer, replay


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    res = simulate(SimConfig(n_nodes=8, days=28, seed=5, failure_rate_per_gpu_day=0.01))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = build_features(res.telemetry)
    f = add_labels(f, res.failures, res.telemetry["timestamp"].max().round("5min"))
    b = train_bundle(f)
    path = b.save(tmp_path_factory.mktemp("bundle"))
    return Bundle.load(path), f, b


@pytest.fixture(scope="module")
def live():
    # 4 days: long enough that the scorer trims old buckets (keep_hours=80).
    res = simulate(SimConfig(n_nodes=2, days=4, seed=11, failure_rate_per_gpu_day=0.02))
    return res.telemetry


def test_bundle_round_trip(bundle):
    loaded, feats, original = bundle
    rows = feats.tail(500)
    pd.testing.assert_frame_equal(loaded.probabilities(rows), original.probabilities(rows))
    assert (loaded.t24, loaded.t72) == (original.t24, original.t72)
    p = loaded.probabilities(rows)
    assert p.min().min() >= 0 and p.max().max() <= 1
    assert set(loaded.alert_level(p)) <= {"", "urgent", "warning"}
    why = loaded.explain(rows.head(3))
    assert len(why) == 3 and all(r["contribution"] > 0 for w in why for r in w)


def test_feature_labels_are_readable():
    assert label("clock_vs_node_24h") == "lower clocks than its node peers (last 24h)"
    assert label("sbe_accel") == "corrected ECC errors accelerating"


def test_stream_matches_batch_features(bundle, live):
    loaded, _, _ = bundle
    sc = StreamingScorer(loaded)
    replay(sc, live, rounds_per_batch=3)
    assert sc.counters()["buckets"] <= 16 * 81  # old hours were trimmed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch = build_features(live)
    hour = sc.last_features["time"].max()
    a = sc.last_features.set_index("gpu_uuid")
    b = batch[batch["time"] == hour].set_index("gpu_uuid").loc[a.index]
    cols = feature_columns(batch)
    np.testing.assert_allclose(a[cols].to_numpy(float), b[cols].to_numpy(float),
                               rtol=1e-4, atol=1e-4)
    assert (a["hours_seen"] == b["hours_seen"]).all()
    assert len(sc.risk) == 16 and sc.risk["p72"].between(0, 1).all()
    assert not sc.history.empty


def test_alert_cooldown_and_warmup(bundle, live):
    loaded, _, _ = bundle
    sc = StreamingScorer(loaded, cooldown_hours=24, warmup_hours=24)
    alerts = replay(sc, live, rounds_per_batch=6)
    start = live["timestamp"].min()
    for a in alerts:
        assert a.time - start >= pd.Timedelta("24h")
    by_gpu = {}
    for a in alerts:
        if a.gpu_uuid in by_gpu:
            assert a.time - by_gpu[a.gpu_uuid] >= pd.Timedelta("24h")
        by_gpu[a.gpu_uuid] = a.time


def test_api_end_to_end(bundle, live):
    from fastapi.testclient import TestClient

    from gpufleet.api import create_app
    from gpufleet.client import telemetry_to_rows
    from gpufleet.ingest import telemetry_to_exporter_text

    loaded, _, _ = bundle
    client = TestClient(create_app(loaded))
    assert client.get("/healthz").json()["ok"]
    assert "GPU fleet health" in client.get("/").text

    rounds = live["timestamp"].dt.round("5min")
    keys = np.sort(rounds.unique())
    for k in keys[:30]:  # 2.5 hours, so at least one hour closes and gets scored
        r = client.post("/v1/telemetry", json={"rows": telemetry_to_rows(live[rounds == k])})
        assert r.status_code == 200, r.text
    risk = client.get("/v1/risk").json()
    assert len(risk) == 16 and {"p24", "p72", "level"} <= set(risk[0])
    gpu = client.get(f"/v1/gpu/{risk[0]['gpu_uuid']}").json()
    assert gpu["history"]
    assert client.get("/v1/gpu/nope").status_code == 404
    s = client.get("/v1/summary").json()
    assert s["counters"]["gpus_tracked"] == 16
    assert isinstance(client.get("/v1/stragglers").json(), list)
    m = client.get("/metrics").text
    assert 'gpufleet_failure_probability{' in m and "gpufleet_ingested_rows_total" in m

    # The same kind of data as DCGM exporter text, one node at a time.
    node = live["node_id"].astype(str).iloc[0]
    one = live[(rounds == keys[30]) & (live["node_id"].astype(str) == node)]
    text = telemetry_to_exporter_text(one)
    r = client.post("/v1/scrape", content=text,
                    params={"timestamp": one["timestamp"].iloc[0].isoformat()})
    assert r.status_code == 200, r.text
    assert client.get("/v1/summary").json()["counters"]["rows_ingested"] > 0

    assert client.post("/v1/telemetry", json={"rows": [{"gpu_uuid": "x"}]}).status_code == 422
    assert client.post("/v1/telemetry", json={"rows": []}).status_code == 422
    assert client.post("/v1/scrape", content="# nothing\n").status_code == 422
