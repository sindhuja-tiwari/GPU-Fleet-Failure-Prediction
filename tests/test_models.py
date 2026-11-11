import warnings

import pytest

from gpufleet.config import SimConfig
from gpufleet.experiment import run_baselines
from gpufleet.features import build_features
from gpufleet.labels import add_labels
from gpufleet.models import RULES, RulesBaseline
from gpufleet.simulator import simulate


@pytest.fixture(scope="module")
def labelled():
    cfg = SimConfig(n_nodes=8, days=24, seed=3, failure_rate_per_gpu_day=0.01,
                    mode_mix={"memory": 0.4, "thermal": 0.3, "nvlink": 0.3})
    res = simulate(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = build_features(res.telemetry)
    return add_labels(f, res.failures, res.telemetry["timestamp"].max().round("5min"))


def test_rules_reference_real_features(labelled):
    for col, _ in RULES.values():
        assert col in labelled.columns
    assert RulesBaseline().score(labelled).max() >= 1


def test_baselines_end_to_end(labelled, tmp_path):
    res = run_baselines(labelled, horizons=(24,), val_days=5, test_days=6)
    r = res.results.set_index("model")
    assert set(r.index) == {"rules", "logistic", "lightgbm"}
    assert (r.loc["lightgbm", "false_alerts_per_1k_gpu_days"]
            <= r.loc["rules", "false_alerts_per_1k_gpu_days"])
    assert r.loc["lightgbm", "predictable_recall"] >= 0.5
    out = res.write(tmp_path / "rep")
    assert (out / "report.md").read_text().count("|") > 20
