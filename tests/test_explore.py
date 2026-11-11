import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from gpufleet.config import SimConfig  # noqa: E402
from gpufleet.explore import (  # noqa: E402
    Dataset,
    epochs,
    fig_collection,
    fig_failure_signatures,
    fig_lookalikes,
    fig_stragglers,
    hourly_signals,
    save_all_figures,
    scrape_signals,
    warning_time,
)
from gpufleet.simulator import simulate  # noqa: E402


@pytest.fixture(scope="module")
def explored():
    mix = {"memory": 0.4, "thermal": 0.3, "nvlink": 0.3}
    res = simulate(SimConfig(n_nodes=8, days=12, seed=21, failure_rate_per_gpu_day=0.02,
                             mode_mix=mix, chronic_straggler_frac=0.1))
    ds = Dataset.from_result(res)
    sig = scrape_signals(ds.telemetry, ds.step)
    return res, ds, sig, hourly_signals(sig)


def test_epochs_only_cover_time_before_failure(explored):
    _, ds, _, hourly = explored
    ep = epochs(hourly, ds.failures, "sbe_per_h", "memory", hours=48)
    assert ep.shape[1] > 0
    assert ep.index.max() < 0 and ep.index.min() > -48


def test_predictable_modes_warn_before_failure(explored):
    _, ds, _, hourly = explored
    wt = warning_time(hourly, ds.failures)
    predictable = wt[wt["mode"].isin(["memory", "thermal", "nvlink"])]
    assert len(predictable) > 0
    assert predictable["warning_hours"].notna().mean() >= 0.8
    assert (predictable["warning_hours"].dropna() > 0).all()


def test_figures_render(explored):
    _, ds, sig, hourly = explored
    for fig in (
        fig_failure_signatures(hourly, ds.failures),
        fig_lookalikes(hourly, sig, ds.failures),
        fig_stragglers(sig, ds.stragglers),
        fig_collection(ds),
    ):
        assert fig.axes


def test_save_all_figures(explored, tmp_path):
    res = explored[0]
    res.write(tmp_path / "data")
    paths = save_all_figures(tmp_path / "data", tmp_path / "figs")
    assert len(paths) == 4 and all(p.stat().st_size > 10_000 for p in paths)
