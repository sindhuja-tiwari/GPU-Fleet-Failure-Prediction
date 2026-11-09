import pytest

from gpufleet.config import SimConfig
from gpufleet.simulator import simulate


@pytest.fixture(scope="session")
def small_cfg() -> SimConfig:
    # High failure and straggler rates so a small, fast run still has every case.
    return SimConfig(
        n_nodes=8,
        days=10,
        seed=7,
        failure_rate_per_gpu_day=0.01,
        chronic_straggler_frac=0.1,
    )


@pytest.fixture(scope="session")
def sim(small_cfg):
    return simulate(small_cfg)
