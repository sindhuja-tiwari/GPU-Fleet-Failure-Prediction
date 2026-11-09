"""Settings for the synthetic fleet simulator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Failure modes the simulator can inject.
# Each has a precursor window (hours of warning before the fatal fault),
# a shape exponent for how fast degradation accelerates, and the Xid code
# logged when the GPU finally fails.
FAILURE_MODES: dict[str, dict] = {
    # ECC errors climb, pages get retired, then an uncorrectable (double-bit) error.
    "memory": {"precursor_hours": (48, 168), "shape": 2.0, "fatal_xid": 48},
    # Cooling degrades: temperature drifts up, clocks throttle, GPU drops off the bus.
    "thermal": {"precursor_hours": (72, 240), "shape": 1.5, "fatal_xid": 79},
    # NVLink CRC errors come in growing bursts, then the link fails.
    "nvlink": {"precursor_hours": (12, 72), "shape": 2.0, "fatal_xid": 74},
    # Power delivery: power draw gets noisy, PCIe replays rise, GPU drops off the bus.
    "power": {"precursor_hours": (2, 12), "shape": 2.0, "fatal_xid": 79},
    # No warning at all (firmware hang, GSP timeout). Kept to stay honest about limits.
    "sudden": {"precursor_hours": (0, 0), "shape": 1.0, "fatal_xid": 119},
}


@dataclass
class SimConfig:
    # Fleet shape
    n_nodes: int = 32
    gpus_per_node: int = 8
    nodes_per_rack: int = 4

    # Time
    days: float = 30.0
    step_minutes: int = 5
    start: str = "2026-01-01T00:00:00"
    seed: int = 0

    # Failures. The default rate is much higher than a real fleet (roughly 1-3%
    # per GPU per year) so a 30-day run has enough failures to learn from.
    failure_rate_per_gpu_day: float = 0.002
    mode_mix: dict[str, float] = field(
        default_factory=lambda: {
            "memory": 0.30,
            "thermal": 0.20,
            "nvlink": 0.20,
            "power": 0.15,
            "sudden": 0.15,
        }
    )
    repair_hours: tuple[float, float] = (24.0, 72.0)

    # Stragglers
    chronic_straggler_frac: float = 0.02  # GPUs that are always a bit slow
    chronic_slowdown: tuple[float, float] = (1.08, 1.25)
    transient_straggler_rate_per_gpu_day: float = 0.01  # host / network hiccups
    transient_slowdown: tuple[float, float] = (1.10, 1.50)
    transient_hours: tuple[float, float] = (0.5, 6.0)

    # Benign look-alikes, so "any ECC error" or "any Xid" is not a perfect signal
    noisy_ecc_frac: float = 0.03  # healthy GPUs with a steady trickle of corrected errors
    noisy_nvlink_frac: float = 0.03  # healthy links with occasional CRC errors
    benign_xid_rate_per_gpu_day: float = 0.05  # application faults (Xid 13, 31, 43)

    # Collection
    scrape_drop_prob: float = 0.001  # chance a single sample is missing

    @property
    def n_gpus(self) -> int:
        return self.n_nodes * self.gpus_per_node

    @property
    def n_steps(self) -> int:
        return int(round(self.days * 24 * 60 / self.step_minutes))

    @property
    def steps_per_hour(self) -> float:
        return 60.0 / self.step_minutes

    def validate(self) -> None:
        if self.n_nodes <= 0 or self.gpus_per_node <= 0 or self.nodes_per_rack <= 0:
            raise ValueError("fleet dimensions must be positive")
        if self.days <= 0 or self.step_minutes <= 0:
            raise ValueError("days and step_minutes must be positive")
        unknown = set(self.mode_mix) - set(FAILURE_MODES)
        if unknown:
            raise ValueError(f"unknown failure modes: {sorted(unknown)}")
        if any(w < 0 for w in self.mode_mix.values()) or sum(self.mode_mix.values()) <= 0:
            raise ValueError("mode_mix weights must be non-negative and not all zero")

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> SimConfig:
        raw = json.loads(Path(path).read_text())
        for key in ("repair_hours", "chronic_slowdown", "transient_slowdown", "transient_hours"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)
