"""Synthetic GPU fleet telemetry with injected failures and stragglers.

How it works, in three passes:

1. **Lifecycles.** Every GPU slot gets a timeline of physical GPUs. Each GPU runs until a
   failure onset is drawn, degrades over a mode-specific precursor window, emits a fatal
   Xid, sits offline for a repair window, and is replaced by a new GPU (new ``gpu_uuid``).
2. **Workload.** Each node alternates between idle and running a job. A job has its own
   utilisation level and step time, and crashes when one of its GPUs fails.
3. **Metrics.** All metrics are generated at once as (time x GPU) arrays from the workload,
   the degradation progress of each GPU, and a few benign look-alikes (noisy ECC counters,
   noisy NVLinks, application Xids) so that no single signal gives the answer away.

Ground truth (failure onsets, straggler intervals) is written to separate tables and never
appears in the telemetry itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FAILURE_MODES, SimConfig
from .schema import COLUMN_NAMES, FAILURE_COLUMNS, STRAGGLER_COLUMNS

MODE_NAMES = ["none", *FAILURE_MODES]  # index 0 = healthy
MODE_CODE = {m: i for i, m in enumerate(MODE_NAMES)}

MAX_SM_CLOCK = 1980.0
IDLE_SM_CLOCK = 345.0
POWER_CAP_W = 700.0
# Graphics exception, MMU fault, app stopped: usually caused by user code, not hardware.
BENIGN_XIDS = np.array([13, 31, 43])
XID_ROW_REMAP = 63


@dataclass
class Life:
    """One physical GPU in one slot, from install to failure (or end of window)."""

    slot: int
    uuid: str
    start: int  # first step online
    end: int  # first step no longer online (exclusive)
    mode: str  # "none" if it never fails inside the window
    onset: float = np.inf  # step at which degradation starts
    fail_step: int | None = None  # last online step, carries the fatal Xid
    repaired_step: int | None = None  # step the replacement comes online
    precursor_steps: float = 0.0


@dataclass
class SimResult:
    telemetry: pd.DataFrame
    failures: pd.DataFrame
    stragglers: pd.DataFrame
    inventory: pd.DataFrame
    config: SimConfig

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.telemetry.to_parquet(out / "telemetry.parquet", index=False)
        self.failures.to_parquet(out / "failures.parquet", index=False)
        self.stragglers.to_parquet(out / "stragglers.parquet", index=False)
        self.inventory.to_parquet(out / "inventory.parquet", index=False)
        self.config.to_json(out / "config.json")
        return out


def _uuid(rng: np.random.Generator) -> str:
    h = rng.integers(0, 16, size=32)
    s = "".join("0123456789abcdef"[i] for i in h)
    return f"GPU-{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}"


def _sample_lives(cfg: SimConfig, rng: np.random.Generator) -> list[Life]:
    T = cfg.n_steps
    sph = cfg.steps_per_hour
    hazard_per_step = cfg.failure_rate_per_gpu_day / (24 * sph)
    modes = list(cfg.mode_mix)
    weights = np.array([cfg.mode_mix[m] for m in modes], dtype=float)
    weights /= weights.sum()

    lives: list[Life] = []
    for slot in range(cfg.n_gpus):
        t = 0
        while t < T:
            uuid = _uuid(rng)
            onset = t + rng.exponential(1 / hazard_per_step) if hazard_per_step > 0 else np.inf
            if onset >= T:
                lives.append(Life(slot, uuid, t, T, "none"))
                break
            mode = modes[rng.choice(len(modes), p=weights)]
            lo, hi = FAILURE_MODES[mode]["precursor_hours"]
            dur = rng.uniform(lo, hi) * sph
            fail_step = int(np.ceil(onset + dur))
            if fail_step >= T:
                # Degradation has started but the GPU outlives the window (censored).
                lives.append(Life(slot, uuid, t, T, mode, onset, None, None, dur))
                break
            repair = fail_step + 1 + int(rng.uniform(*cfg.repair_hours) * sph)
            lives.append(
                Life(slot, uuid, t, fail_step + 1, mode, onset, fail_step, repair, dur)
            )
            t = repair
    return lives


def _workload(cfg: SimConfig, rng: np.random.Generator, node_fail: np.ndarray):
    """Return busy (T,N) bool, job index (T,N) int (-1 idle), and per-job parameters."""
    T, N = cfg.n_steps, cfg.n_nodes
    sph = cfg.steps_per_hour
    p_start = 1 / (2 * sph)  # idle spells average 2 hours
    p_end = 1 / (12 * sph)  # jobs average 12 hours

    job = np.full((T, N), -1, dtype=np.int32)
    current = np.where(rng.random(N) < 0.8, np.arange(N), -1)
    n_jobs = N
    for t in range(T):
        idle = current < 0
        start = idle & (rng.random(N) < p_start)
        n_new = int(start.sum())
        current[start] = np.arange(n_jobs, n_jobs + n_new)
        n_jobs += n_new
        job[t] = current
        stop = (~idle | start) & ((rng.random(N) < p_end) | node_fail[t])
        current = np.where(stop, -1, current)

    job_base_ms = rng.uniform(200, 800, n_jobs).astype(np.float32)
    job_util = rng.uniform(0.75, 0.98, n_jobs).astype(np.float32)
    job_mem = rng.uniform(0.3, 0.9, n_jobs).astype(np.float32)
    return job >= 0, job, job_base_ms, job_util, job_mem


def simulate(cfg: SimConfig | None = None) -> SimResult:
    cfg = cfg or SimConfig()
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    T, G = cfg.n_steps, cfg.n_gpus
    sph = cfg.steps_per_hour
    steps_per_day = 24 * sph

    node_of = np.arange(G) // cfg.gpus_per_node
    rack_of_node = np.arange(cfg.n_nodes) // cfg.nodes_per_rack
    n_racks = int(rack_of_node.max()) + 1

    # ---- 1. Lifecycles -------------------------------------------------------------
    lives = _sample_lives(cfg, rng)
    L = len(lives)
    life_idx = np.full((T, G), -1, dtype=np.int32)
    progress = np.zeros((T, G), dtype=np.float32)
    mode_code = np.zeros((T, G), dtype=np.int8)
    fatal = np.zeros((T, G), dtype=bool)
    steps = np.arange(T)
    for i, lf in enumerate(lives):
        life_idx[lf.start : lf.end, lf.slot] = i
        if lf.mode == "none":
            continue
        s0 = max(int(np.ceil(lf.onset)), lf.start)
        s1 = lf.end
        if lf.precursor_steps > 0:
            frac = (steps[s0:s1] - lf.onset) / lf.precursor_steps
            shape = FAILURE_MODES[lf.mode]["shape"]
            progress[s0:s1, lf.slot] = np.clip(frac, 0, 1) ** shape
        mode_code[s0:s1, lf.slot] = MODE_CODE[lf.mode]
        if lf.fail_step is not None:
            fatal[lf.fail_step, lf.slot] = True
            progress[lf.fail_step, lf.slot] = 1.0
    online = life_idx >= 0
    li = np.where(online, life_idx, 0)

    # Per-GPU (per life) fixed traits
    chronic = np.ones(L, dtype=np.float32)
    is_chronic = rng.random(L) < cfg.chronic_straggler_frac
    chronic[is_chronic] = rng.uniform(*cfg.chronic_slowdown, is_chronic.sum())
    noisy_ecc = (rng.random(L) < cfg.noisy_ecc_frac).astype(np.float32)
    noisy_nvl = (rng.random(L) < cfg.noisy_nvlink_frac).astype(np.float32)
    temp_offset = rng.normal(0, 2.5, L).astype(np.float32)

    # ---- 2. Workload ---------------------------------------------------------------
    node_fail = fatal.reshape(T, cfg.n_nodes, cfg.gpus_per_node).any(axis=2)
    busy_n, job_n, job_base_ms, job_util, job_mem = _workload(cfg, rng, node_fail)
    busy = busy_n[:, node_of] & online
    job = job_n[:, node_of]
    jb = np.where(job >= 0, job, 0)

    # Transient stragglers: host or network hiccups that slow a GPU's steps
    transient = np.ones((T, G), dtype=np.float32)
    n_eps = rng.poisson(cfg.transient_straggler_rate_per_gpu_day * cfg.days * G)
    episodes = []
    for _ in range(n_eps):
        g = int(rng.integers(G))
        s0 = int(rng.integers(T))
        if not online[s0, g]:
            continue
        lf = lives[life_idx[s0, g]]
        s1 = min(s0 + max(1, int(rng.uniform(*cfg.transient_hours) * sph)), lf.end)
        f = float(rng.uniform(*cfg.transient_slowdown))
        transient[s0:s1, g] = np.maximum(transient[s0:s1, g], f)
        episodes.append((lf, s0, s1, f))

    # ---- 3. Metrics ----------------------------------------------------------------
    is_mode = {m: mode_code == MODE_CODE[m] for m in FAILURE_MODES}
    p = progress

    hw_slow = chronic[li] * (1 + 0.35 * p * is_mode["thermal"]) * (1 + 0.05 * p * is_mode["memory"])
    total_slow = hw_slow * transient

    util = np.where(busy, job_util[jb] + rng.normal(0, 0.03, (T, G)), rng.uniform(0, 0.02, (T, G)))
    util = np.clip(util, 0, 1).astype(np.float32)
    mem_util = np.where(busy, job_mem[jb] * util + rng.normal(0, 0.03, (T, G)), 0.0)
    mem_util = np.clip(mem_util, 0, 1).astype(np.float32)

    sm_clock = np.where(
        busy,
        np.minimum(MAX_SM_CLOCK / hw_slow + rng.normal(0, 15, (T, G)), MAX_SM_CLOCK),
        IDLE_SM_CLOCK + rng.normal(0, 10, (T, G)),
    ).astype(np.float32)

    hours = steps / sph
    rack_base = 24 + rng.normal(0, 1.5, n_racks)
    inlet = rack_base[None, :] + 2 * np.sin(2 * np.pi * (hours[:, None] - 9) / 24)
    inlet = inlet[:, rack_of_node[node_of]]
    temp = (
        inlet
        + 10
        + 45 * util
        + temp_offset[li]
        + 25 * p * is_mode["thermal"] * (0.5 + 0.5 * util)
        + rng.normal(0, 0.7, (T, G))
    ).astype(np.float32)
    throttle_thermal = (temp >= 87) | (is_mode["thermal"] & (p > 0.35) & busy)
    temp_mem = (temp + 5 + rng.normal(0, 1, (T, G))).astype(np.float32)

    power = np.where(
        busy,
        100 + 560 * util / np.sqrt(hw_slow) + rng.normal(0, 12, (T, G)),
        65 + rng.normal(0, 3, (T, G)),
    )
    power = power + rng.normal(0, 1, (T, G)) * 150 * p * is_mode["power"]
    power = np.clip(power, 20, 760).astype(np.float32)
    throttle_power = power >= POWER_CAP_W

    # Per-interval error counts
    sbe_lam = 2e-4 + 0.03 * noisy_ecc[li] + 0.003 * np.exp(7 * p) * is_mode["memory"]
    sbe = rng.poisson(sbe_lam)
    retire = rng.poisson(0.02 * sbe + 0.02 * p**2 * is_mode["memory"])
    dbe = rng.poisson(0.005 * p**4 * is_mode["memory"]) + (fatal & is_mode["memory"])
    burst = rng.random((T, G)) < (0.25 + 0.5 * p)
    nvl_lam = busy * (1e-3 + 0.05 * noisy_nvl[li])
    nvl_lam = nvl_lam + burst * 0.02 * np.exp(6 * p) * is_mode["nvlink"]
    nvl = rng.poisson(nvl_lam)
    pcie = rng.poisson(5e-4 + 0.8 * p * is_mode["power"])

    xid = np.zeros((T, G), dtype=np.int32)
    benign = busy & (rng.random((T, G)) < cfg.benign_xid_rate_per_gpu_day / steps_per_day)
    xid[benign] = rng.choice(BENIGN_XIDS, int(benign.sum()))
    xid[retire > 0] = XID_ROW_REMAP
    for m, spec in FAILURE_MODES.items():
        xid[fatal & is_mode[m]] = spec["fatal_xid"]

    def lifetime_total(x: np.ndarray) -> np.ndarray:
        x = np.where(online, x, 0).astype(np.int64)
        c = np.cumsum(x, axis=0)
        for lf in lives:
            if lf.start > 0:
                c[lf.start : lf.end, lf.slot] -= c[lf.start - 1, lf.slot]
        return c

    noise = rng.lognormal(0, 0.03, (T, G)) * rng.lognormal(0, 0.02, (T, cfg.n_nodes))[:, node_of]
    step_ms = np.where(busy, job_base_ms[jb] * total_slow * noise, np.nan).astype(np.float32)

    # ---- Assemble ------------------------------------------------------------------
    keep = online & ((rng.random((T, G)) >= cfg.scrape_drop_prob) | fatal)
    t_idx, g_idx = np.nonzero(keep)
    start = pd.Timestamp(cfg.start)
    timestamps = start + pd.to_timedelta(steps * cfg.step_minutes, unit="min")

    uuids = np.array([lf.uuid for lf in lives])
    node_names = np.array([f"node-{n:03d}" for n in range(cfg.n_nodes)])
    rack_names = np.array([f"rack-{r:02d}" for r in range(n_racks)])
    jobs_g = job[t_idx, g_idx]
    job_names = np.char.add("job-", np.char.zfill(jobs_g.astype(str), 6))
    job_ids = np.where(busy[t_idx, g_idx], job_names, None)

    tel = pd.DataFrame(
        {
            "timestamp": timestamps[t_idx],
            "rack_id": rack_names[rack_of_node[node_of[g_idx]]],
            "node_id": node_names[node_of[g_idx]],
            "gpu_index": (g_idx % cfg.gpus_per_node).astype(np.int16),
            "gpu_uuid": uuids[life_idx[t_idx, g_idx]],
            "job_id": pd.array(job_ids, dtype="string"),
            "gpu_util": np.round(util[t_idx, g_idx] * 100, 1),
            "mem_util": np.round(mem_util[t_idx, g_idx] * 100, 1),
            "sm_clock_mhz": np.round(sm_clock[t_idx, g_idx]),
            "power_w": np.round(power[t_idx, g_idx], 1),
            "temp_gpu_c": np.round(temp[t_idx, g_idx], 1),
            "temp_mem_c": np.round(temp_mem[t_idx, g_idx], 1),
            "throttle_thermal": throttle_thermal[t_idx, g_idx],
            "throttle_power": throttle_power[t_idx, g_idx],
            "ecc_sbe_total": lifetime_total(sbe)[t_idx, g_idx],
            "ecc_dbe_total": lifetime_total(dbe)[t_idx, g_idx],
            "retired_pages_total": lifetime_total(retire)[t_idx, g_idx],
            "pcie_replay_total": lifetime_total(pcie)[t_idx, g_idx],
            "nvlink_crc_total": lifetime_total(nvl)[t_idx, g_idx],
            "xid": xid[t_idx, g_idx],
            "step_compute_ms": np.round(step_ms[t_idx, g_idx], 2),
        }
    )[COLUMN_NAMES]
    for c in ("rack_id", "node_id", "gpu_uuid"):
        tel[c] = tel[c].astype("category")

    def ts(step: float | None):
        if step is None or not np.isfinite(step):
            return pd.NaT
        return start + pd.Timedelta(minutes=float(step) * cfg.step_minutes)

    def where(lf: Life) -> dict:
        node = int(node_of[lf.slot])
        return {
            "rack_id": rack_names[rack_of_node[node]],
            "node_id": node_names[node],
            "gpu_index": lf.slot % cfg.gpus_per_node,
        }

    fail_rows = []
    for lf in lives:
        if lf.mode == "none":
            continue
        fail_at = lf.fail_step
        if fail_at is None:  # censored: report when it would have failed
            fail_at = np.ceil(lf.onset + lf.precursor_steps)
        fail_rows.append(
            {
                "gpu_uuid": lf.uuid,
                **where(lf),
                "mode": lf.mode,
                "fatal_xid": FAILURE_MODES[lf.mode]["fatal_xid"],
                "onset_time": ts(lf.onset),
                "failure_time": ts(fail_at),
                "repaired_time": ts(lf.repaired_step),
                "observed": lf.fail_step is not None,
            }
        )
    failures = pd.DataFrame(fail_rows, columns=FAILURE_COLUMNS)

    strag_rows = []
    for i, lf in enumerate(lives):
        if chronic[i] > 1:
            strag_rows.append((lf, "chronic", float(chronic[i]), lf.start, lf.end))
        if lf.mode == "thermal":
            # Thermal throttling counts as straggling once it costs more than 5% speed.
            shape = FAILURE_MODES["thermal"]["shape"]
            s0 = lf.onset + lf.precursor_steps * (1 / 7) ** (1 / shape)
            if s0 < lf.end:
                strag_rows.append((lf, "thermal", 1.35, int(np.ceil(s0)), lf.end))
    for lf, s0, s1, f in episodes:
        strag_rows.append((lf, "transient", f, s0, s1))
    stragglers = pd.DataFrame(
        [
            {
                "gpu_uuid": lf.uuid,
                "node_id": where(lf)["node_id"],
                "gpu_index": where(lf)["gpu_index"],
                "kind": kind,
                "slowdown": round(f, 3),
                "start": ts(s0),
                "end": ts(s1),
            }
            for lf, kind, f, s0, s1 in strag_rows
        ],
        columns=STRAGGLER_COLUMNS,
    )

    inventory = pd.DataFrame(
        [
            {
                "gpu_uuid": lf.uuid,
                **where(lf),
                "installed_time": ts(lf.start),
                "removed_time": ts(lf.end) if lf.end < T else pd.NaT,
            }
            for lf in lives
        ]
    )

    return SimResult(tel, failures, stragglers, inventory, cfg)
