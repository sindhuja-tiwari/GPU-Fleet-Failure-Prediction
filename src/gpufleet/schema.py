"""Telemetry schema: one row per GPU per scrape.

Column names follow DCGM exporter fields where one exists, so real exports can be
mapped onto this schema later (week 2). Counters ending in ``_total`` are cumulative
over the life of a GPU and reset when a GPU is replaced (new ``gpu_uuid``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # "time", "str", "int", "float", "bool"
    unit: str
    description: str
    dcgm_field: str = ""
    min: float | None = None
    max: float | None = None
    nullable: bool = False


TELEMETRY_COLUMNS: list[Column] = [
    Column("timestamp", "time", "UTC", "Scrape time"),
    Column("rack_id", "str", "", "Rack the node sits in"),
    Column("node_id", "str", "", "Host name"),
    Column("gpu_index", "int", "", "GPU slot on the node (0-7)", min=0),
    Column("gpu_uuid", "str", "", "Physical GPU; changes when a GPU is replaced"),
    Column("job_id", "str", "", "Job running on the node, empty when idle", nullable=True),
    Column("gpu_util", "float", "%", "SM utilisation", "DCGM_FI_DEV_GPU_UTIL", 0, 100),
    Column("mem_util", "float", "%", "Memory bandwidth utilisation", "DCGM_FI_DEV_MEM_COPY_UTIL",
           0, 100),
    Column("sm_clock_mhz", "float", "MHz", "SM clock", "DCGM_FI_DEV_SM_CLOCK", 0, 3000),
    Column("power_w", "float", "W", "Board power draw", "DCGM_FI_DEV_POWER_USAGE", 0, 1000),
    Column("temp_gpu_c", "float", "C", "GPU die temperature", "DCGM_FI_DEV_GPU_TEMP", 0, 120),
    Column("temp_mem_c", "float", "C", "HBM temperature", "DCGM_FI_DEV_MEMORY_TEMP", 0, 130),
    Column("throttle_thermal", "bool", "", "Clocks limited by temperature",
           "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"),
    Column("throttle_power", "bool", "", "Clocks limited by power cap",
           "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"),
    Column("ecc_sbe_total", "int", "count", "Corrected (single-bit) ECC errors",
           "DCGM_FI_DEV_ECC_SBE_AGG_TOTAL", 0),
    Column("ecc_dbe_total", "int", "count", "Uncorrected (double-bit) ECC errors",
           "DCGM_FI_DEV_ECC_DBE_AGG_TOTAL", 0),
    Column("retired_pages_total", "int", "count", "Memory pages retired or rows remapped",
           "DCGM_FI_DEV_RETIRED_SBE", 0),
    Column("pcie_replay_total", "int", "count", "PCIe replay events",
           "DCGM_FI_DEV_PCIE_REPLAY_COUNTER", 0),
    Column("nvlink_crc_total", "int", "count", "NVLink CRC errors (all links)",
           "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL", 0),
    Column("xid", "int", "code", "Last Xid error in this interval, 0 if none",
           "DCGM_FI_DEV_XID_ERRORS", 0),
    Column("step_compute_ms", "float", "ms",
           "Time this GPU spent on its share of a training step; empty when idle",
           min=0, nullable=True),
]

COUNTER_COLUMNS = [c.name for c in TELEMETRY_COLUMNS if c.name.endswith("_total")]
COLUMN_NAMES = [c.name for c in TELEMETRY_COLUMNS]

FAILURE_COLUMNS = [
    "gpu_uuid", "rack_id", "node_id", "gpu_index", "mode", "fatal_xid",
    "onset_time", "failure_time", "repaired_time", "observed",
]
STRAGGLER_COLUMNS = ["gpu_uuid", "node_id", "gpu_index", "kind", "slowdown", "start", "end"]


class SchemaError(ValueError):
    pass


def _kind_ok(series: pd.Series, kind: str) -> bool:
    if kind == "time":
        return pd.api.types.is_datetime64_any_dtype(series)
    if kind == "str":
        return pd.api.types.is_string_dtype(series) or isinstance(
            series.dtype, pd.CategoricalDtype
        )
    if kind == "int":
        return pd.api.types.is_integer_dtype(series)
    if kind == "float":
        return pd.api.types.is_float_dtype(series)
    if kind == "bool":
        return pd.api.types.is_bool_dtype(series)
    raise ValueError(kind)


def validate_telemetry(df: pd.DataFrame, check_counters: bool = True) -> None:
    """Raise SchemaError listing every problem found, or return None if the frame is valid."""
    problems: list[str] = []

    missing = [c for c in COLUMN_NAMES if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    for col in TELEMETRY_COLUMNS:
        s = df[col.name]
        if not _kind_ok(s, col.kind):
            problems.append(f"{col.name}: expected {col.kind}, got {s.dtype}")
            continue
        if not col.nullable and s.isna().any():
            problems.append(f"{col.name}: {int(s.isna().sum())} missing values")
        if col.kind in ("int", "float"):
            vals = s.dropna().to_numpy()
            if col.min is not None and (vals < col.min).any():
                problems.append(f"{col.name}: values below {col.min}")
            if col.max is not None and (vals > col.max).any():
                problems.append(f"{col.name}: values above {col.max}")

    if df.duplicated(["timestamp", "gpu_uuid"]).any():
        problems.append("duplicate (timestamp, gpu_uuid) rows")

    if check_counters and not problems:
        ordered = df.sort_values(["gpu_uuid", "timestamp"])
        for name in COUNTER_COLUMNS:
            diffs = ordered.groupby("gpu_uuid", observed=True)[name].diff().to_numpy()
            if (np.nan_to_num(diffs) < 0).any():
                problems.append(f"{name}: counter decreases within a GPU's life")

    if problems:
        raise SchemaError("; ".join(problems))


def schema_markdown() -> str:
    """Render the schema as a Markdown table (used for docs/telemetry-schema.md)."""
    lines = [
        "| Column | Type | Unit | DCGM field | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in TELEMETRY_COLUMNS:
        dcgm = f"`{c.dcgm_field}`" if c.dcgm_field else ""
        lines.append(f"| `{c.name}` | {c.kind} | {c.unit} | {dcgm} | {c.description} |")
    return "\n".join(lines)
