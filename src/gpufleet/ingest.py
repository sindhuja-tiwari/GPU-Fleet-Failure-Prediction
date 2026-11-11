"""Load real GPU telemetry into the project schema.

Two input formats are supported, both produced by NVIDIA's DCGM exporter:

* **Exporter text**: what ``curl http://<node>:9400/metrics`` returns, one file per scrape.
* **Prometheus range query**: the JSON that ``/api/v1/query_range`` returns, for pulling
  history out of an existing Prometheus.

Both are first parsed into a *long* table (one row per metric sample) and then pivoted by
``to_telemetry`` into the wide schema in ``gpufleet.schema``. The reverse direction
(``telemetry_to_exporter_text`` and ``telemetry_to_prometheus_range``) turns simulated data
into the same formats, which is how the loader is tested without access to a real fleet.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import COLUMN_NAMES, COUNTER_COLUMNS, TELEMETRY_COLUMNS

# Our column -> DCGM fields that can supply it, in order of preference. For the
# retired-pages column, every field present is summed (older GPUs retire pages, newer
# ones remap rows).
FIELD_ALIASES: dict[str, list[str]] = {
    "gpu_util": ["DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_PROF_GR_ENGINE_ACTIVE"],
    "mem_util": ["DCGM_FI_DEV_MEM_COPY_UTIL", "DCGM_FI_PROF_DRAM_ACTIVE"],
    "sm_clock_mhz": ["DCGM_FI_DEV_SM_CLOCK"],
    "power_w": ["DCGM_FI_DEV_POWER_USAGE"],
    "temp_gpu_c": ["DCGM_FI_DEV_GPU_TEMP"],
    "temp_mem_c": ["DCGM_FI_DEV_MEMORY_TEMP"],
    "throttle_reasons": ["DCGM_FI_DEV_CLOCK_THROTTLE_REASONS", "DCGM_FI_DEV_CLOCKS_EVENT_REASONS"],
    "ecc_sbe_total": ["DCGM_FI_DEV_ECC_SBE_AGG_TOTAL", "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL"],
    "ecc_dbe_total": ["DCGM_FI_DEV_ECC_DBE_AGG_TOTAL", "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL"],
    "pcie_replay_total": ["DCGM_FI_DEV_PCIE_REPLAY_COUNTER"],
    "nvlink_crc_total": ["DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL"],
    "xid_last": ["DCGM_FI_DEV_XID_ERRORS"],
}
RETIRED_FIELDS = [
    "DCGM_FI_DEV_RETIRED_SBE",
    "DCGM_FI_DEV_RETIRED_DBE",
    "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS",
    "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS",
]
# Fields written back out when exporting simulated data (first alias of each).
EXPORT_FIELDS = {col: names[0] for col, names in FIELD_ALIASES.items()}
EXPORT_FIELDS["retired_pages_total"] = "DCGM_FI_DEV_RETIRED_SBE"

# Bits of the clock throttle / clock event reason mask (nvml.h).
THROTTLE_POWER_BITS = 0x04 | 0x80  # SW power cap, HW power brake
THROTTLE_THERMAL_BITS = 0x20 | 0x40  # SW thermal slowdown, HW thermal slowdown

REQUIRED_SOURCE = ["gpu_util", "sm_clock_mhz", "power_w", "temp_gpu_c"]

LONG_COLUMNS = ["timestamp", "metric", "node_id", "gpu_index", "gpu_uuid", "job_id", "value"]

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+(?P<value>\S+)(?:\s+(?P<ts>\d+))?$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------


def _labels_to_ids(labels: dict, job_label: str | None) -> tuple[str, int, str, str | None]:
    node = labels.get("Hostname") or labels.get("hostname") or labels.get("instance", "unknown")
    node = node.split(":")[0]  # instance labels carry a port
    gpu = int(labels.get("gpu", labels.get("minor_number", -1)))
    uuid = labels.get("UUID") or labels.get("uuid") or f"{node}/gpu{gpu}"
    job = labels.get(job_label) if job_label else None
    return node, gpu, uuid, (job or None)


def parse_exporter_text(
    text: str, timestamp: pd.Timestamp | str | None = None, job_label: str | None = "pod"
) -> pd.DataFrame:
    """Parse one DCGM exporter scrape (Prometheus text format) into long rows.

    ``timestamp`` is used for samples that carry none (the exporter normally omits them).
    ``job_label`` names the label that identifies the training job; on Kubernetes the
    exporter adds ``pod``. Pass ``None`` if there is no such label.
    """
    default_ts = pd.Timestamp(timestamp) if timestamp is not None else None
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m or not m.group("name").startswith("DCGM_FI_"):
            continue
        labels = dict(_LABEL_RE.findall(m.group("labels") or ""))
        if "gpu" not in labels and "minor_number" not in labels:
            continue  # node-level metric
        if m.group("ts"):
            ts = pd.Timestamp(int(m.group("ts")), unit="ms")
        elif default_ts is not None:
            ts = default_ts
        else:
            raise ValueError("sample has no timestamp and no default timestamp was given")
        node, gpu, uuid, job = _labels_to_ids(labels, job_label)
        rows.append((ts, m.group("name"), node, gpu, uuid, job, float(m.group("value"))))
    return pd.DataFrame(rows, columns=LONG_COLUMNS)


def load_exporter_scrapes(paths: Iterable[str | Path], job_label: str | None = "pod"):
    """Parse many scrape files. Each file's time comes from its name, which must be an ISO
    timestamp or Unix seconds (for example ``2026-01-01T00-05-00Z.prom`` or ``1767225900.prom``).
    """
    frames = []
    for path in paths:
        path = Path(path)
        frames.append(parse_exporter_text(path.read_text(), _time_from_name(path.stem), job_label))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LONG_COLUMNS)


def _time_from_name(stem: str) -> pd.Timestamp:
    if stem.isdigit():
        return pd.Timestamp(int(stem), unit="s")
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[T_ ](\d{2})[-:](\d{2})[-:](\d{2})", stem)
    if not m:
        raise ValueError(f"cannot read a timestamp from file name {stem!r}")
    return pd.Timestamp(f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4)}")


def load_prometheus_range(source: str | Path | dict, job_label: str | None = "pod"):
    """Parse a Prometheus ``query_range`` response (a ``matrix``) into long rows.

    ``source`` is a path to the saved JSON, or the already-decoded response. To fetch
    everything the exporter publishes, query ``{__name__=~"DCGM_FI_.*"}``.
    """
    if isinstance(source, dict):
        payload = source
    else:
        payload = json.loads(Path(source).read_text())
    data = payload.get("data", payload)
    if data.get("resultType") != "matrix":
        raise ValueError("expected a Prometheus range query result (resultType 'matrix')")
    rows = []
    for series in data["result"]:
        labels = series["metric"]
        name = labels.get("__name__", "")
        if not name.startswith("DCGM_FI_"):
            continue
        node, gpu, uuid, job = _labels_to_ids(labels, job_label)
        for ts, value in series["values"]:
            rows.append((ts, name, node, gpu, uuid, job, float(value)))
    out = pd.DataFrame(rows, columns=LONG_COLUMNS)
    out["timestamp"] = pd.to_datetime(out["timestamp"].astype(float), unit="s").dt.round("ms")
    return out


# ---------------------------------------------------------------------------------------
# Long -> wide schema
# ---------------------------------------------------------------------------------------


def to_telemetry(
    long: pd.DataFrame,
    rack_map: dict[str, str] | None = None,
    xid_is_last_value: bool = True,
) -> pd.DataFrame:
    """Pivot long DCGM samples into the telemetry schema.

    * Counters the source does not publish are filled with 0, and a warning names them.
    * The DCGM Xid field holds the *last* Xid seen, so it stays set after an error. With
      ``xid_is_last_value`` the value is turned into an event: the code appears only on the
      scrape where it changed, matching the schema's "Xid in this interval".
    * ``step_compute_ms`` does not come from DCGM. It stays empty unless the frame already
      has a ``step_compute_ms`` metric (for example joined from training logs).
    * ``rack_map`` maps node names to racks; nodes not in it get ``"unknown"``.
    """
    if long.empty:
        raise ValueError("no DCGM samples to convert")
    long = long.copy()
    long["job_id"] = long["job_id"].astype(object).where(long["job_id"].notna(), "")
    keys = ["timestamp", "node_id", "gpu_index", "gpu_uuid"]
    wide = long.pivot_table(
        index=keys, columns="metric", values="value", aggfunc="last"
    ).reset_index()
    jobs = (
        long[long["job_id"] != ""]
        .groupby(keys)["job_id"]
        .agg(lambda s: ",".join(sorted(set(s))))
    )
    wide = wide.merge(jobs.rename("job_id"), left_on=keys, right_index=True, how="left")

    out = wide[keys].copy()
    missing_counters = []
    for col, fields in FIELD_ALIASES.items():
        present = [f for f in fields if f in wide.columns]
        if present:
            values = wide[present[0]]
            for f in present[1:]:
                values = values.fillna(wide[f])
            out[col] = values
        elif col in COUNTER_COLUMNS:
            missing_counters.append(col)
            out[col] = 0.0
        elif col in REQUIRED_SOURCE:
            raise ValueError(f"source has none of {fields}, needed for {col}")
        else:
            out[col] = np.nan

    retired = [f for f in RETIRED_FIELDS if f in wide.columns]
    if retired:
        out["retired_pages_total"] = wide[retired].fillna(0).sum(axis=1)
    else:
        missing_counters.append("retired_pages_total")
        out["retired_pages_total"] = 0.0
    if missing_counters:
        warnings.warn(
            f"source has no data for {missing_counters}; filled with 0", stacklevel=2
        )

    reasons = out.pop("throttle_reasons").fillna(0).astype(np.int64)
    out["throttle_thermal"] = (reasons & THROTTLE_THERMAL_BITS) != 0
    out["throttle_power"] = (reasons & THROTTLE_POWER_BITS) != 0

    out = out.sort_values(["gpu_uuid", "timestamp"])
    last = out.pop("xid_last").fillna(0).astype(np.int64)
    if xid_is_last_value:
        prev = last.groupby(out["gpu_uuid"]).shift().fillna(0).astype(np.int64)
        out["xid"] = np.where(last != prev, last, 0)
    else:
        out["xid"] = last

    out["job_id"] = wide.loc[out.index, "job_id"]
    out["step_compute_ms"] = (
        wide.loc[out.index, "step_compute_ms"] if "step_compute_ms" in wide.columns else np.nan
    )
    racks = rack_map or {}
    out["rack_id"] = [racks.get(n, "unknown") for n in out["node_id"]]
    return _cast(out)


def _cast(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLUMN_NAMES].copy()
    for col in TELEMETRY_COLUMNS:
        if col.kind == "int":
            df[col.name] = df[col.name].fillna(0).round().astype(np.int64)
        elif col.kind == "float":
            df[col.name] = df[col.name].astype(np.float32)
        elif col.kind == "bool":
            df[col.name] = df[col.name].astype(bool)
    df["gpu_index"] = df["gpu_index"].astype(np.int16)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.as_unit("ms")
    df["job_id"] = df["job_id"].astype("string")
    for c in ("rack_id", "node_id", "gpu_uuid"):
        df[c] = df[c].astype(str).astype("category")
    return df.sort_values(["timestamp", "node_id", "gpu_index"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Wide schema -> DCGM formats (for tests and demos)
# ---------------------------------------------------------------------------------------


def _export_values(tel: pd.DataFrame) -> pd.DataFrame:
    """Per-row values of each exported DCGM field, with Xid held as a last-value gauge."""
    t = tel.sort_values(["gpu_uuid", "timestamp"])
    vals = pd.DataFrame(index=t.index)
    for col, field in EXPORT_FIELDS.items():
        if col in ("throttle_reasons", "xid_last"):
            continue
        vals[field] = t[col].astype(float)
    vals[EXPORT_FIELDS["throttle_reasons"]] = (
        t["throttle_thermal"].astype(int) * 0x20 + t["throttle_power"].astype(int) * 0x04
    )
    xid = t["xid"].where(t["xid"] != 0)
    vals[EXPORT_FIELDS["xid_last"]] = xid.groupby(t["gpu_uuid"], observed=True).ffill().fillna(0)
    return vals


def _label_str(row, job_label: str) -> str:
    labels = (
        f'gpu="{row.gpu_index}",UUID="{row.gpu_uuid}",device="nvidia{row.gpu_index}",'
        f'modelName="Simulated GPU",Hostname="{row.node_id}"'
    )
    if isinstance(row.job_id, str) and row.job_id:
        labels += f',{job_label}="{row.job_id}"'
    return labels


def telemetry_to_exporter_text(scrape: pd.DataFrame, job_label: str = "pod") -> str:
    """Render the rows of one scrape (one node, one time) as DCGM exporter text."""
    vals = _export_values(scrape)
    rows = scrape.loc[vals.index]
    lines = []
    for field in vals.columns:
        lines.append(f"# HELP {field} Exported by gpufleet.")
        lines.append(f"# TYPE {field} gauge")
        for row, v in zip(rows.itertuples(), vals[field], strict=True):
            if not (isinstance(v, float) and math.isnan(v)):
                lines.append(f"{field}{{{_label_str(row, job_label)}}} {v:g}")
    return "\n".join(lines) + "\n"


def telemetry_to_prometheus_range(tel: pd.DataFrame, job_label: str = "pod") -> dict:
    """Render telemetry as a Prometheus ``query_range`` response."""
    vals = _export_values(tel)
    rows = tel.loc[vals.index]
    secs = (rows["timestamp"].astype("datetime64[ms]").astype(np.int64) / 1000).to_numpy()
    keys = pd.DataFrame(
        {
            "node": rows["node_id"].astype(str).to_numpy(),
            "gpu": rows["gpu_index"].to_numpy(),
            "uuid": rows["gpu_uuid"].astype(str).to_numpy(),
            "job": rows["job_id"].fillna("").astype(str).to_numpy(),
        }
    )
    result = []
    for (node, gpu, uuid, job), idx in keys.groupby(list(keys.columns)).groups.items():
        idx = np.asarray(idx)
        base = {"gpu": str(gpu), "UUID": uuid, "Hostname": node, "instance": f"{node}:9400",
                "job": "dcgm-exporter"}
        if job:
            base[job_label] = job
        for field in vals.columns:
            v = vals[field].to_numpy()[idx]
            ok = ~np.isnan(v)
            result.append(
                {
                    "metric": {"__name__": field, **base},
                    "values": [[float(s), f"{x:g}"] for s, x in zip(secs[idx][ok], v[ok],
                                                                    strict=True)],
                }
            )
    return {"status": "success", "data": {"resultType": "matrix", "result": result}}
