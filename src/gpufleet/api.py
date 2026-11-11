"""HTTP service around the streaming scorer.

Endpoints:

* ``POST /v1/telemetry``: JSON ``{"rows": [...]}`` in the telemetry schema.
* ``POST /v1/scrape``: one DCGM exporter scrape as Prometheus text (``?timestamp=`` optional).
* ``GET /v1/risk``: latest risk score per GPU, highest first.
* ``GET /v1/alerts``: recent alerts, newest first, with the reasons behind each.
* ``GET /v1/stragglers``: GPUs slowing their job at the latest scrape.
* ``GET /v1/gpu/{uuid}``: one GPU's score history (last 72h).
* ``GET /v1/summary``: everything the dashboard needs in one call.
* ``GET /metrics``: Prometheus metrics, for alert rules (see ``deploy/prometheus``).
* ``GET /healthz`` and ``GET /`` (the dashboard).

The scorer is not thread-safe, so one lock serialises ingest and reads.
"""

from __future__ import annotations

import threading
import time
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request  # needs the "serve" extra
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from .bundle import Bundle
from .ingest import _cast, parse_exporter_text, to_telemetry
from .schema import COLUMN_NAMES
from .stream import StreamingScorer

OPTIONAL = {"job_id": None, "temp_mem_c": np.nan, "step_compute_ms": np.nan, "xid": 0,
            "rack_id": "unknown"}


def rows_to_telemetry(rows: list[dict]) -> pd.DataFrame:
    """Turn JSON rows into a telemetry frame. Raises ValueError on missing columns."""
    df = pd.DataFrame(rows)
    for col, default in OPTIONAL.items():
        if col not in df:
            df[col] = default
    missing = [c for c in COLUMN_NAMES if c not in df]
    if missing:
        raise ValueError(f"rows are missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df["job_id"] = df["job_id"].where(df["job_id"].notna() & (df["job_id"] != ""), None)
    return _cast(df)


def _records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out = out.astype(object).where(out.notna(), None)
    return out.to_dict("records")


class Service:
    """State behind the app: the scorer, a lock, and ingest counters."""

    def __init__(self, scorer: StreamingScorer, rack_map: dict | None = None):
        self.scorer = scorer
        self.rack_map = rack_map or {}
        self.lock = threading.Lock()
        self.rows_total = 0
        self.batches_total = 0
        self.last_ingest_seconds = 0.0
        self.started = time.time()
        self._xid_last: dict[str, int] = {}

    def ingest(self, tel: pd.DataFrame) -> list[dict]:
        with self.lock:
            t0 = time.perf_counter()
            alerts = self.scorer.ingest(tel)
            self.last_ingest_seconds = time.perf_counter() - t0
            self.rows_total += len(tel)
            self.batches_total += 1
        return [a.as_dict() for a in alerts]

    def ingest_exporter_text(self, text: str, timestamp=None, job_label="pod") -> list[dict]:
        long = parse_exporter_text(text, timestamp or pd.Timestamp.now("UTC").tz_localize(None),
                                   job_label or None)
        if long.empty:
            raise ValueError("no per-GPU DCGM samples in the request")
        tel = to_telemetry(long, rack_map=self.rack_map, xid_is_last_value=False)
        # The exporter's Xid field holds the last Xid seen; keep only changes.
        with self.lock:
            xid = []
            for uuid, v in zip(tel["gpu_uuid"].astype(str), tel["xid"], strict=True):
                xid.append(v if v != self._xid_last.get(uuid, 0) else 0)
                self._xid_last[uuid] = int(v)
        tel["xid"] = np.asarray(xid, dtype=np.int64)
        return self.ingest(tel)

    # -- reads ----------------------------------------------------------------------------
    def risk(self, top: int | None = None, level: str | None = None) -> list[dict]:
        with self.lock:
            r = self.scorer.risk
        if r.empty:
            return []
        if level:
            r = r[r["level"] == level]
        r = r.sort_values(["p24", "p72"], ascending=False)
        return _records(r.head(top) if top else r)

    def alerts(self, limit: int = 100) -> list[dict]:
        with self.lock:
            al = list(self.scorer.alerts[-limit:])
        return [a.as_dict() for a in reversed(al)]

    def stragglers(self) -> list[dict]:
        with self.lock:
            s = self.scorer.stragglers
        return _records(s.assign(slowdown=s["slowdown"].astype(float).round(3)))

    def gpu(self, uuid: str) -> dict:
        with self.lock:
            h = self.scorer.history
            r = self.scorer.risk
            al = [a.as_dict() for a in self.scorer.alerts if a.gpu_uuid == uuid]
        hist = h[h["gpu_uuid"] == uuid] if len(h) else h
        cur = r[r["gpu_uuid"] == uuid] if len(r) else r
        if hist.empty and cur.empty:
            raise KeyError(uuid)
        return {"gpu_uuid": uuid, "current": (_records(cur) or [None])[0],
                "history": _records(hist.drop(columns="gpu_uuid")), "alerts": al[-10:]}

    def summary(self, top: int = 15) -> dict:
        with self.lock:
            r = self.scorer.risk
            counters = self.scorer.counters()
        levels = r["level"].value_counts().to_dict() if len(r) else {}
        strag = self.stragglers()
        return {
            "counters": {**counters, "rows_ingested": self.rows_total,
                         "last_ingest_ms": round(1000 * self.last_ingest_seconds, 1)},
            "levels": {"urgent": int(levels.get("urgent", 0)),
                       "warning": int(levels.get("warning", 0))},
            "thresholds": {"p24": self.scorer.bundle.t24, "p72": self.scorer.bundle.t72},
            "risk": self.risk(top), "alerts": self.alerts(20), "stragglers": strag,
        }

    def metrics(self) -> str:
        with self.lock:
            r = self.scorer.risk.copy()
            alerts = list(self.scorer.alerts)
        s = self.stragglers()
        lines = []

        def metric(name, kind, help_, samples):
            lines.append(f"# HELP {name} {help_}")
            lines.append(f"# TYPE {name} {kind}")
            for labels, v in samples:
                lab = ",".join(f'{k}="{val}"' for k, val in labels.items())
                lines.append(f"{name}{{{lab}}} {v:g}" if lab else f"{name} {v:g}")

        def gl(row):
            return {"gpu_uuid": row["gpu_uuid"], "node": row["node_id"],
                    "gpu": int(row["gpu_index"])}

        rows = list(r.iterrows()) if len(r) else []
        metric("gpufleet_failure_probability", "gauge",
               "Calibrated probability that the GPU fails within the horizon.",
               [({**gl(x), "horizon": f"{h}h"}, float(x[f"p{h}"]))
                for _, x in rows for h in (24, 72)])
        metric("gpufleet_alert_level", "gauge",
               "Current alert tier per GPU: 2 urgent, 1 warning, 0 none.",
               [(gl(x), {"urgent": 2, "warning": 1}.get(x["level"], 0)) for _, x in rows])
        metric("gpufleet_straggler_slowdown", "gauge",
               "Step time relative to node peers, for GPUs flagged as stragglers.",
               [({"gpu_uuid": x["gpu_uuid"], "node": x["node_id"], "gpu": int(x["gpu_index"]),
                  "cause": x["cause"]}, float(x["slowdown"])) for x in s])
        counts = {"urgent": 0, "warning": 0}
        for a in alerts:
            counts[a.level] = counts.get(a.level, 0) + 1
        metric("gpufleet_alerts_total", "counter", "Alerts raised since start.",
               [({"level": k}, v) for k, v in counts.items()])
        metric("gpufleet_ingested_rows_total", "counter", "Telemetry rows ingested.",
               [({}, self.rows_total)])
        metric("gpufleet_ingest_last_seconds", "gauge", "Time to process the last batch.",
               [({}, self.last_ingest_seconds)])
        metric("gpufleet_gpus_tracked", "gauge", "GPUs seen so far.",
               [({}, len(r))])
        return "\n".join(lines) + "\n"


def create_app(bundle: Bundle | str | Path, step: str = "5min", rack_map: dict | None = None,
               **scorer_kw):
    if not isinstance(bundle, Bundle):
        bundle = Bundle.load(bundle)
    svc = Service(StreamingScorer(bundle, step=step, **scorer_kw), rack_map)
    app = FastAPI(title="gpufleet", version="0.8",
                  description="GPU failure risk and straggler detection from telemetry.")
    app.state.service = svc

    @app.post("/v1/telemetry")
    def post_telemetry(payload: dict):
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise HTTPException(422, "body must be {\"rows\": [...]} with at least one row")
        try:
            tel = rows_to_telemetry(rows)
        except (ValueError, TypeError, KeyError) as e:
            raise HTTPException(422, str(e)) from e
        return {"accepted": len(tel), "alerts": svc.ingest(tel)}

    @app.post("/v1/scrape")
    async def post_scrape(request: Request, timestamp: str | None = None,
                          job_label: str = "pod"):
        text = (await request.body()).decode()
        try:
            ts = pd.Timestamp(timestamp).tz_localize(None) if timestamp else None
            alerts = svc.ingest_exporter_text(text, ts, job_label)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"alerts": alerts}

    @app.get("/v1/risk")
    def get_risk(top: int | None = None, level: str | None = None):
        return svc.risk(top, level)

    @app.get("/v1/alerts")
    def get_alerts(limit: int = 100):
        return svc.alerts(limit)

    @app.get("/v1/stragglers")
    def get_stragglers():
        return svc.stragglers()

    @app.get("/v1/gpu/{uuid}")
    def get_gpu(uuid: str):
        try:
            return svc.gpu(uuid)
        except KeyError as e:
            raise HTTPException(404, f"unknown GPU {uuid}") from e

    @app.get("/v1/summary")
    def get_summary(top: int = 15):
        return svc.summary(top)

    @app.get("/metrics", response_class=PlainTextResponse)
    def get_metrics():
        return svc.metrics()

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "uptime_s": round(time.time() - svc.started),
                "model": {k: bundle.meta.get(k) for k in ("trained_from", "trained_to")}}

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return resources.files("gpufleet").joinpath("static/dashboard.html").read_text()

    fonts = {"inter.woff2", "newsreader.woff2", "newsreader-italic.woff2"}

    @app.get("/static/fonts/{name}")
    def font(name: str):
        if name not in fonts:
            raise HTTPException(404, "unknown font")
        data = resources.files("gpufleet").joinpath(f"static/fonts/{name}").read_bytes()
        return Response(data, media_type="font/woff2",
                        headers={"Cache-Control": "public, max-age=86400"})

    return app
