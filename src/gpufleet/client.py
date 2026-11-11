"""Client side of the service: send telemetry to a running API (standard library only)."""

from __future__ import annotations

import json
import math
import time
import urllib.request

import numpy as np
import pandas as pd

from .schema import COLUMN_NAMES


def telemetry_to_rows(tel: pd.DataFrame) -> list[dict]:
    """The reverse, for clients (the replayer) that post JSON."""
    out = tel[COLUMN_NAMES].copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    for c in ("rack_id", "node_id", "gpu_uuid"):
        out[c] = out[c].astype(str)
    out["job_id"] = out["job_id"].astype(object).where(out["job_id"].notna(), None)
    recs = out.to_dict("records")
    for r in recs:  # NaN is not valid JSON
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return recs


def post_json(url: str, payload: dict, timeout: float = 30) -> dict:
    """POST JSON with the standard library (the replayer needs no HTTP client package)."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def replay_to_url(tel: pd.DataFrame, url: str, step: str = "5min", rounds_per_sec: float = 0,
                  log=print) -> int:
    """Send telemetry to a running service one scrape round at a time. Returns alerts seen."""
    rounds = tel["timestamp"].dt.round(step)
    keys = np.sort(rounds.unique())
    n_alerts = 0
    for i, k in enumerate(keys):
        t0 = time.time()
        res = post_json(url.rstrip("/") + "/v1/telemetry",
                        {"rows": telemetry_to_rows(tel[rounds == k])})
        for a in res["alerts"]:
            n_alerts += 1
            why = ", ".join(r["label"] for r in a["reasons"][:2])
            log(f"{a['time']} {a['level'].upper():7s} {a['node_id']} gpu{a['gpu_index']} "
                f"p24={a['p24']:.2f} p72={a['p72']:.2f} ({why})")
        if i % 288 == 0:
            log(f"sent round {i + 1}/{len(keys)} ({pd.Timestamp(k)})")
        if rounds_per_sec > 0:
            time.sleep(max(0.0, 1 / rounds_per_sec - (time.time() - t0)))
    return n_alerts
