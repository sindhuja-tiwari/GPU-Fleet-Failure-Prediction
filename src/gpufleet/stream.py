"""Streaming scorer: feed scrapes as they arrive, get risk scores, alerts and stragglers.

State kept per GPU:

* the last scrape (to turn the next one's counters into increments),
* scrapes of the hour still in progress,
* the last ``keep_hours`` hourly buckets, plus running totals of anything trimmed off,
* the last two hours of per-scrape signals, for straggler detection.

When a scrape arrives from a new hour, the finished hour is bucketed and every GPU with
data in it is scored with exactly the same feature code as training (a test checks the
streamed features match the batch ones).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bundle import Bundle
from .features import (
    COUNTERS,
    SINCE_INSTALL,
    features_from_buckets,
    hourly_buckets,
    scrape_signals,
)
from .schema import COLUMN_NAMES, TELEMETRY_COLUMNS
from .stragglers import StragglerConfig, detect


@dataclass
class Alert:
    gpu_uuid: str
    node_id: str
    gpu_index: int
    time: pd.Timestamp
    level: str  # "urgent" or "warning"
    p24: float
    p72: float
    reasons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {**self.__dict__, "time": self.time.isoformat()}


class StreamingScorer:
    def __init__(self, bundle: Bundle, step: str = "5min", keep_hours: int = 80,
                 cooldown_hours: int = 24, warmup_hours: int = 24,
                 straggler_cfg: StragglerConfig | None = None):
        if keep_hours < 73:
            raise ValueError("keep_hours must cover the 72h feature window plus one hour")
        self.bundle = bundle
        self.step = step
        self.keep_hours = keep_hours
        self.cooldown = pd.Timedelta(hours=cooldown_hours)
        self.warmup_hours = warmup_hours  # no alerts until a GPU has this much history
        self.straggler_cfg = straggler_cfg or StragglerConfig()
        self._last = pd.DataFrame(columns=COLUMN_NAMES)  # last raw scrape per GPU
        self._pending = []  # signal frames of the open hour(s)
        self._buckets = pd.DataFrame()
        self._offsets = pd.DataFrame(columns=["hours", *[f"{n}_n" for n in SINCE_INSTALL]])
        self._recent = pd.DataFrame()  # last 2h of signals, for stragglers
        self.current_hour: pd.Timestamp | None = None
        self.risk = pd.DataFrame()  # latest score per GPU
        self.history = pd.DataFrame()  # scores of the last 72 hours, for charts
        self.alerts: list[Alert] = []
        self._last_alert: dict[str, pd.Timestamp] = {}
        self.last_features = pd.DataFrame()

    # -- ingest -----------------------------------------------------------------------
    def ingest(self, batch: pd.DataFrame) -> list[Alert]:
        """Add scrapes (rows in the telemetry schema). Returns alerts raised by this batch."""
        if batch.empty:
            return []
        batch = batch[COLUMN_NAMES].sort_values("timestamp")
        new_alerts = []
        # Process one bucket-hour at a time so hours close in order.
        hour_of = batch["timestamp"].dt.round(self.step).dt.floor("h") + pd.Timedelta("1h")
        for hour, part in batch.groupby(hour_of, sort=True):
            if self.current_hour is not None and hour > self.current_hour:
                new_alerts += self._close_hours(hour)
            self.current_hour = hour if self.current_hour is None else max(self.current_hour,
                                                                           hour)
            self._add_signals(part)
        return new_alerts

    def _add_signals(self, part: pd.DataFrame) -> None:
        prev = self._last[self._last["gpu_uuid"].astype(str).isin(
            set(part["gpu_uuid"].astype(str)))]
        both = pd.concat([_plain(prev).assign(_new=False), _plain(part).assign(_new=True)],
                         ignore_index=True)
        sig = scrape_signals(both, self.step)
        sig = sig[sig["_new"]].drop(columns="_new")
        for c in ("gpu_uuid", "node_id", "rack_id"):
            sig[c] = sig[c].astype(str)
        self._pending.append(sig)
        last = part.sort_values("timestamp").groupby("gpu_uuid", observed=True).tail(1)
        keep = self._last[~self._last["gpu_uuid"].astype(str).isin(
            set(last["gpu_uuid"].astype(str)))]
        self._last = pd.concat([_plain(keep), _plain(last)], ignore_index=True)
        self._update_stragglers(sig)

    def _close_hours(self, upto: pd.Timestamp) -> list[Alert]:
        """Bucket every pending scrape whose hour ended before ``upto`` and score."""
        if not self._pending:
            return []
        sig = pd.concat(self._pending, ignore_index=True)
        hour = sig["timestamp"].dt.floor("h") + pd.Timedelta("1h")
        done, still = sig[hour < upto], sig[hour >= upto]
        self._pending = [still] if len(still) else []
        if done.empty:
            return []
        new = hourly_buckets(done)
        new["gpu_uuid"] = new["gpu_uuid"].astype(str)
        self._buckets = pd.concat([self._buckets, new], ignore_index=True)
        self._trim()
        return self._score(new["time"].max())

    def _trim(self) -> None:
        b = self._buckets
        newest = b["time"].max()
        cutoff = newest - pd.Timedelta(hours=self.keep_hours)
        old = b[b["time"] <= cutoff]
        if old.empty:
            return
        first = b.groupby("gpu_uuid")["time"].min()
        remaining = b[b["time"] > cutoff]
        new_first = remaining.groupby("gpu_uuid")["time"].min()
        hours = ((new_first.reindex(first.index) - first) / pd.Timedelta("1h")).fillna(0)
        add = old.groupby("gpu_uuid")[[f"{n}_n" for n in SINCE_INSTALL]].sum()
        add["hours"] = hours.reindex(add.index).fillna(0)
        self._offsets = self._offsets.add(add, fill_value=0)
        self._buckets = remaining.reset_index(drop=True)
        gone = set(self._offsets.index) - set(self._buckets["gpu_uuid"])
        self._offsets = self._offsets.drop(index=list(gone))

    def _score(self, hour: pd.Timestamp) -> list[Alert]:
        feats = features_from_buckets(self._buckets, self.step, offsets=self._offsets)
        now = feats[feats["time"] == hour].reset_index(drop=True)
        if now.empty:
            return []
        self.last_features = now
        p = self.bundle.probabilities(now)
        level = self.bundle.alert_level(p)
        risk = now[["gpu_uuid", "node_id", "rack_id", "gpu_index", "time"]].assign(
            p24=p["p24"].round(4), p72=p["p72"].round(4), level=level.to_numpy(),
            coverage_24h=now["coverage_24h"].round(3))
        self.risk = risk
        hist = pd.concat([self.history, risk[["gpu_uuid", "time", "p24", "p72", "level"]]],
                         ignore_index=True)
        self.history = hist[hist["time"] > hour - pd.Timedelta("72h")].reset_index(drop=True)
        hot = risk[(risk["level"] != "") & (now["hours_seen"] >= self.warmup_hours)]
        alerts = []
        if len(hot):
            reasons = self.bundle.explain(now.loc[hot.index])
            for (_, r), why in zip(hot.iterrows(), reasons, strict=True):
                last = self._last_alert.get(r["gpu_uuid"])
                if last is not None and r["time"] - last < self.cooldown:
                    continue
                self._last_alert[r["gpu_uuid"]] = r["time"]
                a = Alert(r["gpu_uuid"], r["node_id"], int(r["gpu_index"]), r["time"],
                          r["level"], float(r["p24"]), float(r["p72"]), why)
                alerts.append(a)
        self.alerts.extend(alerts)
        return alerts

    def _update_stragglers(self, sig: pd.DataFrame) -> None:
        recent = pd.concat([self._recent, sig], ignore_index=True)
        newest = recent["timestamp"].max()
        self._recent = recent[recent["timestamp"] > newest - pd.Timedelta("2h")]
        self._stragglers_cache = None

    @property
    def stragglers(self) -> pd.DataFrame:
        """GPUs straggling at the latest scrape (computed on demand from the last 2h)."""
        if getattr(self, "_stragglers_cache", None) is not None:
            return self._stragglers_cache
        cols = ["gpu_uuid", "node_id", "gpu_index", "timestamp", "slowdown", "cause"]
        recent = self._recent
        if recent.empty or recent["step_vs_node"].notna().sum() == 0:
            out = pd.DataFrame(columns=cols)
        else:
            newest = recent["timestamp"].max()
            flags = detect(recent, self.straggler_cfg)
            latest = flags.sort_values("timestamp").groupby("gpu_uuid", observed=True).tail(1)
            latest = latest[latest["timestamp"] >= newest - pd.Timedelta(self.step)]
            out = latest.loc[latest["straggler"], cols].reset_index(drop=True)
        self._stragglers_cache = out
        return out

    # -- convenience --------------------------------------------------------------------
    def counters(self) -> dict:
        return {"gpus_tracked": int(self._last["gpu_uuid"].nunique()),
                "buckets": len(self._buckets), "alerts": len(self.alerts),
                "current_hour": None if self.current_hour is None else str(self.current_hour)}


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    """Drop categorical dtypes so frames from different batches concatenate cleanly."""
    out = df[COLUMN_NAMES].copy()
    for col in TELEMETRY_COLUMNS:
        if col.kind == "int":
            out[col.name] = out[col.name].astype(np.int64)
        elif col.kind == "float":
            out[col.name] = out[col.name].astype(np.float32)
        elif col.kind == "bool":
            out[col.name] = out[col.name].astype(bool)
    for c in ("rack_id", "node_id", "gpu_uuid"):
        out[c] = out[c].astype(str)
    out["job_id"] = out["job_id"].astype("string")
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.as_unit("ms")
    return out


def replay(scorer: StreamingScorer, tel: pd.DataFrame, rounds_per_batch: int = 1,
           on_batch=None) -> list[Alert]:
    """Feed telemetry to ``scorer`` in scrape order, one scrape round per call by default."""
    rounds = tel["timestamp"].dt.round(scorer.step)
    alerts = []
    keys = np.sort(rounds.unique())
    for i in range(0, len(keys), rounds_per_batch):
        sel = rounds.isin(keys[i:i + rounds_per_batch])
        new = scorer.ingest(tel[sel])
        alerts += new
        if on_batch is not None:
            on_batch(i, new)
    return alerts


__all__ = ["Alert", "StreamingScorer", "replay", "COUNTERS"]
