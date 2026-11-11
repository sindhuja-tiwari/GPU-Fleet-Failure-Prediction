"""Labels and time-based splits for failure prediction.

A feature row at time ``t`` (built from data before ``t``) is labelled positive for horizon
``H`` if its GPU fails in ``(t, t + H]``. Rows are dropped or left unlabelled when the answer
is unknowable:

* **After the failure.** Rows with ``t > failure_time`` would describe a dead GPU.
* **Censored.** If ``t + H`` runs past the end of the data and the GPU has not failed by
  then, we do not know whether it would have. The label is left empty (NaN), not 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZONS = (24, 72)  # hours


def add_labels(
    features: pd.DataFrame,
    failures: pd.DataFrame,
    data_end: pd.Timestamp,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Add ``label_{H}h`` (1.0 / 0.0 / NaN), ``ttf_hours`` and ``fail_mode`` columns."""
    obs = failures.loc[failures["observed"].astype(bool), ["gpu_uuid", "failure_time", "mode"]]
    obs = obs.assign(gpu_uuid=obs["gpu_uuid"].astype(str),
                     failure_time=pd.to_datetime(obs["failure_time"]).dt.as_unit("ms"))
    df = features.merge(obs, on="gpu_uuid", how="left")
    df = df[df["failure_time"].isna() | (df["time"] <= df["failure_time"])].copy()

    ttf = (df["failure_time"] - df["time"]) / pd.Timedelta("1h")
    df["ttf_hours"] = ttf
    df["fail_mode"] = df["mode"].astype("string")
    for h in horizons:
        horizon = pd.Timedelta(hours=h)
        label = ((ttf > 0) & (ttf <= h)).astype(float)
        censored = df["failure_time"].isna() & (df["time"] + horizon > data_end)
        df[f"label_{h}h"] = label.where(~censored, np.nan)
    return df.drop(columns=["failure_time", "mode"]).reset_index(drop=True)


@dataclass(frozen=True)
class Split:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    embargo: pd.Timedelta

    def assign(self, df: pd.DataFrame) -> pd.Series:
        """'train', 'val', 'test', or 'gap' for rows dropped by the embargo."""
        t = df["time"]
        part = pd.Series("test", index=df.index, dtype="object")
        part[t < self.val_end] = "val"
        part[t < self.train_end] = "train"
        # Purge rows whose label window reaches into the next part.
        part[(t >= self.train_end - self.embargo) & (t < self.train_end)] = "gap"
        part[(t >= self.val_end - self.embargo) & (t < self.val_end)] = "gap"
        return part


def time_split(
    df: pd.DataFrame, val_days: float = 10, test_days: float = 14, embargo_hours: int = 72
) -> Split:
    """Last ``test_days`` for test, the ``val_days`` before that for validation.

    Rows within ``embargo_hours`` before each boundary are set aside, so no training label
    looks into validation time and no validation label looks into test time.
    """
    end = df["time"].max()
    val_end = (end - pd.Timedelta(days=test_days)).floor("h")
    train_end = (val_end - pd.Timedelta(days=val_days)).floor("h")
    return Split(train_end, val_end, pd.Timedelta(hours=embargo_hours))
