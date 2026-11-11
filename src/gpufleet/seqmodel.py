"""Sequence model: a small neural network on each GPU's raw hourly history.

Instead of hand-made window features, the model sees the last ``lookback`` hours of nine
raw hourly signals (error rates, temperature / clock / step time vs peers, throttling and
coverage) and learns its own patterns. It is a multilayer perceptron over the stacked
window, trained with scikit-learn so it needs no deep-learning framework.

Training uses every positive row and a random share of negative rows (with weights to
undo the sampling), because positives are about 0.3% of rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEQ_COLUMNS = [
    "seq_sbe", "seq_nvlink", "seq_pcie", "seq_retired", "seq_temp_vs_node",
    "seq_clock_vs_node", "seq_step_vs_node", "seq_throttle_thermal", "seq_coverage",
]
LOG_COLUMNS = {"seq_sbe", "seq_nvlink", "seq_pcie", "seq_retired"}


def windows(df: pd.DataFrame, lookback: int = 48) -> np.ndarray:
    """Stack the last ``lookback`` hours of each row's GPU into one vector per row.

    ``df`` must hold complete hourly series per GPU (as ``build_features`` makes them),
    sorted by GPU and time. Hours before the GPU's first hour are zeros.
    """
    vals = df[SEQ_COLUMNS].astype(float).copy()
    for c in LOG_COLUMNS:
        vals[c] = np.log1p(vals[c].clip(lower=0))
    vals["seq_clock_vs_node"] = vals["seq_clock_vs_node"] - 1
    vals["seq_step_vs_node"] = vals["seq_step_vs_node"] - 1
    vals = vals.fillna(0).to_numpy(dtype=np.float32)
    gpu = df["gpu_uuid"].to_numpy()
    n, k = vals.shape
    out = np.zeros((n, lookback, k), dtype=np.float32)
    for lag in range(lookback):
        shifted = np.roll(vals, lag, axis=0)
        same = np.roll(gpu, lag) == gpu
        if lag:
            same[:lag] = False
        out[:, lookback - 1 - lag, :] = np.where(same[:, None], shifted, 0)
    return out.reshape(n, lookback * k)


class SequenceMLP:
    """Fit and score on slices of one full feature table.

    Windows are built once from the full table (``full``), so a row at the start of the
    test period still sees its history from the validation period.
    """

    name = "sequence_mlp"

    def __init__(self, full: pd.DataFrame, lookback: int = 48, neg_share: float = 0.1,
                 seed: int = 0):
        self.lookback = lookback
        self.neg_share = neg_share
        self.seed = seed
        order = full.sort_values(["gpu_uuid", "time"]).index
        w = windows(full.loc[order], lookback).astype(np.float16)
        self._pos = pd.Series(np.arange(len(order)), index=order)
        self._w = w

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        return self._w[self._pos.loc[df.index].to_numpy()].astype(np.float32)

    def fit(self, train, val=None, label="label_24h"):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        y = train[label].to_numpy()
        ok = ~np.isnan(y)
        rng = np.random.default_rng(self.seed)
        keep = ok & ((y == 1) | (rng.random(len(y)) < self.neg_share))
        pos = np.flatnonzero(keep & (y == 1))
        neg = np.flatnonzero(keep & (y == 0))
        # Oversample positives towards a 1:4 mix (MLPClassifier takes no sample weights).
        reps = max(1, int(len(neg) / max(len(pos), 1) / 4))
        idx = np.concatenate([neg, np.repeat(pos, reps)])
        rng.shuffle(idx)
        x = self._x(train.iloc[idx])
        self.scaler = StandardScaler().fit(x)
        self.model = MLPClassifier(hidden_layer_sizes=(128, 32), alpha=1e-3,
                                   learning_rate_init=1e-3, batch_size=256, max_iter=30,
                                   early_stopping=True, n_iter_no_change=4,
                                   random_state=self.seed)
        self.model.fit(self.scaler.transform(x), y[idx].astype(int))
        return self

    def score(self, df, batch: int = 100_000):
        out = np.empty(len(df))
        for i in range(0, len(df), batch):
            x = self._x(df.iloc[i:i + batch])
            out[i:i + batch] = self.model.predict_proba(self.scaler.transform(x))[:, 1]
        return out
