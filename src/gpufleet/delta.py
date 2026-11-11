"""Real-data check on the public NCSA Delta Xid error logs.

Source: Cui et al., "Story of Two GPUs: Characterizing the Resilience of Hopper H100 and
Ampere A100 GPUs" (SC'25, arXiv:2503.11901), artifact on Zenodo, record 15851780,
CC BY 4.0. A100 logs cover 2022-10-03 to 2025-03-15, H100 logs 2024-10-21 to 2025-03-15.

What the public release contains (and what it does not):

* ``*_allfileregexbookkeeping_*.pkl``: for every day, the list of Xid events seen on the
  whole partition, after 5-second de-duplication. **No node, GPU, or time of day**: the
  release was sanitised for privacy. So the finest grain is "partition x day".
* ``*_cond_prob_forward_*.pkl``: for every source Xid event, the closest later Xid on the
  same GPU (or another GPU on the same node) **within 5 seconds**, with the delay; ``-1``
  means nothing followed within 5 seconds.

That rules out the test we would really want (per-GPU, hours ahead). This module runs
the two tests the release does support:

1. **Fleet-day precursors.** Does a burst of non-critical Xids on the partition in the
   last 7 days make a critical Xid (somewhere on the partition) more likely in the next
   1 or 3 days? Scored with blocked cross-validation and a circular-shift null that keeps
   the autocorrelation of both series.
2. **Same-GPU cascades.** How often does each Xid lead to a critical Xid on the same GPU
   within 5 seconds?

Pickles are untrusted input: :func:`safe_load` refuses every class except ``defaultdict``
and ``list``, so a tampered file cannot run code.
"""

from __future__ import annotations

import hashlib
import pickle
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ZENODO_RECORD = 15851780
ZENODO_URL = "https://zenodo.org/api/records/{rec}/files/{name}/content"

FILES = {
    "delta_A100_allfileregexbookkeeping_deduped_5s_sanitized.pkl":
        "d276e770272be5a5a78076efa304b39e34ee6e9e67ff72ab14d5f924a78f1558",
    "delta_H100_allfileregexbookkeeping_deduped_5s_sanitized.pkl":
        "d7e67b3f639ee1297cfadbc089fcafeca1465bb1655be9bb9c8b9a4e1712a0d1",
    "delta_A100_cond_prob_forward_same_dev_id_same_node_pci_deduped_5s_5s_sanitized.pkl":
        "f41ed568724c58ee818078158010a5dd4dc783698c902c664a30828cd46c15dc",
    "delta_H100_cond_prob_forward_same_dev_id_same_node_pci_deduped_5s_5s_sanitized.pkl":
        "c817bb3d391e2a9fb05e4d0d91d637a05b79881e88ccfeb124d417f16c5ee61a",
    "delta_A100_cond_prob_forward_same_node_diff_dev_id_pci_deduped_5s_5s_sanitized.pkl":
        "e877e02ff12e2f226623ca458152ecad3af208f6b3b6be28cc68d733762a125a",
}

# Template IDs in the release are 1000 + Xid code; 368 is a generic "NVRM: GPU at" line.
XID_OFFSET = 1000

# Errors that take a GPU out of service: double-bit ECC, row-remap failure, fell off the
# bus, contained / uncontained memory errors.
HARD_FAILURE = {48, 64, 79, 94, 95}
# GSP firmware timeouts / errors and SPI PMU RPC failures: the GPU usually needs a reset.
GSP_FAILURE = {119, 120, 122, 123}
TARGETS = {"hard": HARD_FAILURE, "gsp": GSP_FAILURE}

XID_NAMES = {
    13: "graphics engine exception (usually user code)",
    31: "MMU fault",
    43: "GPU stopped processing",
    45: "preemptive cleanup",
    48: "double-bit ECC",
    63: "row remap / page retirement",
    64: "row remap failure",
    74: "NVLink error",
    79: "fell off the bus",
    94: "contained ECC error",
    95: "uncontained ECC error",
    109: "context switch timeout",
    119: "GSP RPC timeout",
    120: "GSP error",
    122: "SPI PMU RPC read failure",
    123: "SPI PMU RPC write failure",
    368 - XID_OFFSET: "NVRM 'GPU at' message (not an Xid)",
}


# --------------------------------------------------------------------------- loading


class _SafeUnpickler(pickle.Unpickler):
    _ALLOWED = {("collections", "defaultdict"): defaultdict, ("builtins", "list"): list}

    def find_class(self, module: str, name: str):
        try:
            return self._ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}") from None


def safe_load(path: str | Path):
    """Unpickle plain containers only (dict, list, str, numbers, defaultdict)."""
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: str | Path) -> Path:
    """Fetch the Delta files from Zenodo into ``dest`` (skips files already there)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = dest / name
        if not path.exists():
            url = ZENODO_URL.format(rec=ZENODO_RECORD, name=name)
            print(f"downloading {name}")
            with urllib.request.urlopen(url, timeout=120) as r:
                path.write_bytes(r.read())
        if _sha256(path) != digest:
            raise ValueError(f"{path} does not match the published file (sha256 mismatch)")
    return dest


def xid_of(template_id: int) -> str:
    """Column label for a template: the Xid code as text, or "nvrm" for template 368."""
    t = int(template_id)
    return str(t - XID_OFFSET) if t > XID_OFFSET else "nvrm"


def _key(label: str) -> tuple[int, str]:
    return (int(label), "") if label.isdigit() else (10**6, label)


def _in(label: str, xids: set[int]) -> bool:
    return label.isdigit() and int(label) in xids


def daily_counts(events_by_day: dict) -> pd.DataFrame:
    """Day x Xid table of event counts. Missing days are filled with zeros."""
    rows = defaultdict(lambda: defaultdict(int))
    for day, events in events_by_day.items():
        for e in events:
            rows[pd.Timestamp(day)][xid_of(e["TemplateID"])] += 1
    df = pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(int).sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full, fill_value=0)
    df.index.name = "day"
    return df[sorted(df.columns, key=_key)]


def cascade_table(prop: dict, targets: set[int] | None = None) -> pd.DataFrame:
    """For each source Xid: events, share followed by anything / by a target within 5 s."""
    targets = targets if targets is not None else HARD_FAILURE | GSP_FAILURE
    out = []
    for src, dsts in prop.items():
        n = sum(len(v) for v in dsts.values())
        followed = sum(len(v) for k, v in dsts.items() if k != -1)
        crit = {xid_of(k): len(v) for k, v in dsts.items() if k != -1 and _in(xid_of(k), targets)}
        out.append({
            "source_xid": xid_of(src),
            "events": n,
            "followed_within_5s": followed / n if n else np.nan,
            "critical_within_5s": sum(crit.values()) / n if n else np.nan,
            "critical_followers": ", ".join(
                f"{k}:{v}" for k, v in sorted(crit.items(), key=lambda kv: _key(kv[0]))),
        })
    df = pd.DataFrame(out)
    return df.sort_values("source_xid", key=lambda s: s.map(_key)).reset_index(drop=True)


# --------------------------------------------------------------------- fleet-day test


def day_table(counts: pd.DataFrame, target: set[int], lookback: int = 7, horizon: int = 1,
              min_days: int = 5, level_days: int = 90,
              ) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Features known at the start of day t, and whether a target Xid occurs in t..t+h-1.

    Features use days before t only, so nothing from the label window leaks in:

    * ``xid<N>_last7d``: log count of Xid N over the previous ``lookback`` days;
    * ``xid<N>_burst``: the same count relative to Xid N's own level over the previous
      ``level_days`` days, so a slow drift (new driver, new workload mix) does not count;
    * ``target_last30d``: log count of target Xids over the previous 30 days.

    Returns (X, y, count_columns, burst_columns).
    """
    tcols = [c for c in counts.columns if _in(c, target)]
    present = (counts > 0).sum()
    pcols = [c for c in counts.columns if not _in(c, target) and present[c] >= min_days]

    tgt = counts[tcols].sum(axis=1) if tcols else pd.Series(0, index=counts.index)
    # y_t = any target event in [t, t+h-1]
    future = tgt[::-1].rolling(horizon, min_periods=horizon).sum()[::-1]
    y = (future > 0).astype(int)

    past = counts[pcols].rolling(lookback, min_periods=lookback).sum().shift(1)
    level = counts[pcols].rolling(level_days, min_periods=level_days).sum().shift(1)
    X = np.log1p(past)
    X.columns = count_cols = [f"xid{c}_last{lookback}d" for c in pcols]
    burst = X.to_numpy() - np.log1p(level.to_numpy() * lookback / level_days)
    burst_cols = [f"xid{c}_burst" for c in pcols]
    X = pd.concat([X, pd.DataFrame(burst, index=X.index, columns=burst_cols)], axis=1)
    X["target_last30d"] = np.log1p(tgt.rolling(30, min_periods=30).sum().shift(1))

    # Burst columns need ``level_days`` of history; callers decide whether to drop those rows.
    ok = X[count_cols + ["target_last30d"]].notna().all(axis=1) & future.notna()
    return X[ok], y[ok], count_cols, burst_cols


def _blocked_oof(X: np.ndarray, y: np.ndarray, n_blocks: int, gap: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    oof = np.full(n, np.nan)
    idx = np.arange(n)
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        test = (idx >= a) & (idx < b)
        train = (idx < a - gap) | (idx >= b + gap)
        if y[train].sum() == 0 or y[train].sum() == train.sum():
            oof[test] = y[train].mean()
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=0.3, class_weight="balanced", max_iter=2000))
        model.fit(X[train], y[train])
        oof[test] = model.predict_proba(X[test])[:, 1]
    return oof


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def _shifts(n: int, reps: int, min_shift: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(min_shift, n - min_shift, size=reps)


@dataclass
class ModelResult:
    family: str
    target: str
    horizon_days: int
    model: str
    days: int
    positive_days: int
    roc_auc: float
    pr_auc: float
    base_rate: float
    p_value: float


def model_test(X: pd.DataFrame, y: pd.Series, columns: list[str], *, n_blocks: int = 8,
               gap: int = 10, reps: int = 200, seed: int = 0) -> tuple[float, float, float]:
    """Blocked-CV ROC-AUC and PR-AUC, and a circular-shift p-value for the ROC-AUC."""
    from sklearn.metrics import average_precision_score

    Xa = X[columns].to_numpy(float)
    ya = y.to_numpy()
    oof = _blocked_oof(Xa, ya, n_blocks, gap)
    auc = _auc(ya, oof)
    pr = float(average_precision_score(ya, oof)) if ya.any() else float("nan")
    rng = np.random.default_rng(seed)
    null = []
    for k in _shifts(len(ya), reps, 30, rng):
        yr = np.roll(ya, k)
        null.append(_auc(yr, _blocked_oof(Xa, yr, n_blocks, gap)))
    null = np.array(null)
    p = (1 + np.sum(null >= auc)) / (1 + len(null))
    return auc, pr, float(p)


def univariate_test(X: pd.DataFrame, y: pd.Series, columns: list[str], *, reps: int = 2000,
                    seed: int = 0) -> pd.DataFrame:
    """AUC of each precursor's raw 7-day count, with a circular-shift p-value."""
    rng = np.random.default_rng(seed)
    ya = y.to_numpy()
    shifts = _shifts(len(ya), reps, 30, rng)
    out = []
    for c in columns:
        s = X[c].to_numpy()
        auc = _auc(ya, s)
        null = np.array([_auc(np.roll(ya, k), s) for k in shifts])
        out.append({
            "feature": c,
            "auc": auc,
            "p_value": (1 + np.sum(null >= auc)) / (1 + reps),
            "mean_before_target_day": float(np.expm1(s[ya == 1]).mean()) if ya.any() else np.nan,
            "mean_before_other_day": float(np.expm1(s[ya == 0]).mean()),
        })
    return pd.DataFrame(out).sort_values("auc", ascending=False).reset_index(drop=True)


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


# ------------------------------------------------------------------------- report


def _fmt(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + " --- |" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, floatfmt) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(data_dir: str | Path, out_dir: str | Path, reps: int = 300, seed: int = 0) -> Path:
    """Run both tests on A100 and H100 and write CSVs plus ``report.md`` to ``out_dir``."""
    data_dir = download(data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_rows: list[ModelResult] = []
    uni_tables = []
    count_tables = []
    for fam in ("A100", "H100"):
        counts = daily_counts(safe_load(
            data_dir / f"delta_{fam}_allfileregexbookkeeping_deduped_5s_sanitized.pkl"))
        counts.to_csv(out / f"daily_counts_{fam}.csv")
        totals = pd.DataFrame({
            "xid": counts.columns,
            "events": counts.sum().to_numpy(),
            "days_with_event": (counts > 0).sum().to_numpy(),
        })
        totals["family"] = fam
        count_tables.append(totals)
        for tname, tset in TARGETS.items():
            for h in (1, 3):
                X, y, pcols, bcols = day_table(counts, tset, lookback=7, horizon=h)
                full = X.notna().all(axis=1)
                if full.sum() >= 300:  # enough history for the burst features (A100)
                    X, y = X[full], y[full]
                else:
                    bcols = []
                if y.sum() < 5:
                    continue
                models = {
                    "history only (critical Xids, last 30 d)": ["target_last30d"],
                    "precursor Xids, last 7 d": pcols,
                    "precursors + history": pcols + ["target_last30d"],
                }
                if bcols:
                    models["precursor bursts + history"] = bcols + ["target_last30d"]
                for mname, cols in models.items():
                    auc, pr, p = model_test(X, y, cols, reps=reps, seed=seed)
                    model_rows.append(ModelResult(
                        fam, tname, h, mname, len(y), int(y.sum()), auc, pr,
                        float(y.mean()), p))
                if h == 1:
                    u = univariate_test(X, y, pcols + bcols, reps=reps * 10, seed=seed)
                    u.insert(0, "target", tname)
                    u.insert(0, "family", fam)
                    uni_tables.append(u)

    models_df = pd.DataFrame([r.__dict__ for r in model_rows])
    uni_df = pd.concat(uni_tables, ignore_index=True)
    uni_df["q_value"] = bh_qvalues(uni_df["p_value"].to_numpy())
    totals_df = pd.concat(count_tables, ignore_index=True)
    models_df.to_csv(out / "fleet_day_models.csv", index=False)
    uni_df.to_csv(out / "fleet_day_univariate.csv", index=False)

    casc = []
    for name, label in [
        ("delta_A100_cond_prob_forward_same_dev_id_same_node_pci_deduped_5s_5s_sanitized.pkl",
         "A100 same GPU"),
        ("delta_H100_cond_prob_forward_same_dev_id_same_node_pci_deduped_5s_5s_sanitized.pkl",
         "H100 same GPU"),
        ("delta_A100_cond_prob_forward_same_node_diff_dev_id_pci_deduped_5s_5s_sanitized.pkl",
         "A100 other GPU, same node"),
    ]:
        t = cascade_table(safe_load(data_dir / name))
        t.insert(0, "scope", label)
        casc.append(t)
    casc_df = pd.concat(casc, ignore_index=True)
    casc_df.to_csv(out / "cascades_5s.csv", index=False)

    report = out / "report.md"
    report.write_text(_report(models_df, uni_df, totals_df, casc_df, reps))
    return report


def _report(models: pd.DataFrame, uni: pd.DataFrame, totals: pd.DataFrame,
            casc: pd.DataFrame, reps: int) -> str:
    m = models.copy()
    m = m.rename(columns={"horizon_days": "h (days)", "positive_days": "pos days",
                          "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "base_rate": "base rate",
                          "p_value": "p"})
    m = m.drop(columns=["days"])
    u = uni[uni["p_value"] <= 0.05].copy()
    u = u.rename(columns={"mean_before_target_day": "mean 7d count before critical day",
                          "mean_before_other_day": "mean 7d count before other day",
                          "p_value": "p", "q_value": "q (FDR)"})
    piv = totals.pivot_table(index="xid", columns="family",
                             values=["events", "days_with_event"], fill_value=0).astype(int)
    piv.columns = [f"{a} {b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv.insert(1, "meaning", piv["xid"].map(XID_NAMES).fillna(""))
    c = casc[casc["events"] >= 5].copy()
    return f"""# Real-data check: NCSA Delta Xid logs

Generated by `gpufleet delta`. Data: Zenodo record {ZENODO_RECORD} (Cui et al., SC'25,
CC BY 4.0). Interpretation and caveats:
[docs/delta-real-data.md](../../docs/delta-real-data.md).

Targets: **hard** = Xid {sorted(HARD_FAILURE)} (GPU out of service), **gsp** = Xid
{sorted(GSP_FAILURE)} (firmware hang, reset needed).

## Events in the release

{_fmt(piv)}

## Test 1: do earlier Xids on the partition predict a critical Xid in the next days?

Unit: one partition-day. Features: log counts of each non-critical Xid over the previous
7 days; "bursts" are the same counts relative to each Xid's own 90-day level (A100 only,
since H100 has 146 days in total). Logistic regression, 8 contiguous time blocks with a
10-day gap. `p` is the share of {reps} circular shifts of the label series that scored at
least as well (keeps the autocorrelation of both series). An ROC-AUC of 0.5 is chance.

{_fmt(m)}

### Single Xids (1-day horizon), shown when p <= 0.05

`q (FDR)` corrects for the {len(uni)} single-Xid tests run (Benjamini-Hochberg).
`_burst` features compare the last 7 days with that Xid's own 90-day level.

{_fmt(u) if len(u) else "None."}

## Test 2: does an Xid lead to a critical Xid on the same GPU within 5 seconds?

`followed_within_5s` = any Xid on the same GPU (or another GPU on the node) within 5 s;
`critical_within_5s` = a hard or gsp Xid. Sources with at least 5 events.

{_fmt(c)}
"""
