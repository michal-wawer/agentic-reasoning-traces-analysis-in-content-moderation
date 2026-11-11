"""Statistical evaluation.

* Predictive validity: Pearson r and Spearman rho with two-sided p-values; a Fisher z-transform
  CI for Pearson r:  z = atanh(r),  z +/- z_{1-alpha/2} / sqrt(n - 3),  back-transformed with tanh.
* Routing: AUROC and F1 of the binary high-disagreement label at a fixed operating threshold.
* Operating threshold: the score cut-off that maximizes F1 on the calibration set.
* Supplementary: paired percentile-bootstrap CIs for AUROC / F1 and their differences.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score


@dataclass(frozen=True)
class Correlation:
    n: int
    pearson_r: float
    pearson_p: float
    pearson_ci_low: float
    pearson_ci_high: float
    spearman_rho: float
    spearman_p: float

    def to_dict(self) -> dict:
        return asdict(self)


def fisher_ci(r: float, n: int, level: float = 0.95) -> tuple[float, float]:
    if n < 4 or not np.isfinite(r):
        return math.nan, math.nan
    r = float(np.clip(r, -0.9999999, 0.9999999))
    z = math.atanh(r)
    half = stats.norm.ppf(0.5 + level / 2) / math.sqrt(n - 3)
    return math.tanh(z - half), math.tanh(z + half)


def correlations(signal: np.ndarray, target: np.ndarray, level: float = 0.95) -> Correlation:
    x = np.asarray(signal, dtype=float)
    y = np.asarray(target, dtype=float)
    n = len(x)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        nan = math.nan
        return Correlation(n, nan, nan, nan, nan, nan, nan)
    pr = stats.pearsonr(x, y)  # two-sided by default
    sr = stats.spearmanr(x, y)
    lo, hi = fisher_ci(float(pr.statistic), n, level)
    return Correlation(
        n=n,
        pearson_r=float(pr.statistic),
        pearson_p=float(pr.pvalue),
        pearson_ci_low=lo,
        pearson_ci_high=hi,
        spearman_rho=float(sr.statistic),
        spearman_p=float(sr.pvalue),
    )


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def f1_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    pred = np.asarray(scores, dtype=float) >= threshold
    return float(f1_score(np.asarray(labels, dtype=int), pred.astype(int), zero_division=0))


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return ``(threshold, f1)`` maximizing F1 of ``score >= threshold``.

    The candidates are the distinct observed scores. Ties in F1 go to the *highest* threshold (the
    fewest escalations), which makes the choice deterministic.
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        raise ValueError("empty calibration data")
    best_t, best_f1 = math.inf, -1.0
    for t in np.unique(s)[::-1]:  # descending
        f1 = f1_at(y, s, float(t))
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def _fast_auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties (equal to sklearn's roc_auc_score)."""
    ranks = stats.rankdata(s)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _fast_f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum(pred & (y == 1)))
    denom = int(pred.sum()) + int(y.sum())
    return 2 * tp / denom if denom else 0.0


def paired_bootstrap(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    n_boot: int,
    seed: int,
    level: float = 0.95,
    diffs: list[tuple[str, str]] | None = None,
) -> dict[str, dict[str, float]]:
    """Paired percentile bootstrap over items for AUROC and F1 (at frozen thresholds).

    Returns ``{name: {auroc_low, auroc_high, f1_low, f1_high}}`` and, for each pair (a, b) in
    ``diffs``, an entry ``"a-b"`` with CIs of the AUROC/F1 difference and the one-sided bootstrap
    proportion of resamples where the difference is <= 0.
    """
    y = np.asarray(labels, dtype=int)
    n = len(y)
    rng = np.random.default_rng(seed)
    names = list(scores)
    au = {k: [] for k in names}
    f1 = {k: [] for k in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for k in names:
            sb = scores[k][idx]
            au[k].append(_fast_auroc(yb, sb))
            f1[k].append(_fast_f1(yb, sb >= thresholds[k]))
    lo_q, hi_q = 50 * (1 - level), 50 * (1 + level)
    out: dict[str, dict[str, float]] = {}
    for k in names:
        a, f = np.array(au[k]), np.array(f1[k])
        out[k] = {
            "auroc_low": float(np.percentile(a, lo_q)),
            "auroc_high": float(np.percentile(a, hi_q)),
            "f1_low": float(np.percentile(f, lo_q)),
            "f1_high": float(np.percentile(f, hi_q)),
            "n_valid_resamples": len(a),
        }
    for a_name, b_name in diffs or []:
        da = np.array(au[a_name]) - np.array(au[b_name])
        df = np.array(f1[a_name]) - np.array(f1[b_name])
        out[f"{a_name}-{b_name}"] = {
            "d_auroc_low": float(np.percentile(da, lo_q)),
            "d_auroc_high": float(np.percentile(da, hi_q)),
            "d_auroc_p_le_0": float((da <= 0).mean()),
            "d_f1_low": float(np.percentile(df, lo_q)),
            "d_f1_high": float(np.percentile(df, hi_q)),
            "d_f1_p_le_0": float((df <= 0).mean()),
        }
    return out
