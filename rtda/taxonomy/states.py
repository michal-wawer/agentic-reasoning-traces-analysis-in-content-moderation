"""Four-state (dis)agreement taxonomyagreement taxonomy").

q(x)      = max_y (1/N) sum_i 1[y_i = y]          conclusion agreement (majority rate)
A(x)      = 1[q(x) >= tau_a]                       binarised agreement
S(x)      = s_mean(x)                              mean pairwise trace similarity
CA: S >= tau_s and A = 1     DA: S < tau_s and A = 1
CD: S >= tau_s and A = 0     DD: S < tau_s and A = 0
"""

from __future__ import annotations

import numpy as np

# Absolute tolerance for comparisons at the threshold, so that e.g. q = 4/5 counts as >= 0.80 even
# with floating-point representation error.
_ATOL = 1e-9

REMOVE, KEEP = "Remove", "Keep"


def conclusion_agreement(decisions: list[str]) -> float:
    """q(x): fraction of agents that agree with the majority decision."""
    if not decisions:
        raise ValueError("no decisions")
    n_remove = sum(d == REMOVE for d in decisions)
    if any(d not in (REMOVE, KEEP) for d in decisions):
        raise ValueError(f"invalid decisions: {decisions}")
    p = n_remove / len(decisions)
    return max(p, 1.0 - p)


def vote_entropy(decisions: list[str]) -> float:
    """Shannon entropy (bits) of the binary Keep/Remove vote distribution, in [0, 1] (B2)."""
    p = sum(d == REMOVE for d in decisions) / len(decisions)
    return float(-sum(x * np.log2(x) for x in (p, 1.0 - p) if x > 0))


def agreement_indicator(q: np.ndarray | float, tau_a: float) -> np.ndarray:
    return np.asarray(q, dtype=float) + _ATOL >= tau_a


def classify_states(s_mean: np.ndarray, agree: np.ndarray, tau_s: float) -> np.ndarray:
    """Vectorised four-state classification. Returns an array of 'CA'/'DA'/'CD'/'DD' strings."""
    s_mean = np.asarray(s_mean, dtype=float)
    agree = np.asarray(agree, dtype=bool)
    if s_mean.shape != agree.shape:
        raise ValueError("s_mean and agree must have the same shape")
    convergent = s_mean + _ATOL >= tau_s
    out = np.empty(s_mean.shape, dtype=object)
    out[convergent & agree] = "CA"
    out[~convergent & agree] = "DA"
    out[convergent & ~agree] = "CD"
    out[~convergent & ~agree] = "DD"
    return out
