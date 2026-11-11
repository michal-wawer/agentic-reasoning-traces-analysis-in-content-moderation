"""Scalar structural signal Phi and Phi-based routing.

The taxonomy assigns a categorical state. Correlations and AUROC need a scalar, so we define

    Phi(x) = rank(state(x)) + (1 - eps) * u(x),     u(x) = (H(x) + clip(s_mean(x), 0, 1)) / 2

* ``rank`` is the position of the state in an ascending escalation-priority order. The order is
  estimated on the **calibration set** as the ordering of states by mean human disagreement d_h.
* ``u(x)`` in [0, 1] breaks ties within a state: higher vote entropy H (more split conclusions)
  and higher reasoning similarity (more "convergent") give higher priority. Because
  (1 - eps) * u < 1, the state ordering always dominates.
* **Routing** escalates the top-``m`` states in that order, i.e. ``Phi >= 4 - m``. ``m`` is chosen
  together with tau_s on the calibration set.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rtda.config import STATES


def tiebreak(vote_entropy: np.ndarray, s_mean: np.ndarray) -> np.ndarray:
    h = np.asarray(vote_entropy, dtype=float)
    s = np.clip(np.asarray(s_mean, dtype=float), 0.0, 1.0)
    return 0.5 * (h + s)


def state_order_by_mean(
    states: np.ndarray, d_h: np.ndarray, fallback_order: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Order states ascending by mean d_h. Returns ``(order, missing_states)``.

    States that don't occur are inserted at their position in ``fallback_order``. Ties in mean d_h
    are broken by ``fallback_order``.
    """
    states = np.asarray(states)
    d_h = np.asarray(d_h, dtype=float)
    fb = {s: i for i, s in enumerate(fallback_order)}
    present = [s for s in STATES if np.any(states == s)]
    missing = [s for s in fallback_order if s not in present]
    order = sorted(present, key=lambda s: (float(d_h[states == s].mean()), fb[s]))
    for s in missing:
        order.insert(min(fb[s], len(order)), s)
    return order, missing


def phi_score(states: np.ndarray, u: np.ndarray, order: Sequence[str], eps: float) -> np.ndarray:
    rank = {s: i for i, s in enumerate(order)}
    r = np.array([rank[s] for s in np.asarray(states)], dtype=float)
    return r + (1.0 - eps) * np.asarray(u, dtype=float)


def phi_route(states: np.ndarray, order: Sequence[str], m: int) -> np.ndarray:
    if not 1 <= m <= len(order):
        raise ValueError("m must be in [1, number of states]")
    escalated = set(order[-m:])
    return np.array([s in escalated for s in np.asarray(states)], dtype=bool)
