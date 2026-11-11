"""Pairwise cosine similarity of reasoning traces and per-item aggregates.

For agents i < j: s_ij = e_i . e_j / (||e_i|| ||e_j||). Per item:
  s_mean = mean over the C(N,2) pairs (mean similarity, S(x)),
  s_min  = minimum pairwise similarity,
  s_max  = maximum pairwise similarity (B6 feature only),
  s_var  = population variance of the pairwise similarities (sigma_s in the paper).
"""

from __future__ import annotations

import numpy as np


def pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
    """Return the C(N,2) upper-triangle cosine similarities for an (N, d) array, in (i<j) order."""
    e = np.asarray(embeddings, dtype=np.float64)
    if e.ndim != 2 or e.shape[0] < 2:
        raise ValueError("need an (N, d) array with N >= 2")
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding")
    e = e / norms
    sim = e @ e.T
    iu = np.triu_indices(e.shape[0], k=1)
    return np.clip(sim[iu], -1.0, 1.0)


def similarity_stats(pairwise: np.ndarray) -> dict[str, float]:
    p = np.asarray(pairwise, dtype=np.float64)
    return {
        "s_mean": float(p.mean()),
        "s_min": float(p.min()),
        "s_max": float(p.max()),
        "s_var": float(p.var(ddof=0)),
    }
