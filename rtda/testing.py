"""Lightweight, fully offline stand-ins for the embedding and NLI models.

**Only for smoke tests and unit tests.** They let the whole pipeline run without downloading
all-mpnet-base-v2 or DeBERTa-MNLI. Any run that uses them records ``"fake_models": true`` in its
manifest, and its numbers must never be reported.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN = re.compile(r"[a-z]+")


class HashingEncoder:
    """Bag-of-words feature hashing (L2-normalised), so similar texts get similar vectors."""

    model_name = "fake/hashing-encoder-256"
    max_seq_length = 384

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _TOKEN.findall(t.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "big")
                out[i, h % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            out[i] = out[i] / norm if norm > 0 else out[i]
            if norm == 0:
                out[i, 0] = 1.0
        return out

    def n_tokens(self, texts: list[str]) -> list[int]:
        return [len(_TOKEN.findall(t.lower())) + 2 for t in texts]


class JaccardEntailment:
    """'Entailment' if the token Jaccard overlap is at least ``threshold``."""

    model_name = "fake/jaccard-entailment"

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def entails(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        res = []
        for p, h in zip(premises, hypotheses):
            a, b = set(_TOKEN.findall(p.lower())), set(_TOKEN.findall(h.lower()))
            res.append(len(a & b) / max(len(a | b), 1) >= self.threshold)
        return np.array(res, dtype=bool)
