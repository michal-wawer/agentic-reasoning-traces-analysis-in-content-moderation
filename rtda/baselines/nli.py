"""Meaning-equivalence clustering and discrete semantic entropy (B5).

Follows Kuhn et al. (2023): two generations are meaning-equivalent if each entails the other under
an NLI model (bidirectional entailment), and clusters are formed greedily. Provider APIs do not
expose sequence log-probabilities consistently (e.g. none for gpt-5.4-nano reasoning output), so
we use *discrete* semantic entropy (Farquhar et al., Nature 2024), where cluster probabilities are
empirical frequencies: p(c) = |c| / K and SE = -sum_c p(c) ln p(c).
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np


class EntailmentModel(Protocol):
    model_name: str

    def entails(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:  # bool (n,)
        ...


class HFEntailmentModel:
    """Hugging Face NLI classifier (default ``microsoft/deberta-large-mnli``), argmax == entailment."""

    def __init__(self, model_name: str, batch_size: int = 16, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.batch_size = batch_size
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self._torch = torch
        self.device = device
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self._model.eval()
        labels = {i: lab.lower() for i, lab in self._model.config.id2label.items()}
        matches = [i for i, lab in labels.items() if lab.startswith("entail")]
        if len(matches) != 1:
            raise ValueError(f"cannot identify entailment label in {labels}")
        self._entail_idx = matches[0]

    def entails(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        out = []
        with self._torch.inference_mode():
            for start in range(0, len(premises), self.batch_size):
                enc = self._tok(
                    premises[start : start + self.batch_size],
                    hypotheses[start : start + self.batch_size],
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**enc).logits
                out.append((logits.argmax(dim=-1) == self._entail_idx).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=bool)


class CachedEntailment:
    """SQLite cache in front of an :class:`EntailmentModel` (directional premise->hypothesis)."""

    def __init__(self, model: EntailmentModel, path: Path):
        self.model = model
        self.model_name = model.model_name
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS nli (key TEXT PRIMARY KEY, entail INTEGER)")
        self._conn.commit()

    def _key(self, p: str, h: str) -> str:
        return hashlib.sha256(f"{self.model_name}\x00{p}\x00{h}".encode()).hexdigest()

    def entails(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        keys = [self._key(p, h) for p, h in zip(premises, hypotheses)]
        known: dict[str, bool] = {}
        for start in range(0, len(keys), 900):
            chunk = keys[start : start + 900]
            q = f"SELECT key, entail FROM nli WHERE key IN ({','.join('?' * len(chunk))})"
            known.update({k: bool(v) for k, v in self._conn.execute(q, chunk).fetchall()})
        todo = [i for i, k in enumerate(keys) if k not in known]
        if todo:
            res = self.model.entails([premises[i] for i in todo], [hypotheses[i] for i in todo])
            self._conn.executemany(
                "INSERT OR REPLACE INTO nli VALUES (?, ?)",
                [(keys[i], int(r)) for i, r in zip(todo, res)],
            )
            self._conn.commit()
            known.update({keys[i]: bool(r) for i, r in zip(todo, res)})
        return np.array([known[k] for k in keys], dtype=bool)


def sample_text(decision: str, trace: str) -> str:
    return f"Decision: {decision}. {trace}"


def semantic_clusters(
    texts: list[str],
    decisions: list[str],
    nli: EntailmentModel,
    require_same_decision: bool = True,
) -> list[list[int]]:
    """Greedy bidirectional-entailment clustering (Kuhn et al. 2023, Alg. 1)."""
    n = len(texts)
    pairs = [
        (i, j)
        for i in range(n)
        for j in range(n)
        if i != j and (not require_same_decision or decisions[i] == decisions[j])
    ]
    entail: dict[tuple[int, int], bool] = {}
    if pairs:
        res = nli.entails([texts[i] for i, _ in pairs], [texts[j] for _, j in pairs])
        entail = {p: bool(r) for p, r in zip(pairs, res)}
    clusters: list[list[int]] = []
    for i in range(n):
        for cluster in clusters:
            rep = cluster[0]
            if entail.get((rep, i), False) and entail.get((i, rep), False):
                cluster.append(i)
                break
        else:
            clusters.append([i])
    return clusters


def discrete_semantic_entropy(clusters: list[list[int]]) -> float:
    sizes = np.array([len(c) for c in clusters], dtype=float)
    p = sizes / sizes.sum()
    return float(-(p * np.log(p)).sum())
