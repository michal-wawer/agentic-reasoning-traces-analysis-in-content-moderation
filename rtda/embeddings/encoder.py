"""Reasoning-trace embedding with all-mpnet-base-v2.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np

from rtda.config import EmbeddingsConfig


class TextEncoder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray:  # (n, d), L2-normalised float32
        ...

    def n_tokens(self, texts: list[str]) -> list[int]: ...

    @property
    def max_seq_length(self) -> int: ...


class SentenceTransformerEncoder:
    def __init__(self, cfg: EmbeddingsConfig):
        from sentence_transformers import SentenceTransformer

        self.model_name = cfg.model
        self.batch_size = cfg.batch_size
        self._model = SentenceTransformer(cfg.model, device=cfg.device)

    @property
    def max_seq_length(self) -> int:
        return int(self._model.max_seq_length)

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
        )
        return emb.astype(np.float32)

    def n_tokens(self, texts: list[str]) -> list[int]:
        tok = self._model.tokenizer
        return [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts]


class EmbeddingCache:
    """Wrap an encoder with a persistent SQLite cache of normalised embeddings."""

    def __init__(self, encoder: TextEncoder, path: Path):
        self.encoder = encoder
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, dim INTEGER, n_tokens INTEGER,"
            " vec BLOB)"
        )
        self._conn.commit()

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.encoder.model_name}\x00{text}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(embeddings (n, d), token_counts (n,))``, computing only uncached texts."""
        keys = [self._key(t) for t in texts]
        found: dict[str, tuple[np.ndarray, int]] = {}
        unique_keys = list(dict.fromkeys(keys))
        for start in range(0, len(unique_keys), 900):
            chunk = unique_keys[start : start + 900]
            rows = self._conn.execute(
                f"SELECT key, dim, n_tokens, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})",
                chunk,
            ).fetchall()
            for key, dim, n_tok, blob in rows:
                found[key] = (np.frombuffer(blob, dtype=np.float32, count=dim), int(n_tok))
        missing = {}
        for key, text in zip(keys, texts):
            if key not in found:
                missing.setdefault(key, text)
        if missing:
            m_keys, m_texts = list(missing), list(missing.values())
            vecs = self.encoder.encode(m_texts)
            n_toks = self.encoder.n_tokens(m_texts)
            self._conn.executemany(
                "INSERT OR REPLACE INTO emb VALUES (?, ?, ?, ?)",
                [
                    (k, int(v.shape[0]), int(n), v.astype(np.float32).tobytes())
                    for k, v, n in zip(m_keys, vecs, n_toks)
                ],
            )
            self._conn.commit()
            for k, v, n in zip(m_keys, vecs, n_toks):
                found[k] = (v.astype(np.float32), int(n))
        emb = np.stack([found[k][0] for k in keys]) if keys else np.zeros((0, 0), np.float32)
        return emb, np.array([found[k][1] for k in keys], dtype=int)

    def close(self) -> None:
        self._conn.close()
