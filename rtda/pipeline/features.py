"""Per-item feature tables built from stored generations.

``panel_features``   One row per item for a composition: q, A, vote entropy, confidence stats, and
                     similarity stats over the C(k,2) trace pairs (Phi, B1, B2, B6).
``neutral_features`` One row per item from the K neutral samples (B3, B4, B5).

Items where any required generation is missing or failed to parse get ``complete = False`` and NaN
features. Evaluation drops them explicitly and reports how many were dropped.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rtda.agents.compositions import Composition, Unit
from rtda.baselines.nli import (
    EntailmentModel,
    discrete_semantic_entropy,
    sample_text,
    semantic_clusters,
)
from rtda.config import ExperimentConfig
from rtda.embeddings.encoder import EmbeddingCache
from rtda.embeddings.similarity import pairwise_cosine, similarity_stats
from rtda.taxonomy.states import agreement_indicator, conclusion_agreement, vote_entropy

ITEM_COLUMNS = ["item_id", "split", "d_h", "high_disagreement"]


def _unit_lookup(generations: pd.DataFrame) -> dict[tuple[str, str, int], pd.Series]:
    ok = generations[generations["status"] == "ok"]
    return {(r.item_id, r.role, int(r.sample_index)): r for r in ok.itertuples(index=False)}


def panel_features(
    cfg: ExperimentConfig,
    items: pd.DataFrame,
    generations: pd.DataFrame,
    composition: Composition,
    embeddings: EmbeddingCache,
    max_seq_length: int,
) -> pd.DataFrame:
    lookup = _unit_lookup(generations)
    rows, traces, trace_owner = [], [], []
    for item in items.itertuples(index=False):
        recs = [lookup.get((item.item_id, u.role, u.sample_index)) for u in composition.units]
        row = {c: getattr(item, c) for c in ITEM_COLUMNS}
        row.update(composition=composition.name, k=composition.k)
        row["complete"] = all(r is not None for r in recs)
        if row["complete"]:
            decisions = [r.decision for r in recs]
            conf = np.array([r.confidence for r in recs], dtype=float)
            q = conclusion_agreement(decisions)
            row.update(
                n_remove=sum(d == "Remove" for d in decisions),
                q=q,
                agree=bool(agreement_indicator(q, cfg.taxonomy.tau_a)),
                vote_entropy=vote_entropy(decisions),
                conf_mean=float(conf.mean()),
                conf_std=float(conf.std(ddof=0)),
                decisions=json.dumps(decisions),
            )
            traces.extend(r.trace for r in recs)
            trace_owner.append((len(rows), len(recs)))
        rows.append(row)

    emb, n_tokens = embeddings.embed(traces) if traces else (np.zeros((0, 0)), np.zeros(0))
    offset = 0
    for row_idx, n in trace_owner:
        e, toks = emb[offset : offset + n], n_tokens[offset : offset + n]
        offset += n
        pairs = pairwise_cosine(e)
        rows[row_idx].update(similarity_stats(pairs))
        rows[row_idx]["pairwise_sims"] = json.dumps([round(float(p), 6) for p in pairs])
        rows[row_idx]["max_trace_tokens"] = int(toks.max())
        rows[row_idx]["n_truncated_traces"] = int((toks > max_seq_length).sum())
    return pd.DataFrame(rows)


def neutral_features(
    cfg: ExperimentConfig,
    items: pd.DataFrame,
    generations: pd.DataFrame,
    units: tuple[Unit, ...],
    nli: EntailmentModel,
) -> pd.DataFrame:
    lookup = _unit_lookup(generations)
    rows = []
    for item in items.itertuples(index=False):
        recs = [lookup.get((item.item_id, u.role, u.sample_index)) for u in units]
        row = {c: getattr(item, c) for c in ITEM_COLUMNS}
        row["neutral_complete"] = all(r is not None for r in recs)
        if row["neutral_complete"]:
            decisions = [r.decision for r in recs]
            texts = [sample_text(r.decision, r.trace) for r in recs]
            clusters = semantic_clusters(
                texts, decisions, nli, cfg.baselines.nli_require_same_decision
            )
            row.update(
                b3_confidence=float(recs[cfg.baselines.b3_sample_index].confidence),
                b4_majority_rate=conclusion_agreement(decisions),
                b5_semantic_entropy=discrete_semantic_entropy(clusters),
                b5_n_clusters=len(clusters),
            )
        rows.append(row)
    return pd.DataFrame(rows)
