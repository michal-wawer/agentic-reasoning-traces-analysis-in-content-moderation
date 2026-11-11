"""Measuring Hate Speech corpus: loading, human-disagreement target, and seeded splits.

Per-item human disagreement: ``d_h(c)`` is the *normalized standard deviation* of the
per-annotator ordinal ratings, i.e. ``std(ratings) / max_std`` where ``max_std = (rating_max - rating_min) / 2``
is the largest attainable population std-dev on the rating scale, so ``d_h in [0, 1]``.

Items in the top quartile of ``d_h`` are the "high annotator-disagreement" routing targets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rtda.config import DataConfig, ExperimentConfig

log = logging.getLogger(__name__)

SPLITS = ("train", "cal", "test")


@dataclass(frozen=True)
class Item:
    item_id: str
    text: str
    d_h: float
    high_disagreement: bool
    split: str
    n_annotators: int
    mean_rating: float


def load_raw_annotations(cfg: DataConfig, data_dir: Path) -> pd.DataFrame:
    """Load raw per-annotator rows (HF `datasets`), caching a parquet copy under ``data/raw``."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"{cfg.hf_dataset.replace('/', '__')}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from datasets import load_dataset

    log.info("Downloading %s (%s split) from the Hugging Face Hub", cfg.hf_dataset, cfg.hf_split)
    ds = load_dataset(cfg.hf_dataset, split=cfg.hf_split)
    cols = [cfg.item_id_column, cfg.text_column, cfg.annotator_column, cfg.rating_column]
    df = ds.to_pandas()[cols]
    df.to_parquet(cache, index=False)
    return df


def normalized_std(ratings: np.ndarray, rating_min: float, rating_max: float) -> float:
    """Population std-dev normalised by the maximum attainable std on the rating scale."""
    max_std = (rating_max - rating_min) / 2.0
    if max_std <= 0:
        raise ValueError("rating_max must exceed rating_min")
    return float(np.clip(np.std(ratings, ddof=0) / max_std, 0.0, 1.0))


def aggregate_items(annotations: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Collapse per-annotator rows to one row per item, preserving disagreement statistics."""
    df = annotations.dropna(subset=[cfg.rating_column, cfg.text_column]).copy()
    df[cfg.rating_column] = df[cfg.rating_column].astype(float)
    grouped = df.groupby(cfg.item_id_column)
    ratings = grouped[cfg.rating_column]
    items = pd.DataFrame(
        {
            "item_id": grouped[cfg.item_id_column].first().astype(str),
            "text": grouped[cfg.text_column].first(),
            "n_annotators": ratings.count(),
            "mean_rating": ratings.mean(),
            "majority_rating": ratings.agg(lambda s: round(s.mode().iloc[0])),
            "d_h": ratings.agg(
                lambda s: normalized_std(s.to_numpy(), cfg.rating_min, cfg.rating_max)
            ),
        }
    ).reset_index(drop=True)
    items = items[items["n_annotators"] >= cfg.min_annotators].reset_index(drop=True)
    return items


def label_high_disagreement(items: pd.DataFrame, quantile: float) -> tuple[pd.DataFrame, float]:
    """Attach the binary routing label (top quantile of d_h over the eligible corpus)."""
    threshold = float(items["d_h"].quantile(quantile))
    items = items.copy()
    items["high_disagreement"] = items["d_h"] > threshold
    return items, threshold


def stratified_sample(
    items: pd.DataFrame, n: int, stratify: str | None, rng: np.random.Generator
) -> pd.DataFrame:
    if n > len(items):
        raise ValueError(f"Requested {n} items but only {len(items)} available")
    if stratify is None:
        idx = rng.choice(len(items), size=n, replace=False)
        return items.iloc[np.sort(idx)]
    parts = []
    strata = items.groupby(stratify, sort=True)
    sizes = strata.size()
    alloc = np.floor(sizes / sizes.sum() * n).astype(int)
    remainder = n - int(alloc.sum())
    # Largest-remainder allocation for the leftover slots.
    frac = (sizes / sizes.sum() * n) - alloc
    for key in frac.sort_values(ascending=False).index[:remainder]:
        alloc[key] += 1
    for key, group in strata:
        k = int(min(alloc[key], len(group)))
        if k == 0:
            continue
        idx = rng.choice(len(group), size=k, replace=False)
        parts.append(group.iloc[np.sort(idx)])
    return pd.concat(parts).sort_index()


def make_splits(items: pd.DataFrame, cfg: DataConfig, seed: int) -> pd.DataFrame:
    """Disjoint train / cal / test samples with a fixed seed. Returns items with a ``split`` column."""
    rng = np.random.default_rng(seed)
    remaining = items.copy()
    out = []
    for split, n in (("train", cfg.n_train), ("cal", cfg.n_cal), ("test", cfg.n_test)):
        sampled = stratified_sample(remaining, n, cfg.stratify, rng)
        sampled = sampled.copy()
        sampled["split"] = split
        out.append(sampled)
        remaining = remaining.drop(sampled.index)
    return pd.concat(out).reset_index(drop=True)


def prepare_dataset(cfg: ExperimentConfig, force: bool = False) -> pd.DataFrame:
    """End-to-end: raw annotations -> items with d_h, labels and splits. Persists to ``data/processed``."""
    processed_dir = cfg.paths.data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "items.parquet"
    meta_path = processed_dir / "items.meta.json"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    annotations = load_raw_annotations(cfg.data, cfg.paths.data_dir)
    items = aggregate_items(annotations, cfg.data)
    items, threshold = label_high_disagreement(items, cfg.data.high_disagreement_quantile)
    split_items = make_splits(items, cfg.data, cfg.seed)
    split_items.to_parquet(out_path, index=False)
    meta = {
        "seed": cfg.seed,
        "hf_dataset": cfg.data.hf_dataset,
        "rating_column": cfg.data.rating_column,
        "n_eligible_items": len(items),
        "high_disagreement_threshold": threshold,
        "high_disagreement_quantile": cfg.data.high_disagreement_quantile,
        "split_sizes": split_items["split"].value_counts().to_dict(),
        "positive_rate_by_split": split_items.groupby("split")["high_disagreement"]
        .mean()
        .to_dict(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Prepared dataset: %s", meta)
    return split_items


def load_items(cfg: ExperimentConfig, split: str) -> list[Item]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    df = prepare_dataset(cfg)
    df = df[df["split"] == split]
    return [
        Item(
            item_id=str(r.item_id),
            text=str(r.text),
            d_h=float(r.d_h),
            high_disagreement=bool(r.high_disagreement),
            split=split,
            n_annotators=int(r.n_annotators),
            mean_rating=float(r.mean_rating),
        )
        for r in df.itertuples(index=False)
    ]


def items_to_frame(items: list[Item]) -> pd.DataFrame:
    return pd.DataFrame([i.__dict__ for i in items])
