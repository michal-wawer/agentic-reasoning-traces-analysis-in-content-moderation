"""Baselines B1-B6 behind one interface.

Every scorer maps a per-item feature table to an uncertainty score where **higher means more likely
to be a high annotator-disagreement item**. That lets the structural signal Phi and all baselines
go through the same calibration and evaluation harness.

    B1 divergence-only        1 - s_mean
    B2 vote entropy           H(votes of the panel agents)
    B3 verbalized confidence  1 - gamma of one neutral single-agent generation (Lin et al. 2022)
    B4 self-consistency       1 - majority rate over K neutral samples (Wang et al. 2023)
    B5 semantic uncertainty   discrete semantic entropy over K neutral samples (Kuhn et al. 2023)
    B6 supervised             LightGBM P(high disagreement), trained on the train split only
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtda.config import B6Config


class LeakageError(RuntimeError):
    """Raised when data from the wrong split reaches a fitting/calibration routine."""


def require_split(df: pd.DataFrame, split: str, what: str) -> None:
    splits = set(df["split"].unique())
    if splits != {split}:
        raise LeakageError(f"{what} must only receive '{split}' items, got splits {sorted(splits)}")


@dataclass
class ColumnScorer:
    """A baseline whose score is a fixed transform of one precomputed feature column."""

    method_id: str
    label: str
    column: str
    transform: str = "identity"  # or "one_minus"
    supervised: bool = False

    def score(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.column].to_numpy(dtype=float)
        return 1.0 - x if self.transform == "one_minus" else x


def _lgb_fit(params: dict[str, Any], x: np.ndarray, y: np.ndarray, model_path: str) -> str:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(**params).fit(x, y)
    model.booster_.save_model(model_path)
    return model_path


def _lgb_predict(model_path: str, x: np.ndarray) -> np.ndarray:
    import lightgbm as lgb

    return lgb.Booster(model_file=model_path).predict(x)  # binary objective -> P(y = 1)


def _run_isolated(fn: Any, *args: Any) -> Any:
    """Run ``fn`` in a freshly spawned interpreter.

    LightGBM and PyTorch each ship an OpenMP runtime. Once both are loaded into one process, LightGBM
    segfaults on macOS. The feature and embedding stages load PyTorch, so B6 always runs in a clean
    child process that never imports it.
    """
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return pool.submit(fn, *args).result()


class LightGBMScorer:
    """B6: the only baseline trained with human-disagreement labels (train split only).

    The fitted booster is persisted as a text model file (``b6_model.txt``) that is released with the run.
    """

    method_id = "B6"
    label = "B6 Supervised"
    supervised = True

    def __init__(self, cfg: B6Config, seed: int):
        self.cfg = cfg
        self.seed = seed
        self.model_path: Path | None = None

    def params(self) -> dict[str, Any]:
        return {
            "objective": "binary",
            "n_estimators": self.cfg.n_estimators,
            "learning_rate": self.cfg.learning_rate,
            "num_leaves": self.cfg.num_leaves,
            "min_child_samples": self.cfg.min_child_samples,
            "subsample": self.cfg.subsample,
            "subsample_freq": self.cfg.subsample_freq,
            "colsample_bytree": self.cfg.colsample_bytree,
            "random_state": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "n_jobs": 1,
            "verbose": -1,
        }

    def fit(self, train: pd.DataFrame, model_path: Path) -> LightGBMScorer:
        require_split(train, "train", "B6.fit")
        y = train["high_disagreement"].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError("B6 training data contains a single class")
        x = train[self.cfg.features].to_numpy(dtype=float)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        _run_isolated(_lgb_fit, self.params(), x, y, str(model_path))
        self.model_path = Path(model_path)
        return self

    @classmethod
    def load(cls, cfg: B6Config, seed: int, path: Path) -> LightGBMScorer:
        scorer = cls(cfg, seed)
        scorer.model_path = Path(path)
        return scorer

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if self.model_path is None:
            raise RuntimeError("B6 must be fitted on the train split before scoring")
        x = df[self.cfg.features].to_numpy(dtype=float)
        return np.asarray(_run_isolated(_lgb_predict, str(self.model_path), x))


def unsupervised_baselines(include_single_agent: bool = True) -> list[ColumnScorer]:
    scorers = [
        ColumnScorer("B1", "B1 Divergence-only", "s_mean", "one_minus"),
        ColumnScorer("B2", "B2 Vote entropy", "vote_entropy"),
    ]
    if include_single_agent:
        scorers += [
            ColumnScorer("B3", "B3 Verbalized conf.", "b3_confidence", "one_minus"),
            ColumnScorer("B4", "B4 Self-consistency", "b4_majority_rate", "one_minus"),
            ColumnScorer("B5", "B5 Semantic uncertainty", "b5_semantic_entropy"),
        ]
    return scorers
