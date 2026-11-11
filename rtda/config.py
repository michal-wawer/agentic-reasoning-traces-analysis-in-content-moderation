"""Typed experiment configuration (loaded from YAML) and seed utilities.

All experiment thresholds and hyperparameters live here as named fields so that
nothing is hardcoded inline in the pipeline code.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator

State = Literal["CA", "DA", "CD", "DD"]
STATES: tuple[State, ...] = ("CA", "DA", "DD", "CD")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")
    runs_dir: Path = Path("runs")
    results_dir: Path = Path("results")


class DataConfig(BaseModel):
    hf_dataset: str = "ucberkeley-dlab/measuring-hate-speech"
    hf_split: str = "train"
    item_id_column: str = "comment_id"
    text_column: str = "text"
    annotator_column: str = "annotator_id"
    rating_column: str = "hatespeech"
    rating_min: float = 0.0
    rating_max: float = 2.0
    min_annotators: int = 3
    n_train: int = 200
    n_cal: int = 200
    n_test: int = 1000
    stratify: str | None = "majority_rating"
    high_disagreement_quantile: float = 0.75


class LLMConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    headline_model: str = "openai/gpt-5.4-nano"
    ablation_models: list[str] = Field(
        default_factory=lambda: ["anthropic/claude-sonnet-4.5", "minimax/minimax-m3"]
    )
    temperature: float | None = 0.7
    reasoning_effort: str | None = None
    max_tokens: int = 4000
    concurrency: int = 8
    max_retries: int = 4
    parse_retries: int = 2
    timeout_s: float = 120.0
    provider_order: list[str] = Field(default_factory=list)


class PromptsConfig(BaseModel):
    path: Path = Path("configs/prompts/v1.yaml")
    trace_fields: list[str] = Field(
        default_factory=lambda: ["interpretation", "considerations", "trade_offs", "judgment"]
    )


class PanelConfig(BaseModel):
    perspectives: list[str] = Field(
        default_factory=lambda: [
            "harm_focused",
            "context_sensitive",
            "community_norms",
            "free_expression",
            "legal_framework",
        ]
    )


class EmbeddingsConfig(BaseModel):
    model: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 32
    device: str | None = None


class TaxonomyConfig(BaseModel):
    tau_a: float = 0.80
    # Candidate similarity thresholds evaluated on the calibration set only.
    tau_s_candidates: list[float] = Field(
        default_factory=lambda: [0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85]
    )
    # Number of highest-priority states escalated by Phi routing (selected on calibration).
    escalate_m_candidates: list[int] = Field(default_factory=lambda: [1, 2, 3])
    # Phi score = rank(state) + (1 - eps) * u(x), u in [0, 1]; eps keeps states strictly ordered.
    tiebreak_eps: float = 1e-6
    # Order used only when a state is empty on the calibration set (ascending priority).
    fallback_state_order: list[State] = Field(default_factory=lambda: ["CA", "DA", "DD", "CD"])

    @model_validator(mode="after")
    def _check(self) -> TaxonomyConfig:
        if sorted(self.fallback_state_order) != ["CA", "CD", "DA", "DD"]:
            raise ValueError("fallback_state_order must be a permutation of CA, DA, CD, DD")
        if not 0.0 < self.tau_a <= 1.0:
            raise ValueError("tau_a must be in (0, 1]")
        return self


class B6Config(BaseModel):
    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 7
    min_child_samples: int = 10
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    features: list[str] = Field(
        default_factory=lambda: [
            "s_mean",
            "s_min",
            "s_max",
            "s_var",
            "q",
            "vote_entropy",
            "conf_mean",
            "conf_std",
        ]
    )


class BaselinesConfig(BaseModel):
    # B3/B4/B5 (and the no-differentiation control) use a neutral, undifferentiated prompt.
    single_agent_perspective: str = "neutral"
    k_samples: int = 10
    b3_sample_index: int = 0
    nli_model: str = "microsoft/deberta-large-mnli"
    nli_batch_size: int = 16
    nli_device: str | None = None
    # Two samples with different Keep/Remove decisions are never meaning-equivalent.
    nli_require_same_decision: bool = True
    b6: B6Config = Field(default_factory=B6Config)


class EvaluationConfig(BaseModel):
    ci_level: float = 0.95
    n_bootstrap: int = 10_000


class AblationsConfig(BaseModel):
    compositions: list[str] = Field(
        default_factory=lambda: [
            "full_5",
            "k3_drop_legal_community",
            "k7_add_procedural_victim",
            "random_swap",
            "drop_harm_focused",
            "drop_free_expression",
            "no_differentiation",
        ]
    )
    extra_perspectives: list[str] = Field(default_factory=lambda: ["procedural", "victim_centered"])


class ExperimentConfig(BaseModel):
    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    panel: PanelConfig = Field(default_factory=PanelConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig)
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    ablations: AblationsConfig = Field(default_factory=AblationsConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> ExperimentConfig:
        cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)

    def fingerprint(self) -> str:
        """SHA-256 of the canonical JSON form of the configuration."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def set_global_seed(seed: int) -> None:
    """Fix Python / NumPy seeds (and PYTHONHASHSEED for subprocesses)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def derive_seed(base_seed: int, *components: str | int) -> int:
    """Deterministically derive a sub-seed from a base seed and string/int components.

    Used to give every (model, item, agent, sample) call a stable, logged decoding seed.
    """
    payload = "|".join([str(base_seed), *map(str, components)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31 - 1)
