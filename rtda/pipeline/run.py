"""End-to-end orchestration: generate -> features -> calibrate -> evaluate -> tables.

Each stage reads the previous stage's persisted outputs and is idempotent: generation resumes
from ``generations.jsonl``, embeddings and NLI calls are cached, and
calibration refuses to overwrite frozen parameters unless forced.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from rtda.agents.compositions import (
    HEADLINE,
    Composition,
    Unit,
    get_composition,
    neutral_sample_units,
)
from rtda.agents.panel import GenerationStore, generate
from rtda.agents.prompts import load_prompts
from rtda.baselines.scorers import LightGBMScorer, unsupervised_baselines
from rtda.config import ExperimentConfig, set_global_seed
from rtda.evaluation import reporting
from rtda.evaluation.calibration import (
    calibrate_phi,
    calibrate_thresholds,
    item_ids_sha256,
    write_frozen,
)
from rtda.evaluation.test_eval import evaluate_test, json_safe, load_frozen
from rtda.llm import make_backend
from rtda.pipeline.features import neutral_features, panel_features
from rtda.pipeline.manifest import write_manifest

log = logging.getLogger(__name__)

Stage = Literal["headline", "models", "compositions"]
STAGES: tuple[Stage, ...] = ("headline", "models", "compositions")


def slug(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


@dataclass(frozen=True)
class RunOptions:
    mock: bool = False  # deterministic MockBackend instead of OpenRouter
    fake_models: bool = False  # hashing encoder + Jaccard "NLI" (offline smoke tests only)
    limit: int | None = None  # first N items per split (smoke runs)

    @property
    def tag(self) -> str:
        tag = "mock" if self.mock else "live"
        if self.limit:
            tag += f"-limit{self.limit}"
        if self.fake_models:
            tag += "-fake"
        return tag


@dataclass(frozen=True)
class GenerationJob:
    model: str
    units: tuple[Unit, ...]
    splits: tuple[str, ...]
    purpose: str


class Experiment:
    def __init__(self, cfg: ExperimentConfig, opts: RunOptions, items: pd.DataFrame | None = None):
        set_global_seed(cfg.seed)
        self.cfg = cfg
        self.opts = opts
        self.run_dir = Path(cfg.paths.runs_dir) / opts.tag
        self.results_dir = Path(cfg.paths.results_dir) / opts.tag
        self.prompts = load_prompts(cfg.prompts)
        self._items = items
        self._backend = None
        self._embeddings = None
        self._nli = None
        self._stores: dict[str, GenerationStore] = {}
        self._capabilities: dict[str, Any] = {}

    # ------------------------------------------------------------------ resources
    def items(self, splits: tuple[str, ...]) -> pd.DataFrame:
        if self._items is None:
            from rtda.data.measuring_hate_speech import prepare_dataset

            df = prepare_dataset(self.cfg)
            df["item_id"] = df["item_id"].astype(str)
            self._items = df
        parts = []
        for split in splits:
            part = self._items[self._items["split"] == split]
            parts.append(part.head(self.opts.limit) if self.opts.limit else part)
        return pd.concat(parts, ignore_index=True)

    @property
    def backend(self):
        if self._backend is None:
            self._backend = make_backend(self.cfg.llm, mock=self.opts.mock)
        return self._backend

    @property
    def embeddings(self):
        if self._embeddings is None:
            from rtda.embeddings.encoder import EmbeddingCache, SentenceTransformerEncoder

            if self.opts.fake_models:
                from rtda.testing import HashingEncoder

                encoder = HashingEncoder()
            else:
                encoder = SentenceTransformerEncoder(self.cfg.embeddings)
            path = (
                Path(self.cfg.paths.cache_dir) / "embeddings" / f"{slug(encoder.model_name)}.sqlite"
            )
            self._embeddings = EmbeddingCache(encoder, path)
        return self._embeddings

    @property
    def nli(self):
        if self._nli is None:
            from rtda.baselines.nli import CachedEntailment, HFEntailmentModel

            if self.opts.fake_models:
                from rtda.testing import JaccardEntailment

                model = JaccardEntailment()
            else:
                b = self.cfg.baselines
                model = HFEntailmentModel(b.nli_model, b.nli_batch_size, b.nli_device)
            path = Path(self.cfg.paths.cache_dir) / "nli" / f"{slug(model.model_name)}.sqlite"
            self._nli = CachedEntailment(model, path)
        return self._nli

    def model_dir(self, model: str) -> Path:
        return self.run_dir / slug(model)

    def comp_dir(self, model: str, composition: str) -> Path:
        return self.model_dir(model) / composition

    def store(self, model: str) -> GenerationStore:
        if model not in self._stores:
            self._stores[model] = GenerationStore(self.model_dir(model) / "generations.jsonl")
        return self._stores[model]

    def is_headline(self, model: str, composition: str) -> bool:
        return model == self.cfg.llm.headline_model and composition == HEADLINE

    def panel_splits(self, model: str, composition: str) -> tuple[str, ...]:
        # B6 (headline only) needs panel features on the train split.
        return ("train", "cal", "test") if self.is_headline(model, composition) else ("cal", "test")

    # ------------------------------------------------------------------ plans
    def generation_jobs(self, stage: Stage) -> list[GenerationJob]:
        cfg, head = self.cfg, self.cfg.llm.headline_model
        full = get_composition(HEADLINE, cfg)
        if stage == "headline":
            return [
                GenerationJob(head, full.units, ("train", "cal", "test"), "panel"),
                GenerationJob(
                    head, neutral_sample_units(cfg), ("cal", "test"), "neutral K samples"
                ),
            ]
        if stage == "models":
            return [
                GenerationJob(m, full.units, ("cal", "test"), "panel")
                for m in cfg.llm.ablation_models
            ]
        units: dict[Unit, None] = {}
        for name in cfg.ablations.compositions:
            units.update(dict.fromkeys(get_composition(name, cfg).units))
        return [GenerationJob(head, tuple(units), ("cal", "test"), "composition units")]

    def evaluation_targets(self, stage: Stage) -> list[tuple[str, str]]:
        head = self.cfg.llm.headline_model
        if stage == "headline":
            return [(head, HEADLINE)]
        if stage == "models":
            return [(m, HEADLINE) for m in self.cfg.llm.ablation_models]
        return [(head, c) for c in self.cfg.ablations.compositions if c != HEADLINE]

    # ------------------------------------------------------------------ stages
    def generate(self, stage: Stage) -> list[dict[str, Any]]:
        async def run_all() -> list[dict[str, Any]]:
            out = []
            jobs = self.generation_jobs(stage)
            try:
                # Fail fast on unknown model IDs, and record provider-reported capabilities.
                for model in dict.fromkeys(j.model for j in jobs):
                    self._capabilities[model] = self.backend.capabilities(model)
                for job in jobs:
                    stats = await generate(
                        self.backend,
                        self.cfg,
                        self.prompts,
                        job.model,
                        self.items(job.splits),
                        job.units,
                        self.store(job.model),
                        desc=f"{slug(job.model)}:{job.purpose}",
                    )
                    out.append(
                        {**asdict(job), "units": [u.label for u in job.units], **asdict(stats)}
                    )
            finally:
                await self.backend.aclose()
                self._backend = None
            return out

        results = asyncio.run(run_all())
        self._manifest(f"generate-{stage}", {"generation_jobs": results})
        return results

    def build_features(self, stage: Stage) -> None:
        for model, comp_name in self.evaluation_targets(stage):
            comp = get_composition(comp_name, self.cfg)
            items = self.items(self.panel_splits(model, comp_name))
            feats = panel_features(
                self.cfg,
                items,
                self.store(model).frame(),
                comp,
                self.embeddings,
                self.embeddings.encoder.max_seq_length,
            )
            out = self.comp_dir(model, comp_name) / "features.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            feats.to_parquet(out, index=False)
            log.info("wrote %s (%d items, %d complete)", out, len(feats), feats["complete"].sum())
        if stage == "headline":
            model = self.cfg.llm.headline_model
            feats = neutral_features(
                self.cfg,
                self.items(("cal", "test")),
                self.store(model).frame(),
                neutral_sample_units(self.cfg),
                self.nli,
            )
            feats.to_parquet(self.model_dir(model) / "neutral_features.parquet", index=False)
        self._manifest(f"features-{stage}", {})

    def _features(self, model: str, comp_name: str) -> tuple[pd.DataFrame, dict[str, int]]:
        feats = pd.read_parquet(self.comp_dir(model, comp_name) / "features.parquet")
        mask = feats["complete"].astype(bool)
        if self.is_headline(model, comp_name):
            neutral = pd.read_parquet(self.model_dir(model) / "neutral_features.parquet")
            keep = [c for c in neutral.columns if c.startswith(("b3_", "b4_", "b5_", "neutral_"))]
            feats = feats.merge(neutral[["item_id", *keep]], on="item_id", how="left")
            # train items have no neutral samples (B3-B5 are not trained); only require them on cal/test
            needs_neutral = feats["split"] != "train"
            mask = feats["complete"].astype(bool) & (
                ~needs_neutral | feats["neutral_complete"].fillna(False).astype(bool)
            )
        exclusions = {
            f"{split}_excluded_incomplete": int((~mask[feats["split"] == split]).sum())
            for split in feats["split"].unique()
        }
        return feats[mask].reset_index(drop=True), exclusions

    def _scorers(self, model: str, comp_name: str):
        return unsupervised_baselines(include_single_agent=self.is_headline(model, comp_name))

    def calibrate(self, stage: Stage, force: bool = False) -> None:
        for model, comp_name in self.evaluation_targets(stage):
            feats, exclusions = self._features(model, comp_name)
            cal = feats[feats["split"] == "cal"].reset_index(drop=True)
            scorers = self._scorers(model, comp_name)
            cal_dir = self.comp_dir(model, comp_name) / "calibration"
            payload: dict[str, Any] = {
                "model": model,
                "composition": comp_name,
                "calibration_item_ids": sorted(cal["item_id"]),
                "calibration_item_ids_sha256": item_ids_sha256(list(cal["item_id"])),
                "n_cal_complete": len(cal),
                "exclusions": exclusions,
                "phi": calibrate_phi(self.cfg, cal),
            }
            if self.is_headline(model, comp_name):
                train = feats[feats["split"] == "train"].reset_index(drop=True)
                b6 = LightGBMScorer(self.cfg.baselines.b6, self.cfg.seed).fit(
                    train, cal_dir / "b6_model.txt"
                )
                payload["b6"] = {"n_train": len(train), "model_file": "b6_model.txt"}
                scorers = [*scorers, b6]
            payload["methods"] = calibrate_thresholds(cal, scorers)
            write_frozen(cal_dir / "frozen_params.json", json_safe(payload), force=force)
            log.info("frozen %s/%s: tau_s=%s", model, comp_name, payload["phi"]["tau_s"])
        self._manifest(f"calibrate-{stage}", {"force": force})

    def evaluate(self, stage: Stage, bootstrap: bool = True) -> None:
        for model, comp_name in self.evaluation_targets(stage):
            comp_dir = self.comp_dir(model, comp_name)
            frozen = load_frozen(comp_dir / "calibration" / "frozen_params.json")
            feats, exclusions = self._features(model, comp_name)
            test = feats[feats["split"] == "test"].reset_index(drop=True)
            scorers = self._scorers(model, comp_name)
            if self.is_headline(model, comp_name):
                b6_path = comp_dir / "calibration" / frozen["b6"]["model_file"]
                scorers = [
                    *scorers,
                    LightGBMScorer.load(self.cfg.baselines.b6, self.cfg.seed, b6_path),
                ]
            comp = get_composition(comp_name, self.cfg)
            result, per_item = evaluate_test(
                self.cfg, test, frozen, scorers, label=f"{model}/{comp_name}", bootstrap=bootstrap
            )
            result.update(
                model=model,
                composition=comp_name,
                composition_label=comp.description,
                composition_details=comp.details,
                k=comp.k,
                exclusions=exclusions,
                accounting=self._accounting(test, model, comp),
                trace_truncation={
                    "items_with_truncated_trace": int((test["n_truncated_traces"] > 0).sum()),
                    "truncated_traces": int(test["n_truncated_traces"].sum()),
                    "max_trace_tokens": int(test["max_trace_tokens"].max()) if len(test) else 0,
                },
            )
            out = comp_dir / "test"
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.json").write_text(json.dumps(json_safe(result), indent=2))
            per_item.to_parquet(out / "per_item_scores.parquet", index=False)
            with open(out / "evaluation_log.jsonl", "a", encoding="utf-8") as fh:
                entry = {"time": time.time(), "frozen_params_sha256": frozen["_sha256"]}
                fh.write(json.dumps(entry) + "\n")
        self._manifest(f"evaluate-{stage}", {"bootstrap": bootstrap})

    def _accounting(self, test: pd.DataFrame, model: str, comp: Composition) -> dict[str, int]:
        """Cost accounting: N generations, N embeddings, C(N,2) comparisons per item."""
        n = len(test)
        acc = {
            "items": n,
            "panel_generations": n * comp.k,
            "trace_embeddings": n * comp.k,
            "pairwise_comparisons": n * comb(comp.k, 2),
        }
        if self.is_headline(model, comp.name):
            acc["neutral_sample_generations_b4_b5"] = n * self.cfg.baselines.k_samples
        return acc

    def tables(self) -> Path:
        cfg, head = self.cfg, self.cfg.llm.headline_model
        tables_dir, fig_dir = self.results_dir / "tables", self.results_dir / "figures"
        metrics = lambda m, c: reporting.load_metrics(self.comp_dir(m, c) / "test" / "metrics.json")
        written = []
        headline = metrics(head, HEADLINE)
        if headline is not None:
            reporting.headline_table(headline, tables_dir)
            reporting.cd_subset_table(headline, tables_dir)
            reporting.threshold_sweep_figure(headline, fig_dir)
            written += ["headline", "cd_subset", "threshold_sweep"]
        model_entries = [
            (m, x)
            for m in [head, *cfg.llm.ablation_models]
            if (x := metrics(m, HEADLINE)) is not None
        ]
        if model_entries:
            reporting.model_ablation_table(model_entries, tables_dir)
            written.append("model_ablation")
        comp_entries = [
            (get_composition(c, cfg).description, x)
            for c in cfg.ablations.compositions
            if (x := metrics(head, c)) is not None
        ]
        if comp_entries:
            reporting.composition_table(comp_entries, tables_dir)
            written.append("composition_ablation")
        self._manifest("tables", {"written": written})
        return self.results_dir

    # ------------------------------------------------------------------ manifest
    def _manifest(self, command: str, extra: dict[str, Any]) -> None:
        summaries = {}
        for model, store in self._stores.items():
            frame = store.frame()
            if frame.empty:
                continue
            summaries[model] = {
                "units": len(frame),
                "status_counts": frame["status"].value_counts().to_dict(),
                "retried_units": int((frame["n_attempts"] > 1).sum()),
                "providers": frame["provider"].value_counts(dropna=False).to_dict(),
                "finish_reasons": frame["finish_reason"].value_counts(dropna=False).to_dict(),
                "prompt_tokens": float(frame["prompt_tokens"].sum()),
                "completion_tokens": float(frame["completion_tokens"].sum()),
                "reasoning_tokens": float(frame["reasoning_tokens"].fillna(0).sum()),
                "cost_usd": float(frame["cost_usd"].fillna(0).sum()),
            }
        write_manifest(
            self.run_dir,
            command,
            self.cfg,
            {
                "options": asdict(self.opts),
                "run_tag": self.opts.tag,
                "prompts": {"version": self.prompts.version, "sha256": self.prompts.sha256},
                "embedding_model": None
                if self._embeddings is None
                else self._embeddings.encoder.model_name,
                "nli_model": None if self._nli is None else self._nli.model_name,
                "generation_summary": summaries,
                "model_capabilities": self._capabilities,
                **extra,
            },
        )
