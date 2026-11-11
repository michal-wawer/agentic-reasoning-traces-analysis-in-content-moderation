"""Generation counts and API cost estimates, before spending money.

Counts follow the paper's accounting: N generations per item for the panel and K
generations per item for B4/B5. Costs use live OpenRouter pricing. Token usage per generation is
taken from generations already stored for that model (e.g. after a ``--limit`` smoke run) when
available, and otherwise from the assumptions passed in.
"""

from __future__ import annotations

from dataclasses import dataclass

from rtda.agents.panel import GenerationStore
from rtda.llm.openrouter import fetch_model_catalog
from rtda.pipeline.run import STAGES, Experiment


@dataclass
class CostRow:
    stage: str
    model: str
    purpose: str
    n_items: int
    n_units: int
    n_generations: int
    n_missing: int
    prompt_tokens_per_gen: float
    completion_tokens_per_gen: float
    token_source: str
    est_cost_usd: float | None


def estimate(
    exp: Experiment,
    assumed_prompt_tokens: float,
    assumed_completion_tokens: float,
    observed_run: Experiment | None = None,
) -> list[CostRow]:
    catalog = fetch_model_catalog(exp.cfg.llm.base_url)
    rows = []
    for stage in STAGES:
        for job in exp.generation_jobs(stage):
            items = exp.items(job.splits)
            store = exp.store(job.model)
            n_gen = len(items) * len(job.units)
            missing = sum(
                (i, u.role, u.sample_index) not in store
                for i in items["item_id"]
                for u in job.units
            )
            p_tok, c_tok, source = assumed_prompt_tokens, assumed_completion_tokens, "assumed"
            obs_store: GenerationStore | None = (
                observed_run.store(job.model) if observed_run else None
            )
            for candidate in (store, obs_store):
                if candidate is None or len(candidate) == 0:
                    continue
                frame = candidate.frame().dropna(subset=["prompt_tokens", "completion_tokens"])
                if len(frame):
                    # per unit, summed over parse retries, so retries are included in the estimate
                    p_tok = frame["prompt_tokens"].mean()
                    c_tok = frame["completion_tokens"].mean()
                    source = f"observed ({len(frame)} units)"
                    break
            pricing = catalog.get(job.model, {}).get("pricing", {})
            cost = None
            if pricing:
                cost = missing * (
                    p_tok * float(pricing.get("prompt", 0))
                    + c_tok * float(pricing.get("completion", 0))
                )
            rows.append(
                CostRow(
                    stage, job.model, job.purpose, len(items), len(job.units), n_gen, missing,
                    round(p_tok, 1), round(c_tok, 1), source, cost,
                )
            )  # fmt: skip
    return rows
