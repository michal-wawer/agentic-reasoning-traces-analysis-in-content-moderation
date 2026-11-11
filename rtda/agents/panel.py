"""Deliberation stage: independent agent generations with raw-output persistence.

Every generation unit ``(item, role, sample_index)`` is stored as one JSON line in
``runs/<run>/<model>/generations.jsonl``. The line holds **every** attempt: the raw response
content, the exact parameters sent, the seed, usage and cost, and the parse error if any. It is
written before any aggregation. A unit whose output still fails validation after
``parse_retries`` re-asks is stored with ``status="parse_failed"``. It is never silently dropped:
downstream code excludes those items explicitly and reports how many.

Transport failures (the API still failing after the configured retries) are logged to
``errors.jsonl`` and not stored, so they are retried the next time the stage runs.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from rtda.agents.compositions import Unit
from rtda.agents.prompts import PromptSet
from rtda.agents.schema import ParseError, parse_agent_output
from rtda.config import ExperimentConfig, derive_seed
from rtda.llm.base import ChatRequest, LLMBackend

log = logging.getLogger(__name__)

Key = tuple[str, str, int]  # (item_id, role, sample_index)


class GenerationStore:
    """Append-only JSONL store of generation units, keyed by ``(item_id, role, sample_index)``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.errors_path = self.path.with_name("errors.jsonl")
        self._records: dict[Key, dict[str, Any]] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        self._records[(rec["item_id"], rec["role"], rec["sample_index"])] = rec

    def __contains__(self, key: Key) -> bool:
        return key in self._records

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: dict[str, Any]) -> None:
        key = (record["item_id"], record["role"], record["sample_index"])
        if key in self._records:
            raise ValueError(f"duplicate generation unit {key}")
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._records[key] = record

    def log_error(self, payload: dict[str, Any]) -> None:
        with open(self.errors_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def records(self) -> Iterable[dict[str, Any]]:
        return self._records.values()

    def frame(self) -> pd.DataFrame:
        """One row per unit, with the parsed fields flattened and usage summed over attempts."""
        rows = []
        for rec in self._records.values():
            out = rec.get("output") or {}
            attempts = rec["attempts"]
            rows.append(
                {
                    "item_id": rec["item_id"],
                    "split": rec["split"],
                    "role": rec["role"],
                    "sample_index": rec["sample_index"],
                    "status": rec["status"],
                    "decision": out.get("decision"),
                    "confidence": out.get("confidence"),
                    "trace": rec.get("trace"),
                    "n_attempts": len(attempts),
                    "finish_reason": attempts[-1]["response"].get("finish_reason"),
                    "provider": attempts[-1]["response"].get("provider"),
                    "prompt_tokens": _sum(a["response"].get("prompt_tokens") for a in attempts),
                    "completion_tokens": _sum(
                        a["response"].get("completion_tokens") for a in attempts
                    ),
                    "reasoning_tokens": _sum(
                        a["response"].get("reasoning_tokens") for a in attempts
                    ),
                    "cost_usd": _sum(a["response"].get("cost_usd") for a in attempts),
                    "n_cached_attempts": sum(bool(a["response"].get("cached")) for a in attempts),
                }
            )
        columns = [
            "item_id", "split", "role", "sample_index", "status", "decision", "confidence",
            "trace", "n_attempts", "finish_reason", "provider", "prompt_tokens",
            "completion_tokens", "reasoning_tokens", "cost_usd", "n_cached_attempts",
        ]  # fmt: skip
        return pd.DataFrame(rows, columns=columns)


def _sum(values: Iterable[float | int | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(sum(vals)) if vals else None


@dataclass
class GenerationStats:
    requested: int = 0
    already_stored: int = 0
    generated: int = 0
    parse_failed: int = 0
    transport_failed: int = 0


async def _generate_unit(
    backend: LLMBackend,
    cfg: ExperimentConfig,
    prompts: PromptSet,
    model: str,
    item: Any,
    unit: Unit,
) -> dict[str, Any]:
    attempts = []
    system_prompt = prompts.system_prompt(unit.role)
    user_prompt = prompts.user_prompt(item.text)
    for attempt in range(cfg.llm.parse_retries + 1):
        seed = derive_seed(cfg.seed, model, item.item_id, unit.role, unit.sample_index, attempt)
        request = ChatRequest(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            seed=seed,
            json_mode=True,
            response_schema=prompts.response_schema,
            reasoning_effort=cfg.llm.reasoning_effort,
            tag=f"{unit.role}|attempt={attempt}",
            sample_index=unit.sample_index,
        )
        response = await backend.complete(request)
        entry: dict[str, Any] = {
            "attempt": attempt,
            "seed": seed,
            "response": response.to_dict(),
            "parse_error": None,
        }
        attempts.append(entry)
        try:
            output = parse_agent_output(response.content)
        except ParseError as exc:
            entry["parse_error"] = str(exc)
            continue
        return _record(cfg, prompts, model, item, unit, "ok", attempts, output)
    return _record(cfg, prompts, model, item, unit, "parse_failed", attempts, None)


def _record(cfg, prompts, model, item, unit, status, attempts, output) -> dict[str, Any]:
    return {
        "model": model,
        "item_id": item.item_id,
        "split": item.split,
        "role": unit.role,
        "sample_index": unit.sample_index,
        "prompt_version": prompts.version,
        "prompt_sha256": prompts.sha256,
        "system_prompt": prompts.system_prompt(unit.role),
        "status": status,
        "attempts": attempts,
        "output": output.model_dump() if output is not None else None,
        "trace": output.trace_text(cfg.prompts.trace_fields) if output is not None else None,
        "created_at": time.time(),
    }


async def generate(
    backend: LLMBackend,
    cfg: ExperimentConfig,
    prompts: PromptSet,
    model: str,
    items: pd.DataFrame,
    units: Iterable[Unit],
    store: GenerationStore,
    desc: str = "generate",
) -> GenerationStats:
    """Generate every missing ``(item, unit)`` pair and append it to ``store``."""
    units = list(dict.fromkeys(units))
    stats = GenerationStats()
    todo = []
    for item in items.itertuples(index=False):
        for unit in units:
            stats.requested += 1
            if (item.item_id, unit.role, unit.sample_index) in store:
                stats.already_stored += 1
            else:
                todo.append((item, unit))

    async def run(item: Any, unit: Unit) -> None:
        try:
            record = await _generate_unit(backend, cfg, prompts, model, item, unit)
        except Exception as exc:  # noqa: BLE001 - transport failure after retries; retried next run
            stats.transport_failed += 1
            store.log_error(
                {
                    "model": model,
                    "item_id": item.item_id,
                    "role": unit.role,
                    "sample_index": unit.sample_index,
                    "error": f"{type(exc).__name__}: {exc}",
                    "time": time.time(),
                }
            )
            log.warning("generation failed for %s/%s: %s", item.item_id, unit.label, exc)
            return
        store.add(record)
        stats.generated += 1
        stats.parse_failed += record["status"] == "parse_failed"

    if todo:
        await tqdm_asyncio.gather(*(run(i, u) for i, u in todo), desc=desc)
    return stats
