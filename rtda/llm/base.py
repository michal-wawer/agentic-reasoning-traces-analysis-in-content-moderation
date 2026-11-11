"""Common LLM backend interface.

All backends (OpenRouter-hosted GPT-5.4-Nano / Claude 4.5 Sonnet / MiniMax M3, or the offline mock)
implement :class:`LLMBackend`, so the base-model ablation only swaps the backend, never the pipeline.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChatRequest:
    """A single chat-completion request.

    ``tag`` and ``sample_index`` are never sent to the provider. They contribute to
    the request fingerprint used by the deterministic mock backend, distinguishing
    independent samples even when their prompts are identical.
    """

    model: str
    system_prompt: str
    user_prompt: str
    temperature: float | None
    max_tokens: int
    seed: int | None = None
    json_mode: bool = True
    response_schema: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    tag: str = ""
    sample_index: int = 0

    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ChatResponse:
    content: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    provider: str | None = None
    sent_params: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMBackend(ABC):
    """Asynchronous chat-completion backend."""

    name: str = "base"

    def __init__(self) -> None:
        self.n_api_calls = 0
        self.n_cache_hits = 0

    @abstractmethod
    async def complete(self, request: ChatRequest) -> ChatResponse:  # pragma: no cover - interface
        ...

    def capabilities(self, model: str) -> dict[str, Any]:
        """Provider-reported metadata for ``model`` (supported parameters, pricing), if known."""
        return {}

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None
