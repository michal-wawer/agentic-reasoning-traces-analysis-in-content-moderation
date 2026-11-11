from __future__ import annotations

from rtda.config import LLMConfig
from rtda.llm.base import ChatRequest, ChatResponse, LLMBackend


def make_backend(cfg: LLMConfig, mock: bool = False) -> LLMBackend:
    """Return the OpenRouter backend, or the deterministic mock when ``mock=True``."""
    if mock:
        from rtda.llm.mock import MockBackend

        return MockBackend()
    from rtda.llm.openrouter import OpenRouterBackend

    return OpenRouterBackend(cfg)


__all__ = ["ChatRequest", "ChatResponse", "LLMBackend", "make_backend"]
