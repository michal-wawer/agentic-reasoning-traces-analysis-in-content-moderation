"""OpenRouter chat-completion backend (OpenAI-compatible API).

Configured OpenRouter model IDs:
  * ``openai/gpt-5.4-nano``          - headline model (no ``temperature`` support: reasoning model)
  * ``anthropic/claude-sonnet-4.5``  - cross-model ablation (no ``seed`` support)
  * ``minimax/minimax-m3``           - cross-model ablation

Decoding parameters are sent **only if the model advertises support for them** in
``supported_parameters``. Every response records the exact parameters that were sent
(``ChatResponse.sent_params``), the provider that served it, token usage and cost, so the decoding
configuration actually applied is auditable per call.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from rtda.config import LLMConfig
from rtda.llm.base import ChatRequest, ChatResponse, LLMBackend

log = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    # Retry server-side errors only; 4xx (bad request, auth) must fail loudly.
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


def fetch_model_catalog(base_url: str, timeout_s: float = 30.0) -> dict[str, dict[str, Any]]:
    """Return ``{model_id: {supported_parameters, pricing, context_length}}`` from OpenRouter."""
    resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout_s)
    resp.raise_for_status()
    return {
        m["id"]: {
            "supported_parameters": sorted(m.get("supported_parameters") or []),
            "pricing": m.get("pricing") or {},
            "context_length": m.get("context_length"),
        }
        for m in resp.json()["data"]
    }


class OpenRouterBackend(LLMBackend):
    name = "openrouter"

    def __init__(
        self,
        cfg: LLMConfig,
        api_key: str | None = None,
    ):
        super().__init__()
        load_dotenv()
        key = api_key or os.environ.get(cfg.api_key_env)
        if not key:
            raise RuntimeError(
                f"OpenRouter API key not found. Set ${cfg.api_key_env} (see .env.example)."
            )
        headers = {}
        if os.environ.get("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
        if os.environ.get("OPENROUTER_APP_TITLE"):
            headers["X-Title"] = os.environ["OPENROUTER_APP_TITLE"]
        self._client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=key,
            timeout=cfg.timeout_s,
            max_retries=0,  # retries are handled by tenacity for logging/jitter control
            default_headers=headers or None,
        )
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.concurrency)
        self._catalog: dict[str, dict[str, Any]] | None = None

    # ------------------------------------------------------------------ capabilities
    def _get_catalog(self) -> dict[str, dict[str, Any]]:
        if self._catalog is None:
            self._catalog = fetch_model_catalog(self.cfg.base_url, self.cfg.timeout_s)
        return self._catalog

    def capabilities(self, model: str) -> dict[str, Any]:
        catalog = self._get_catalog()
        if model not in catalog:
            raise ValueError(f"Model {model!r} is not listed by OpenRouter /models")
        return catalog[model]

    def build_params(self, request: ChatRequest) -> dict[str, Any]:
        """Translate a request into API kwargs, dropping parameters the model does not support."""
        supported = set(self.capabilities(request.model)["supported_parameters"])
        params: dict[str, Any] = {"max_tokens": request.max_tokens}
        extra: dict[str, Any] = {"usage": {"include": True}}
        if request.temperature is not None and "temperature" in supported:
            params["temperature"] = request.temperature
        if request.seed is not None and "seed" in supported:
            params["seed"] = request.seed
        if request.response_schema is not None and "structured_outputs" in supported:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        elif request.json_mode and "response_format" in supported:
            params["response_format"] = {"type": "json_object"}
        if request.reasoning_effort is not None and "reasoning" in supported:
            extra["reasoning"] = {"effort": request.reasoning_effort}
        provider: dict[str, Any] = {"require_parameters": True}
        if self.cfg.provider_order:
            provider["order"] = list(self.cfg.provider_order)
            provider["allow_fallbacks"] = False
        extra["provider"] = provider
        params["extra_body"] = extra
        return params

    # ------------------------------------------------------------------ completion
    async def complete(self, request: ChatRequest) -> ChatResponse:
        async with self._sem:
            return await self._call_with_retry(request)

    async def _call_with_retry(self, request: ChatRequest) -> ChatResponse:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self.cfg.max_retries + 1),
            wait=wait_random_exponential(multiplier=1, max=60),
            reraise=True,
        ):
            with attempt:
                return await self._call(request)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _call(self, request: ChatRequest) -> ChatResponse:
        params = self.build_params(request)
        completion = await self._client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            **params,
        )
        self.n_api_calls += 1
        choice = completion.choices[0]
        usage = completion.usage
        usage_extra = (usage.model_extra or {}) if usage is not None else {}
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        sent = {k: v for k, v in params.items() if k != "extra_body"}
        sent.update({k: v for k, v in params["extra_body"].items() if k != "usage"})
        return ChatResponse(
            content=choice.message.content or "",
            model=completion.model or request.model,
            finish_reason=choice.finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
            cost_usd=usage_extra.get("cost"),
            provider=(completion.model_extra or {}).get("provider"),
            sent_params=sent,
            raw={"id": completion.id},
        )

    async def aclose(self) -> None:
        await self._client.close()
