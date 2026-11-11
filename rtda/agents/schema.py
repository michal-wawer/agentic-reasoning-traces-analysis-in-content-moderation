"""Strict output schema for one agent judgment ``<r_i, d_i, gamma_i>``."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interpretation: str = Field(min_length=1)
    considerations: list[str] = Field(min_length=1)
    trade_offs: str = Field(min_length=1)
    judgment: str = Field(min_length=1)
    decision: Literal["Keep", "Remove"]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("decision", mode="before")
    @classmethod
    def _normalize_decision(cls, v: object) -> object:
        # Accept case variants only ("remove" -> "Remove"); anything else fails validation.
        if isinstance(v, str) and v.strip().lower() in {"keep", "remove"}:
            return v.strip().capitalize()
        return v

    @field_validator("considerations")
    @classmethod
    def _non_empty_items(cls, v: list[str]) -> list[str]:
        cleaned = [c.strip() for c in v if c and c.strip()]
        if not cleaned:
            raise ValueError("considerations must contain at least one non-empty string")
        return cleaned

    def trace_text(self, fields: list[str]) -> str:
        """Reasoning trace r_i: the configured reasoning fields, excluding decision/confidence."""
        parts = []
        for name in fields:
            if name in {"decision", "confidence"}:
                raise ValueError("decision/confidence must not be part of the embedded trace")
            value = getattr(self, name)
            parts.append("; ".join(value) if isinstance(value, list) else value.strip())
        return " ".join(parts)


class ParseError(ValueError):
    pass


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_agent_output(content: str) -> AgentOutput:
    """Parse and validate a model response. Raises :class:`ParseError` with the reason."""
    text = _FENCE.sub("", content.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Tolerate prose around a single JSON object, but nothing else.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ParseError("no JSON object found") from None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ParseError("top-level JSON value is not an object")
    try:
        return AgentOutput.model_validate(data)
    except ValidationError as exc:
        raise ParseError(f"schema validation failed: {exc.errors(include_url=False)}") from None
