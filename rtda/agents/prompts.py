"""Versioned prompt templates for the perspective-differentiated agents.

Prompts live in ``configs/prompts/<version>.yaml``,
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rtda.config import REPO_ROOT, PromptsConfig


@dataclass(frozen=True)
class PromptSet:
    version: str
    perspectives: dict[str, str]
    shared_instructions: str
    user_template: str
    response_schema: dict[str, Any]
    sha256: str

    def system_prompt(self, role: str) -> str:
        if role not in self.perspectives:
            raise KeyError(f"Unknown perspective {role!r}; known: {sorted(self.perspectives)}")
        return f"{self.perspectives[role].strip()}\n\n{self.shared_instructions.strip()}"

    def user_prompt(self, text: str) -> str:
        return self.user_template.replace("{text}", text)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_prompts(cfg: PromptsConfig) -> PromptSet:
    path = resolve_path(cfg.path)
    raw_bytes = path.read_bytes()
    data = yaml.safe_load(raw_bytes)
    return PromptSet(
        version=str(data["version"]),
        perspectives={k: str(v) for k, v in data["perspectives"].items()},
        shared_instructions=str(data["shared_instructions"]),
        user_template=str(data["user_template"]),
        response_schema=data["response_schema"],
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
