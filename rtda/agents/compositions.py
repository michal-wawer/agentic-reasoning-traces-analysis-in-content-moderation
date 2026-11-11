"""Agent compositions for the headline panel and the composition ablation.

A composition is a list of *generation units* ``(role, sample_index)``. Every unit is generated
independently (no inter-agent communication), so compositions that are subsets of the full panel
reuse those generations instead of re-querying the model:

* ``drop_*`` and ``k3_*``: subsets of the full five-agent panel.
* ``k7_add_procedural_victim``: full panel plus two extra perspectives.
* ``random_swap``: one panel role, chosen with a seeded RNG, replaced by one extra perspective.
* ``no_differentiation``: the neutral prompt, samples 0..4 (different decoding seeds only). These are
  the same generations used as the K samples of B4/B5.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from rtda.config import ExperimentConfig, derive_seed

HEADLINE = "full_5"


@dataclass(frozen=True)
class Unit:
    role: str
    sample_index: int = 0

    @property
    def label(self) -> str:
        return self.role if self.sample_index == 0 else f"{self.role}#{self.sample_index}"


@dataclass(frozen=True)
class Composition:
    name: str
    units: tuple[Unit, ...]
    description: str
    details: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.units)


def _drop(roles: list[str], *dropped: str) -> list[str]:
    missing = set(dropped) - set(roles)
    if missing:
        raise ValueError(f"cannot drop perspectives not in the panel: {sorted(missing)}")
    return [r for r in roles if r not in dropped]


def get_composition(name: str, cfg: ExperimentConfig) -> Composition:
    base = list(cfg.panel.perspectives)
    extra = list(cfg.ablations.extra_perspectives)
    if name == "full_5":
        roles, desc, details = base, "Full 5 perspectives", {}
    elif name == "k3_drop_legal_community":
        roles = _drop(base, "legal_framework", "community_norms")
        desc, details = "k=3 (drop legal, community)", {}
    elif name == "k7_add_procedural_victim":
        roles, desc, details = base + extra, "k=7 (add procedural, victim)", {}
    elif name == "random_swap":
        rng = random.Random(derive_seed(cfg.seed, "composition", "random_swap"))
        replaced = rng.choice(base)
        replacement = rng.choice(extra)
        roles = [replacement if r == replaced else r for r in base]
        desc = "Random perspective swap"
        details = {"replaced": replaced, "replacement": replacement}
    elif name == "drop_harm_focused":
        roles, desc, details = _drop(base, "harm_focused"), "Drop harm-focused", {}
    elif name == "drop_free_expression":
        roles, desc, details = _drop(base, "free_expression"), "Drop free-expression", {}
    elif name == "no_differentiation":
        neutral = cfg.baselines.single_agent_perspective
        n = len(base)
        if n > cfg.baselines.k_samples:
            raise ValueError("no_differentiation needs k_samples >= panel size")
        units = tuple(Unit(neutral, s) for s in range(n))
        return Composition(name, units, "No differentiation", {"prompt": neutral})
    else:
        raise ValueError(f"Unknown composition {name!r}")
    return Composition(name, tuple(Unit(r, 0) for r in roles), desc, details)


def neutral_sample_units(cfg: ExperimentConfig) -> tuple[Unit, ...]:
    """The K neutral samples shared by B3 (sample b3_sample_index), B4, B5."""
    role = cfg.baselines.single_agent_perspective
    return tuple(Unit(role, s) for s in range(cfg.baselines.k_samples))
