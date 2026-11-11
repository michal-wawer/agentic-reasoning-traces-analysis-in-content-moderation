"""Test-set evaluation with frozen parameters.

This module never selects a threshold. It loads ``calibration/frozen_params.json``, checks that the
test items don't overlap the calibration items, and computes:

* for Phi and each baseline: Pearson r (Fisher-z CI), Spearman rho, two-sided p-values, AUROC, and
  F1 at the frozen operating point;
* the per-state breakdown: n, mean d_h, Pearson r of the within-state tie-break u(x) with d_h and
  its CI;
* a post-hoc tau_s sensitivity sweep on test. The state order, m and all
  baseline thresholds stay frozen; the sweep is reported, never used for selection;
* supplementary paired bootstrap CIs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtda.baselines.scorers import ColumnScorer, LeakageError, LightGBMScorer, require_split
from rtda.config import ExperimentConfig, derive_seed
from rtda.evaluation.stats import auroc, correlations, f1_at, paired_bootstrap
from rtda.taxonomy.score import phi_route, phi_score, tiebreak
from rtda.taxonomy.states import classify_states

PHI = "Phi"


def load_frozen(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: run `rtda calibrate` before evaluating on the test set."
        )
    text = path.read_text()
    frozen = json.loads(text)
    frozen["_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return frozen


def phi_outputs(test: pd.DataFrame, phi: dict[str, Any], tau_s: float | None = None):
    tau_s = phi["tau_s"] if tau_s is None else tau_s
    states = classify_states(test["s_mean"].to_numpy(), test["agree"].to_numpy(bool), tau_s)
    u = tiebreak(test["vote_entropy"].to_numpy(), test["s_mean"].to_numpy())
    score = phi_score(states, u, phi["state_order"], phi["tiebreak_eps"])
    route = phi_route(states, phi["state_order"], phi["m"])
    return states, u, score, route


def evaluate_test(
    cfg: ExperimentConfig,
    test: pd.DataFrame,
    frozen: dict[str, Any],
    scorers: list[ColumnScorer | LightGBMScorer],
    label: str,
    bootstrap: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    require_split(test, "test", "evaluate_test")
    overlap = set(test["item_id"]) & set(frozen["calibration_item_ids"])
    if overlap:
        raise LeakageError(f"{len(overlap)} test items also appear in the calibration set")
    level = cfg.evaluation.ci_level
    y = test["high_disagreement"].to_numpy(dtype=int)
    d_h = test["d_h"].to_numpy(dtype=float)
    phi = frozen["phi"]

    states, u, phi_sc, phi_rt = phi_outputs(test, phi)
    per_item = test[["item_id", "d_h", "high_disagreement"]].copy()
    per_item["state"], per_item["u"], per_item[PHI] = states, u, phi_sc
    per_item[f"{PHI}_escalate"] = phi_rt

    methods: dict[str, dict[str, Any]] = {}
    scores = {PHI: phi_sc}
    thresholds = {PHI: phi["score_threshold"]}
    methods[PHI] = {
        "label": "Structural (Phi)",
        "threshold": phi["score_threshold"],
        **correlations(phi_sc, d_h, level).to_dict(),
        "auroc": auroc(y, phi_sc),
        "f1": f1_at(y, phi_rt.astype(float), 0.5),
        "escalation_rate": float(phi_rt.mean()),
    }
    for sc in scorers:
        s = sc.score(test)
        t = frozen["methods"][sc.method_id]["threshold"]
        scores[sc.method_id], thresholds[sc.method_id] = s, t
        per_item[sc.method_id] = s
        methods[sc.method_id] = {
            "label": sc.label,
            "supervised": sc.supervised,
            "threshold": t,
            **correlations(s, d_h, level).to_dict(),
            "auroc": auroc(y, s),
            "f1": f1_at(y, s, t),
            "escalation_rate": float((s >= t).mean()),
        }

    per_state = []
    for state in phi["state_order"]:
        mask = states == state
        c = correlations(u[mask], d_h[mask], level)
        per_state.append(
            {
                "state": state,
                "n": int(mask.sum()),
                "mean_d_h": float(d_h[mask].mean()) if mask.any() else float("nan"),
                "positive_rate": float(y[mask].mean()) if mask.any() else float("nan"),
                **{f"u_{k}": v for k, v in c.to_dict().items() if k != "n"},
            }
        )

    sweep = []
    b1_f1 = methods["B1"]["f1"] if "B1" in methods else float("nan")
    for tau_s in cfg.taxonomy.tau_s_candidates:
        _, _, _, rt = phi_outputs(test, phi, tau_s)
        sweep.append({"tau_s": tau_s, "phi_f1": f1_at(y, rt.astype(float), 0.5), "b1_f1": b1_f1})

    result: dict[str, Any] = {
        "label": label,
        "n_test": len(test),
        "positive_rate": float(y.mean()),
        "frozen_params_sha256": frozen.get("_sha256"),
        "tau_s": phi["tau_s"],
        "tau_a": phi["tau_a"],
        "state_order": phi["state_order"],
        "escalated_states": phi["escalated_states"],
        "methods": methods,
        "per_state": per_state,
        "threshold_sweep_posthoc": sweep,
    }
    if bootstrap:
        diffs = [(PHI, m) for m in scores if m != PHI]
        result["bootstrap"] = paired_bootstrap(
            y,
            scores,
            thresholds,
            n_boot=cfg.evaluation.n_bootstrap,
            seed=derive_seed(cfg.seed, "bootstrap", label),
            level=level,
            diffs=diffs,
        )
    return result, per_item


def within_state_order_holds(per_state: list[dict[str, Any]]) -> dict[str, bool]:
    """Check the within-state ordering hypothesis (CD highest, DA lowest mean d_h)."""
    means = {r["state"]: r["mean_d_h"] for r in per_state if r["n"] > 0}
    if not means:
        return {"cd_highest": False, "da_lowest": False}
    return {
        "cd_highest": "CD" in means and means["CD"] == max(means.values()),
        "da_lowest": "DA" in means and means["DA"] == min(means.values()),
    }


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    return obj
