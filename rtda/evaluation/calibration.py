"""Calibration: select every operating point on the calibration split, then freeze it.

This module is the **only** place thresholds are chosen. Its inputs are checked with
:func:`require_split`, so a table containing any non-calibration item raises ``LeakageError``.
The selected parameters are written to ``calibration/frozen_params.json``. Test-time evaluation
(:mod:`rtda.evaluation.test_eval`) only reads that file.

Selection rules (all maximize routing F1 for the high-disagreement label on the calibration set):

* Phi: grid over ``tau_s in tau_s_candidates`` x ``m in escalate_m_candidates``. For each tau_s,
  the state order is re-estimated by mean d_h. Ties in F1 go to the smaller m, then the smaller
  tau_s. tau_a is fixed a priori and not tuned.
* B1-B5: score threshold maximizing F1 (see ``best_f1_threshold``).
* B6: trained on the *train* split (``LightGBMScorer.fit`` also checks the split); only its
  decision threshold is selected on calibration.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtda.baselines.scorers import ColumnScorer, LightGBMScorer, require_split
from rtda.config import ExperimentConfig
from rtda.evaluation.stats import best_f1_threshold, f1_at
from rtda.taxonomy.score import phi_route, state_order_by_mean
from rtda.taxonomy.states import classify_states


def item_ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def calibrate_phi(cfg: ExperimentConfig, cal: pd.DataFrame) -> dict[str, Any]:
    require_split(cal, "cal", "calibrate_phi")
    y = cal["high_disagreement"].to_numpy(dtype=int)
    d_h = cal["d_h"].to_numpy(dtype=float)
    grid = []
    for tau_s in cfg.taxonomy.tau_s_candidates:
        states = classify_states(cal["s_mean"].to_numpy(), cal["agree"].to_numpy(bool), tau_s)
        order, missing = state_order_by_mean(states, d_h, cfg.taxonomy.fallback_state_order)
        counts = {s: int((states == s).sum()) for s in order}
        for m in cfg.taxonomy.escalate_m_candidates:
            route = phi_route(states, order, m)
            f1 = f1_at(y, route.astype(float), 0.5)
            grid.append(
                {
                    "tau_s": tau_s,
                    "m": m,
                    "f1": f1,
                    "state_order": order,
                    "missing_states": missing,
                    "state_counts": counts,
                }
            )
    best = min(grid, key=lambda g: (-round(g["f1"], 12), g["m"], g["tau_s"]))
    return {
        "tau_a": cfg.taxonomy.tau_a,
        "tau_s": best["tau_s"],
        "m": best["m"],
        "state_order": best["state_order"],
        "escalated_states": best["state_order"][-best["m"] :],
        "missing_states_on_cal": best["missing_states"],
        "score_threshold": float(len(best["state_order"]) - best["m"]),
        "cal_f1": best["f1"],
        "tiebreak_eps": cfg.taxonomy.tiebreak_eps,
        "grid": grid,
    }


def calibrate_thresholds(
    cal: pd.DataFrame, scorers: list[ColumnScorer | LightGBMScorer]
) -> dict[str, dict[str, Any]]:
    require_split(cal, "cal", "calibrate_thresholds")
    y = cal["high_disagreement"].to_numpy(dtype=int)
    out = {}
    for sc in scorers:
        t, f1 = best_f1_threshold(y, sc.score(cal))
        out[sc.method_id] = {"label": sc.label, "threshold": t, "cal_f1": f1}
    return out


def write_frozen(path: Path, payload: dict[str, Any], force: bool = False) -> None:
    """Write frozen params. Refuses to overwrite unless ``force``; every write is logged."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. Calibration is frozen; pass --force to recalibrate "
            "(the overwrite is recorded in calibration/history.jsonl)."
        )
    payload = {**payload, "frozen_at": time.time()}
    text = json.dumps(payload, indent=2, default=_json_default)
    path.write_text(text)
    with open(path.with_name("history.jsonl"), "a", encoding="utf-8") as fh:
        entry = {
            "time": payload["frozen_at"],
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "overwrote_existing": force,
        }
        fh.write(json.dumps(entry) + "\n")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
