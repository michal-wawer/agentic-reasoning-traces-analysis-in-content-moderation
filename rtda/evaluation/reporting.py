"""Regenerate the paper's tables and figure from test-set metrics files.

Outputs, under ``results/<run_tag>/``:
  tables/headline.{csv,md,tex}
  tables/cd_subset.{csv,md,tex}
  tables/model_ablation.{csv,md,tex}
  tables/composition_ablation.{csv,md,tex}
  tables/headline_bootstrap.csv             supplementary paired-bootstrap CIs
  figures/threshold_sweep.{csv,pdf} (+ pgfplots coordinates .txt)

The CSV files have full precision plus CIs and p-values. The ``.tex`` files contain booktabs rows in
a booktabs-compatible layout for inclusion in a manuscript.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

METHOD_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "Phi"]
PAPER_LABELS = {
    "B1": "B1 Divergence-only",
    "B2": "B2 Vote entropy",
    "B3": "B3 Verbalized conf.",
    "B4": "B4 Self-consistency",
    "B5": "B5 Semantic uncertainty",
    "B6": "B6 Supervised$^{\\dagger}$",
    "Phi": "\\textbf{Structural ($\\Phi$)}",
}


def _f(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}"


def _p(x: Any) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return "<.001" if x < 0.001 else f"{x:.3f}"


def _write(df: pd.DataFrame, tex_rows: list[str], out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".3f") + "\n")
    (out / f"{name}.tex").write_text("\n".join(tex_rows) + "\n")


def load_metrics(path: Path) -> dict[str, Any] | None:
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def headline_table(metrics: dict[str, Any], out: Path) -> pd.DataFrame:
    rows, tex = [], []
    for mid in METHOD_ORDER:
        m = metrics["methods"].get(mid)
        if m is None:
            continue
        rows.append(
            {
                "method": mid,
                "label": m["label"],
                "n": m["n"],
                "pearson_r": m["pearson_r"],
                "pearson_ci_low": m["pearson_ci_low"],
                "pearson_ci_high": m["pearson_ci_high"],
                "pearson_p": m["pearson_p"],
                "spearman_rho": m["spearman_rho"],
                "spearman_p": m["spearman_p"],
                "auroc": m["auroc"],
                "f1": m["f1"],
                "threshold": m["threshold"],
                "escalation_rate": m["escalation_rate"],
            }
        )
        cells = [_f(m["pearson_r"]), _f(m["spearman_rho"]), _f(m["auroc"]), _f(m["f1"])]
        if mid == "Phi":
            tex.append("\\midrule")
            cells = [f"\\textbf{{{c}}}" for c in cells]
        tex.append(f"{PAPER_LABELS[mid]} & " + " & ".join(cells) + " \\\\")
    df = pd.DataFrame(rows)
    _write(df, tex, out, "headline")
    if "bootstrap" in metrics:
        pd.DataFrame([{"comparison": k, **v} for k, v in metrics["bootstrap"].items()]).to_csv(
            out / "headline_bootstrap.csv", index=False
        )
    return df


def cd_subset_table(metrics: dict[str, Any], out: Path) -> pd.DataFrame:
    rows, tex = [], []
    for r in sorted(
        metrics["per_state"], key=lambda r: math.inf if r["mean_d_h"] is None else r["mean_d_h"]
    ):
        rows.append(r)
        ci = f"$[{_f(r['u_pearson_ci_low'])},{_f(r['u_pearson_ci_high'])}]$"
        tex.append(
            f"${r['state']}(c)$ & {r['n']} & {_f(r['mean_d_h'])} & {_f(r['u_pearson_r'])} {ci} \\\\"
        )
    df = pd.DataFrame(rows)
    _write(df, tex, out, "cd_subset")
    return df


def model_ablation_table(entries: list[tuple[str, dict[str, Any]]], out: Path) -> pd.DataFrame:
    rows, tex = [], []
    for model, m in entries:
        cd = next((r for r in m["per_state"] if r["state"] == "CD"), None)
        phi_f1, b1_f1 = m["methods"]["Phi"]["f1"], m["methods"]["B1"]["f1"]
        means = {r["state"]: r["mean_d_h"] for r in m["per_state"] if r["n"] > 0}
        rows.append(
            {
                "model": model,
                "phi_f1": phi_f1,
                "b1_f1": b1_f1,
                "delta_f1": phi_f1 - b1_f1,
                "r_on_cd": cd["u_pearson_r"] if cd else None,
                "r_on_cd_ci_low": cd["u_pearson_ci_low"] if cd else None,
                "r_on_cd_ci_high": cd["u_pearson_ci_high"] if cd else None,
                "n_cd": cd["n"] if cd else 0,
                "tau_s": m["tau_s"],
                "cd_highest_mean_d_h": bool(means) and means.get("CD") == max(means.values()),
                "da_lowest_mean_d_h": bool(means) and means.get("DA") == min(means.values()),
            }
        )
        tex.append(
            f"{model} & {_f(phi_f1)} & {_f(b1_f1)} & {phi_f1 - b1_f1:+.2f} & "
            f"{_f(cd['u_pearson_r'] if cd else None)} \\\\"
        )
    df = pd.DataFrame(rows)
    _write(df, tex, out, "model_ablation")
    return df


def composition_table(entries: list[tuple[str, dict[str, Any]]], out: Path) -> pd.DataFrame:
    rows, tex = [], []
    for label, m in entries:
        phi_f1, b1_f1 = m["methods"]["Phi"]["f1"], m["methods"]["B1"]["f1"]
        rows.append(
            {
                "configuration": label,
                "k": m.get("k"),
                "phi_f1": phi_f1,
                "b1_f1": b1_f1,
                "delta_f1_vs_b1": phi_f1 - b1_f1,
                "phi_auroc": m["methods"]["Phi"]["auroc"],
                "b1_auroc": m["methods"]["B1"]["auroc"],
                "tau_s": m["tau_s"],
                "n_test": m["n_test"],
                "details": json.dumps(m.get("composition_details", {})),
            }
        )
        tex.append(f"{label} & {_f(phi_f1)} & {phi_f1 - b1_f1:+.2f} \\\\")
    df = pd.DataFrame(rows)
    _write(df, tex, out, "composition_ablation")
    return df


def threshold_sweep_figure(metrics: dict[str, Any], out: Path) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics["threshold_sweep_posthoc"])
    df.to_csv(out / "threshold_sweep.csv", index=False)
    coords = lambda col: " ".join(f"({t:.2f},{v:.3f})" for t, v in zip(df.tau_s, df[col]))
    (out / "threshold_sweep_pgfplots.txt").write_text(
        f"% Structural (Phi)\n{coords('phi_f1')}\n% Divergence-only (B1)\n{coords('b1_f1')}\n"
    )
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.plot(df.tau_s, df.phi_f1, marker="o", label="Structural (Φ)")
    ax.plot(df.tau_s, df.b1_f1, marker="s", linestyle="--", label="Divergence-only (B1)")
    ax.axvline(metrics["tau_s"], color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Similarity threshold τ_s")
    ax.set_ylabel("F1 (high-disagr. routing)")
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "threshold_sweep.pdf")
    plt.close(fig)
    return df
