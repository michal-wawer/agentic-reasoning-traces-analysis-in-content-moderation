"""Command-line interface: ``poetry run rtda --help``.

Reproduction see README.md:
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rtda.config import ExperimentConfig
from rtda.pipeline.run import STAGES, Experiment, RunOptions

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__.split("\n")[0])
console = Console()

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Experiment config YAML")]
MockOpt = Annotated[bool, typer.Option("--mock", help="Offline deterministic MockBackend")]
FakeOpt = Annotated[
    bool, typer.Option("--fake-models", help="Offline hashing encoder / Jaccard NLI (smoke only)")
]
LimitOpt = Annotated[int | None, typer.Option("--limit", help="First N items per split")]
StageArg = Annotated[str, typer.Argument(help="headline | models | compositions | all")]

DEFAULT_CONFIG = Path("configs/default.yaml")


def _setup(config: Path, mock: bool, fake_models: bool, limit: int | None) -> Experiment:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpcore", "openai", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = ExperimentConfig.load(config)
    return Experiment(cfg, RunOptions(mock=mock, fake_models=fake_models, limit=limit))


def _stages(stage: str) -> list[str]:
    if stage == "all":
        return list(STAGES)
    if stage not in STAGES:
        raise typer.BadParameter(f"stage must be one of {STAGES} or 'all'")
    return [stage]


@app.command("prepare-data")
def prepare_data(config: ConfigOpt = DEFAULT_CONFIG, force: bool = False) -> None:
    """Download Measuring Hate Speech, compute d_h, and make seeded train/cal/test splits."""
    from rtda.data.measuring_hate_speech import prepare_dataset

    cfg = ExperimentConfig.load(config)
    df = prepare_dataset(cfg, force=force)
    summary = df.groupby("split").agg(
        n=("item_id", "size"), positive_rate=("high_disagreement", "mean"), mean_d_h=("d_h", "mean")
    )
    console.print(summary)


@app.command("estimate-cost")
def estimate_cost(
    config: ConfigOpt = DEFAULT_CONFIG,
    limit: LimitOpt = None,
    prompt_tokens: float = 650.0,
    completion_tokens: float = 900.0,
    observed_limit: Annotated[
        int | None, typer.Option(help="Use token usage observed in the live --limit N run")
    ] = None,
) -> None:
    """Print generation counts and the estimated OpenRouter cost for the remaining generations."""
    from rtda.pipeline.cost import estimate

    exp = _setup(config, False, False, limit)
    observed = (
        Experiment(exp.cfg, RunOptions(limit=observed_limit))
        if observed_limit is not None
        else None
    )
    rows = estimate(exp, prompt_tokens, completion_tokens, observed)
    table = Table(title=f"Generation plan ({exp.opts.tag})")
    for col in ["stage", "model", "purpose", "items", "units", "generations", "missing",
                "prompt tok/gen", "compl. tok/gen", "tokens from", "est. cost (USD)"]:  # fmt: skip
        table.add_column(col)
    total = 0.0
    for r in rows:
        total += r.est_cost_usd or 0.0
        values = list(asdict(r).values())
        values[-1] = "n/a" if r.est_cost_usd is None else f"{r.est_cost_usd:,.2f}"
        table.add_row(*map(str, values))
    console.print(table)
    console.print(f"Estimated total for missing generations: [bold]${total:,.2f}[/bold]")


@app.command()
def generate(
    stage: StageArg,
    config: ConfigOpt = DEFAULT_CONFIG,
    mock: MockOpt = False,
    limit: LimitOpt = None,
) -> None:
    """Run the LLM generations for a stage (resumable; raw outputs go to generations.jsonl)."""
    exp = _setup(config, mock, False, limit)
    for s in _stages(stage):
        for job in exp.generate(s):
            console.print(job)


@app.command()
def features(
    stage: StageArg,
    config: ConfigOpt = DEFAULT_CONFIG,
    mock: MockOpt = False,
    fake_models: FakeOpt = False,
    limit: LimitOpt = None,
) -> None:
    """Embed traces, compute similarity/agreement features, and the B3-B5 neutral-sample features."""
    exp = _setup(config, mock, fake_models, limit)
    for s in _stages(stage):
        exp.build_features(s)


@app.command()
def calibrate(
    stage: StageArg,
    config: ConfigOpt = DEFAULT_CONFIG,
    mock: MockOpt = False,
    fake_models: FakeOpt = False,
    limit: LimitOpt = None,
    force: Annotated[bool, typer.Option(help="Overwrite frozen params (logged)")] = False,
) -> None:
    """Select tau_s, the state order, m, and all baseline thresholds on the CALIBRATION split only."""
    exp = _setup(config, mock, fake_models, limit)
    for s in _stages(stage):
        exp.calibrate(s, force=force)


@app.command()
def evaluate(
    stage: StageArg,
    config: ConfigOpt = DEFAULT_CONFIG,
    mock: MockOpt = False,
    fake_models: FakeOpt = False,
    limit: LimitOpt = None,
    bootstrap: bool = True,
) -> None:
    """Evaluate on the TEST split with frozen calibration parameters."""
    exp = _setup(config, mock, fake_models, limit)
    for s in _stages(stage):
        exp.evaluate(s, bootstrap=bootstrap)


@app.command()
def tables(
    config: ConfigOpt = DEFAULT_CONFIG,
    mock: MockOpt = False,
    fake_models: FakeOpt = False,
    limit: LimitOpt = None,
) -> None:
    """Write paper tables (CSV/Markdown/LaTeX) and the threshold-sweep figure."""
    exp = _setup(config, mock, fake_models, limit)
    out = exp.tables()
    console.print(f"Results written to {out}")


@app.command("run-all")
def run_all(
    config: ConfigOpt = DEFAULT_CONFIG,
    stage: Annotated[str, typer.Option(help="headline | models | compositions | all")] = "all",
    mock: MockOpt = False,
    fake_models: FakeOpt = False,
    limit: LimitOpt = None,
    bootstrap: bool = True,
    force: bool = False,
) -> None:
    """Run every stage end to end (use --mock --fake-models --limit 20 for an offline smoke test)."""
    exp = _setup(config, mock, fake_models, limit)
    for s in _stages(stage):
        exp.generate(s)
        exp.build_features(s)
        exp.calibrate(s, force=force)
        exp.evaluate(s, bootstrap=bootstrap)
    console.print(f"Results written to {exp.tables()}")


if __name__ == "__main__":
    app()
