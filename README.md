# RTDA: Reasoning-Trace Disagreement Analysis

Experiment code for *Knowing When to Escalate: Disagreement Patterns for Uncertainty-Aware Content
Moderation*. This release contains the implementation and configuration,
not verified paper results or cached model responses. The pipeline can generate evaluation tables
and figures when run with downloaded data and model outputs.

A panel of five perspective-differentiated LLM agents judges each item (Keep/Remove, a structured
justification, and a confidence). The justifications are embedded, and each item is assigned to one of four
states from reasoning similarity × conclusion agreement:

| State | Mean trace similarity S(x) | Conclusion agreement A(x) |
|---|---|---|
| CA Convergent Agreement | ≥ τ_s | 1 |
| DA Divergent Agreement | < τ_s | 1 |
| CD Convergent Disagreement | ≥ τ_s | 0 |
| DD Divergent Disagreement | < τ_s | 0 |

The structural signal Φ is compared with six baselines (B1–B6) as a predictor of human annotator
disagreement on the Measuring Hate Speech corpus.

## Setup

```bash
poetry install                    # Python 3.11-3.14
brew install libomp               # macOS only: runtime needed by LightGBM (B6)
cp .env.example .env              # then set OPENROUTER_API_KEY
```

All LLM calls go through [OpenRouter](https://openrouter.ai). All parameters live in
[`configs/default.yaml`](configs/default.yaml). Agent prompts live in
[`configs/prompts/v1.yaml`](configs/prompts/v1.yaml).

## Dataset download and exact splits

The corpus is downloaded from [Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)
by UC Berkeley D-Lab (Kennedy et al., 2020; Sachdeva et al., 2022). The upstream dataset card
lists CC BY 4.0; retain its attribution and consult its documentation when using the data.
Corpus text and annotator records are not bundled in this release.

Run `poetry run rtda prepare-data` to download the corpus and generate local processed data.
The default configuration uses seed 42 and selects 200 training, 200 calibration, and 1,000 test items.
[`splits.csv`](splits.csv) records the existing exact assignment with only `item_id` (the upstream
`comment_id`) and `split` (`train`, `cal`, or `test`). It contains no comment text or annotator records.
The current loader generates splits from the seed; it does not automatically import this manifest.
Regeneration requires the same upstream dataset contents and compatible preprocessing behavior.

## Running the experiments

```bash
poetry run rtda prepare-data                 # download corpus, compute d_h, seeded train/cal/test splits

# 1. Headline experiment (tab:headline, tab:cd-subset, fig:threshold-sweep)
poetry run rtda generate  headline
poetry run rtda features  headline
poetry run rtda calibrate headline           # calibration split ONLY -> frozen_params.json
poetry run rtda evaluate  headline           # test split with frozen params

# 2. Base-model ablation (tab:model-ablation)
poetry run rtda generate models && poetry run rtda features models
poetry run rtda calibrate models && poetry run rtda evaluate models

# 3. Agent-composition ablation (tab:composition-ablation)
poetry run rtda generate compositions && poetry run rtda features compositions
poetry run rtda calibrate compositions && poetry run rtda evaluate compositions
```

`poetry run rtda run-all` runs all of the above in sequence. Every stage is resumable: completed generations
are recorded in `generations.jsonl` and reused when resuming. The OpenRouter backend does not
cache API responses in SQLite.

## Citations

```
@inproceedings{10.1145/3799682.3840059,
  author = {Wawer, Micha{\l} and Chudziak, Jaros{\l}aw A.},
  title = {Knowing When to Escalate: Disagreement Patterns for Uncertainty-Aware Content Moderation},
  year = {2026},
  url = {https://doi.org/10.1145/3799682.3840059},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  pages = {7936–7941},
  numpages = {6},
  location = {Rome, Italy},
  series = {CIKM '26}
}
```
