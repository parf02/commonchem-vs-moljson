# CommonChem vs MolJSON on GPT-5.4

This directory is a GitHub-ready bundle of the exact code, prompts, schemas, raw model outputs, and scored results used for a 500-query follow-up to *Molecular Representations for Large Language Models*.

The goal of the run was narrow: compare `CommonChem` against `MolJSON` on a representative subset of the `MolJSON-data` benchmark, using strict structured outputs through the OpenAI Responses API.

## Headline result

- Queries: `500` total API calls (`250` paired prompts)
- Model: `gpt-5.4`
- Overall accuracy: MolJSON `83.2%`, CommonChem `82.0%`
- Paired difference: CommonChem minus MolJSON = `-1.2` points
- 95% paired bootstrap CI: `-6.4` to `+4.0` points
- Exact McNemar `p = 0.761`

| Slice | Pairs | MolJSON | CommonChem | CommonChem - MolJSON |
| --- | ---: | ---: | ---: | ---: |
| Text -> graph | 40 | 87.5% | 77.5% | -10.0 pts |
| Graph -> SMILES | 40 | 67.5% | 77.5% | +10.0 pts |
| Shortest path | 50 | 100.0% | 98.0% | -2.0 pts |
| Constrained generation | 120 | 80.0% | 78.3% | -1.7 pts |

## Repository layout

- [`data/queries.jsonl`](data/queries.jsonl): manifest for the exact 500 API calls
- [`data/query_specs/`](data/query_specs): one JSON spec per query, including prompt, expected answer, and local response path
- [`outputs/responses/`](outputs/responses): raw model outputs for all 500 calls
- [`outputs/results.csv`](outputs/results.csv): per-query scored results
- [`outputs/summary_by_family.csv`](outputs/summary_by_family.csv): unpaired accuracy summary
- [`outputs/paired_stats.csv`](outputs/paired_stats.csv): paired comparison statistics
- [`outputs/api_run_log.csv`](outputs/api_run_log.csv): API-side execution log with token usage
- [`outputs/api_run_summary.json`](outputs/api_run_summary.json): aggregate run metadata
- [`outputs/blog_post.md`](outputs/blog_post.md): long-form response to the paper
- [`outputs/linkedin_reply.md`](outputs/linkedin_reply.md): short LinkedIn reply draft
- [`outputs/figure_1_accuracy_by_task.png`](outputs/figure_1_accuracy_by_task.png) and [`outputs/figure_3_paired_outcomes.png`](outputs/figure_3_paired_outcomes.png): publication-style figures generated from the bundled CSV outputs
- [`schemas/commonchem_strict_schema.json`](schemas/commonchem_strict_schema.json): exact CommonChem schema used in the run
- [`schemas/commonchem_repo_schema.json`](schemas/commonchem_repo_schema.json): upstream `commonchem.json` copied from the CommonChem repo
- [`schemas/moljson_schema.json`](schemas/moljson_schema.json): MolJSON schema used in the run
- [`third_party_licenses/`](third_party_licenses): upstream license texts for redistributed code, schema files, and benchmark materials
- [`scripts/`](scripts): scripts used to generate the subset, run the API benchmark, score the outputs, and package this directory

## Quick start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Re-score the bundled responses:

```bash
python scripts/evaluate_benchmark.py --usage-summary-json outputs/api_run_summary.json --execution-mode responses_api --model gpt-5.4 --skip-plot
```

Regenerate the publication figures from the bundled raw outputs:

```bash
python scripts/make_publication_figures.py
```

Rerun the same 500 API calls against the bundled manifest:

```bash
export OPENAI_API_KEY=...
python scripts/run_openai_api_benchmark.py --resume
```

Regenerate the representative subset:

```bash
python scripts/prepare_api500_subset.py
```

`prepare_api500_subset.py` expects a checkout of `MolJSON-data` next to this repo. The bundled `run_openai_api_benchmark.py` and `evaluate_benchmark.py` are self-contained inside this directory.

## Notes

- The CommonChem run used a strict benchmark schema derived from the upstream repo schema plus the CommonChem spec. The exact resolved schema is bundled in [`schemas/commonchem_strict_schema.json`](schemas/commonchem_strict_schema.json).
- The raw `outputs/responses/` directory is intentionally included so someone else can audit parser failures, invalid JSON, and near misses directly.
- `scripts/moljson/` vendors the minimal MolJSON Python package needed by the benchmark scripts. Its upstream BSD-3-Clause license is preserved at [`scripts/moljson/LICENSE`](scripts/moljson/LICENSE).
- This bundle also redistributes benchmark-derived files from `MolJSON-data` and a schema file from `CommonChem`. Upstream license texts are preserved under [`third_party_licenses/`](third_party_licenses).
- `metadata.json` contains a compact machine-readable summary of the run.

## Upstream sources

- [CommonChem](https://github.com/CommonChem/CommonChem)
- [MolJSON](https://github.com/oxpig/MolJSON)
- [MolJSON-data](https://github.com/oxpig/MolJSON-data)
- [Paper](https://arxiv.org/html/2605.01822v1)
