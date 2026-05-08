from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parents[1]
ARTIFACTS_DIR = ROOT / "artifacts_api500"
TARGET_DIR = ROOT / "github_release"
MOLJSON_SRC = WORKSPACE_ROOT / "MolJSON" / "src"
MOLJSON_PACKAGE_SRC = MOLJSON_SRC / "moljson"
MOLJSON_LICENSE_SRC = WORKSPACE_ROOT / "MolJSON" / "LICENSE"
MOLJSON_DATA_LICENSE_SRC = WORKSPACE_ROOT / "MolJSON-data" / "LICENSE"
COMMONCHEM_LICENSE_SRC = WORKSPACE_ROOT / "CommonChem" / "LICENSE"
COMMONCHEM_REPO_SCHEMA_SRC = WORKSPACE_ROOT / "CommonChem" / "schema" / "commonchem.json"
if str(MOLJSON_SRC) not in sys.path:
    sys.path.insert(0, str(MOLJSON_SRC))

from moljson import GetSchema  # noqa: E402


SCRIPT_FILES = [
    ROOT / "benchmark_utils.py",
    ROOT / "prepare_api500_subset.py",
    ROOT / "run_openai_api_benchmark.py",
    ROOT / "evaluate_benchmark.py",
    ROOT / "prepare_github_release.py",
]

TOP_LEVEL_FILES = [
    ARTIFACTS_DIR / "queries.jsonl",
    ARTIFACTS_DIR / "selected_ids.txt",
    ARTIFACTS_DIR / "subset_summary.json",
    ARTIFACTS_DIR / "results.csv",
    ARTIFACTS_DIR / "summary_by_family.csv",
    ARTIFACTS_DIR / "paired_stats.csv",
    ARTIFACTS_DIR / "api_run_log.csv",
    ARTIFACTS_DIR / "api_run_summary.json",
    ARTIFACTS_DIR / "blog_post.md",
    ARTIFACTS_DIR / "linkedin_reply.md",
]


def ensure_dirs() -> None:
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    (TARGET_DIR / "scripts").mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "data" / "query_specs").mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "schemas").mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "outputs" / "responses").mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "third_party_licenses").mkdir(parents=True, exist_ok=True)


def release_path_for_response(name: str) -> str:
    return f"outputs/responses/{name}"


def release_path_for_spec(name: str) -> str:
    return f"data/query_specs/{name}"


def rewrite_manifest() -> None:
    src = ARTIFACTS_DIR / "queries.jsonl"
    dst = TARGET_DIR / "data" / "queries.jsonl"
    with src.open("r", encoding="utf-8") as in_handle, dst.open("w", encoding="utf-8") as out_handle:
        for line in in_handle:
            row = json.loads(line)
            row["spec_path"] = release_path_for_spec(Path(row["spec_path"]).name)
            row["response_path"] = release_path_for_response(Path(row["response_path"]).name)
            out_handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def rewrite_query_specs() -> None:
    src_dir = ARTIFACTS_DIR / "query_specs"
    dst_dir = TARGET_DIR / "data" / "query_specs"
    for src_path in sorted(src_dir.glob("*.json")):
        row = json.loads(src_path.read_text(encoding="utf-8"))
        row["spec_path"] = release_path_for_spec(src_path.name)
        row["response_path"] = release_path_for_response(Path(row["response_path"]).name)
        (dst_dir / src_path.name).write_text(json.dumps(row, indent=2), encoding="utf-8")


def rewrite_csv_with_relative_response_paths(src: Path, dst: Path) -> None:
    with src.open("r", newline="", encoding="utf-8") as in_handle, dst.open("w", newline="", encoding="utf-8") as out_handle:
        reader = csv.DictReader(in_handle)
        writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if "response_path" in row and row["response_path"]:
                row["response_path"] = release_path_for_response(Path(row["response_path"]).name)
            writer.writerow(row)


def copy_responses() -> None:
    src_dir = ARTIFACTS_DIR / "responses"
    dst_dir = TARGET_DIR / "outputs" / "responses"
    for src_path in sorted(src_dir.glob("*.txt")):
        shutil.copy2(src_path, dst_dir / src_path.name)


def copy_scripts() -> None:
    for src_path in SCRIPT_FILES:
        shutil.copy2(src_path, TARGET_DIR / "scripts" / src_path.name)
    shutil.copytree(
        MOLJSON_PACKAGE_SRC,
        TARGET_DIR / "scripts" / "moljson",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(MOLJSON_LICENSE_SRC, TARGET_DIR / "scripts" / "moljson" / "LICENSE")


def copy_simple_files() -> None:
    shutil.copy2(ARTIFACTS_DIR / "selected_ids.txt", TARGET_DIR / "data" / "selected_ids.txt")
    shutil.copy2(ARTIFACTS_DIR / "subset_summary.json", TARGET_DIR / "data" / "subset_summary.json")
    shutil.copy2(ARTIFACTS_DIR / "summary_by_family.csv", TARGET_DIR / "outputs" / "summary_by_family.csv")
    shutil.copy2(ARTIFACTS_DIR / "paired_stats.csv", TARGET_DIR / "outputs" / "paired_stats.csv")
    shutil.copy2(ARTIFACTS_DIR / "api_run_summary.json", TARGET_DIR / "outputs" / "api_run_summary.json")


def copy_third_party_licenses() -> None:
    shutil.copy2(MOLJSON_LICENSE_SRC, TARGET_DIR / "third_party_licenses" / "MolJSON-LICENSE")
    shutil.copy2(MOLJSON_DATA_LICENSE_SRC, TARGET_DIR / "third_party_licenses" / "MolJSON-data-LICENSE")
    shutil.copy2(COMMONCHEM_LICENSE_SRC, TARGET_DIR / "third_party_licenses" / "CommonChem-LICENSE")


def dump_schemas() -> None:
    commonchem_src = ARTIFACTS_DIR / "commonchem_strict_schema.json"
    shutil.copy2(commonchem_src, TARGET_DIR / "schemas" / "commonchem_strict_schema.json")
    shutil.copy2(COMMONCHEM_REPO_SCHEMA_SRC, TARGET_DIR / "schemas" / "commonchem_repo_schema.json")
    moljson_schema = GetSchema()
    (TARGET_DIR / "schemas" / "moljson_schema.json").write_text(
        json.dumps(moljson_schema, indent=2),
        encoding="utf-8",
    )


def rewrite_markdown_links(text: str) -> str:
    replacements = {
        "[summary_by_family.csv](/Users/frederickparsons/Documents/CommonChem/experiments/commonchem_vs_moljson/artifacts_api500/summary_by_family.csv)": "[summary_by_family.csv](summary_by_family.csv)",
        "[paired_stats.csv](/Users/frederickparsons/Documents/CommonChem/experiments/commonchem_vs_moljson/artifacts_api500/paired_stats.csv)": "[paired_stats.csv](paired_stats.csv)",
        "[results.csv](/Users/frederickparsons/Documents/CommonChem/experiments/commonchem_vs_moljson/artifacts_api500/results.csv)": "[results.csv](results.csv)",
        "[api_run_summary.json](/Users/frederickparsons/Documents/CommonChem/experiments/commonchem_vs_moljson/artifacts_api500/api_run_summary.json)": "[api_run_summary.json](api_run_summary.json)",
        "[commonchem_strict_schema.json](/Users/frederickparsons/Documents/CommonChem/experiments/commonchem_vs_moljson/artifacts_api500/commonchem_strict_schema.json)": "[commonchem_strict_schema.json](../schemas/commonchem_strict_schema.json)",
        "[blog_post.md](/Users/frederickparsons/Documents/CommonChem/experiments/commonchem_vs_moljson/artifacts_api500/blog_post.md)": "[blog_post.md](blog_post.md)",
        "[linkedin_reply.md](/Users/frederickparsons/Documents/CommonChem/experiments/commonchem_vs_moljson/artifacts_api500/linkedin_reply.md)": "[linkedin_reply.md](linkedin_reply.md)",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def rewrite_markdown_outputs() -> None:
    for name in ("blog_post.md", "linkedin_reply.md"):
        src = ARTIFACTS_DIR / name
        text = src.read_text(encoding="utf-8")
        text = rewrite_markdown_links(text)
        (TARGET_DIR / "outputs" / name).write_text(text, encoding="utf-8")


def write_requirements() -> None:
    text = """openai
numpy
pandas
scipy
matplotlib
rdkit
"""
    (TARGET_DIR / "requirements.txt").write_text(text, encoding="utf-8")


def write_metadata() -> None:
    summary = json.loads((ARTIFACTS_DIR / "api_run_summary.json").read_text(encoding="utf-8"))
    paired_rows: list[dict[str, str]] = []
    with (ARTIFACTS_DIR / "paired_stats.csv").open("r", newline="", encoding="utf-8") as handle:
        paired_rows = list(csv.DictReader(handle))
    overall = next(row for row in paired_rows if row["family"] == "overall")
    metadata = {
        "run_date": "2026-05-05",
        "model": summary["model"],
        "query_count": summary["query_count"],
        "completed_ok": summary["completed_ok"],
        "input_tokens": summary["input_tokens"],
        "output_tokens": summary["output_tokens"],
        "total_tokens": summary["total_tokens"],
        "overall_commonchem_accuracy": float(overall["commonchem_accuracy"]),
        "overall_moljson_accuracy": float(overall["moljson_accuracy"]),
        "overall_diff_commonchem_minus_moljson": float(overall["accuracy_diff_commonchem_minus_moljson"]),
        "overall_mcnemar_p_value": float(overall["mcnemar_p_value"]),
    }
    (TARGET_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_readme() -> None:
    summary = json.loads((ARTIFACTS_DIR / "api_run_summary.json").read_text(encoding="utf-8"))
    with (ARTIFACTS_DIR / "paired_stats.csv").open("r", newline="", encoding="utf-8") as handle:
        paired_rows = list(csv.DictReader(handle))
    overall = next(row for row in paired_rows if row["family"] == "overall")
    family_rows = [row for row in paired_rows if row["family"] != "overall"]
    family_labels = {
        "translation_to_graph": "Text -> graph",
        "translation_graph_to_smiles": "Graph -> SMILES",
        "translation_to_smiles": "Graph -> SMILES",
        "shortest_path": "Shortest path",
        "constrained_generation": "Constrained generation",
    }
    family_table = "\n".join(
        [
            "| Slice | Pairs | MolJSON | CommonChem | CommonChem - MolJSON |",
            "| --- | ---: | ---: | ---: | ---: |",
            *[
                (
                    f"| {family_labels.get(row['family'], row['family'])} | {row['n_pairs']} | "
                    f"{100.0 * float(row['moljson_accuracy']):.1f}% | "
                    f"{100.0 * float(row['commonchem_accuracy']):.1f}% | "
                    f"{100.0 * float(row['accuracy_diff_commonchem_minus_moljson']):+.1f} pts |"
                )
                for row in family_rows
            ],
        ]
    )
    readme = f"""# CommonChem vs MolJSON on GPT-5.4

This directory is a GitHub-ready bundle of the exact code, prompts, schemas, raw model outputs, and scored results used for a 500-query follow-up to *Molecular Representations for Large Language Models*.

The goal of the run was narrow: compare `CommonChem` against `MolJSON` on a representative subset of the `MolJSON-data` benchmark, using strict structured outputs through the OpenAI Responses API.

## Headline result

- Queries: `{summary["query_count"]}` total API calls (`250` paired prompts)
- Model: `{summary["model"]}`
- Overall accuracy: MolJSON `{100.0 * float(overall["moljson_accuracy"]):.1f}%`, CommonChem `{100.0 * float(overall["commonchem_accuracy"]):.1f}%`
- Paired difference: CommonChem minus MolJSON = `{100.0 * float(overall["accuracy_diff_commonchem_minus_moljson"]):+.1f}` points
- 95% paired bootstrap CI: `{100.0 * float(overall["diff_ci_low"]):+.1f}` to `{100.0 * float(overall["diff_ci_high"]):+.1f}` points
- Exact McNemar `p = {float(overall["mcnemar_p_value"]):.3f}`

{family_table}

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
"""
    (TARGET_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    copy_scripts()
    rewrite_manifest()
    rewrite_query_specs()
    copy_responses()
    copy_simple_files()
    copy_third_party_licenses()
    rewrite_csv_with_relative_response_paths(
        ARTIFACTS_DIR / "results.csv",
        TARGET_DIR / "outputs" / "results.csv",
    )
    rewrite_csv_with_relative_response_paths(
        ARTIFACTS_DIR / "api_run_log.csv",
        TARGET_DIR / "outputs" / "api_run_log.csv",
    )
    dump_schemas()
    rewrite_markdown_outputs()
    write_requirements()
    write_metadata()
    write_readme()
    print(TARGET_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
