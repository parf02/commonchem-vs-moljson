from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

from benchmark_utils import (
    canonical_smiles,
    check_constrained_molecule,
    exact_mcnemar,
    paired_bootstrap_ci,
    parse_molecule_from_response,
    parse_wrapped_integer,
    parse_wrapped_string,
    wilson_interval,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_ROOT.parent if (SCRIPT_ROOT.parent / "data").exists() and (SCRIPT_ROOT.parent / "outputs").exists() else None
PROJECT_ROOT = RELEASE_ROOT or SCRIPT_ROOT
if RELEASE_ROOT is None:
    DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    DEFAULT_MANIFEST_PATH = DEFAULT_ARTIFACTS_DIR / "queries.jsonl"
    DEFAULT_RESULTS_CSV = DEFAULT_ARTIFACTS_DIR / "results.csv"
    DEFAULT_SUMMARY_CSV = DEFAULT_ARTIFACTS_DIR / "summary_by_family.csv"
    DEFAULT_PAIRED_CSV = DEFAULT_ARTIFACTS_DIR / "paired_stats.csv"
    DEFAULT_BLOG_POST_PATH = PROJECT_ROOT / "blog_post.md"
    DEFAULT_PLOT_PATH = DEFAULT_ARTIFACTS_DIR / "accuracy_by_family.png"
    DEFAULT_IDS_PATH = DEFAULT_ARTIFACTS_DIR / "selected_ids.txt"
else:
    DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "queries.jsonl"
    DEFAULT_RESULTS_CSV = PROJECT_ROOT / "outputs" / "results.csv"
    DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "summary_by_family.csv"
    DEFAULT_PAIRED_CSV = PROJECT_ROOT / "outputs" / "paired_stats.csv"
    DEFAULT_BLOG_POST_PATH = PROJECT_ROOT / "outputs" / "blog_post.md"
    DEFAULT_PLOT_PATH = PROJECT_ROOT / "outputs" / "accuracy_by_family.png"
    DEFAULT_IDS_PATH = PROJECT_ROOT / "data" / "selected_ids.txt"

FAMILY_LABELS = {
    "translation_to_graph": "Translation: text to graph",
    "translation_smiles_to_graph": "Translation: SMILES to graph",
    "translation_graph_to_smiles": "Translation: graph to SMILES",
    "shortest_path": "Shortest path",
    "constrained_generation": "Constrained generation",
    "overall": "Overall",
}
REPRESENTATION_LABELS = {"moljson": "MolJSON", "commonchem": "CommonChem"}

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the selected CommonChem vs MolJSON benchmark subset.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--paired-csv", default=str(DEFAULT_PAIRED_CSV))
    parser.add_argument("--blog-post", default=str(DEFAULT_BLOG_POST_PATH))
    parser.add_argument("--plot-path", default=str(DEFAULT_PLOT_PATH))
    parser.add_argument("--ids-file", default=str(DEFAULT_IDS_PATH))
    parser.add_argument("--usage-summary-json")
    parser.add_argument("--execution-mode", choices=["subagent", "responses_api"], default="subagent")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--title", default="CommonChem vs MolJSON on GPT-5.4")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_selected_ids(path: Path | None) -> set[str] | None:
    if path is None or not path.exists():
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_usage_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_row(query: dict[str, Any]) -> dict[str, Any]:
    response_path = Path(query["response_path"])
    row = {
        "id": query["id"],
        "pair_id": query["pair_id"],
        "family": query["family"],
        "representation": query["representation"],
        "evaluation_kind": query["evaluation_kind"],
        "source_smiles": query.get("expected_smiles") or query.get("metadata", {}).get("source_smiles"),
        "response_path": str(response_path),
        "has_response": response_path.exists(),
        "is_correct": False,
        "predicted_smiles": None,
        "predicted_integer": None,
        "error": None,
        "failure_details": None,
    }

    if not response_path.exists():
        row["error"] = "missing_response_file"
        return row

    response_text = response_path.read_text(encoding="utf-8").strip()
    row["response_text"] = response_text
    if not response_text:
        row["error"] = "empty_response"
        return row

    try:
        if query["evaluation_kind"] == "translation_to_graph":
            mol = parse_molecule_from_response(response_text, query["representation"])
            pred_smiles = Chem.MolToSmiles(mol, canonical=True)
            row["predicted_smiles"] = pred_smiles
            row["is_correct"] = pred_smiles == canonical_smiles(query["expected_smiles"])
            if not row["is_correct"]:
                row["error"] = "smiles_mismatch"
            return row

        if query["evaluation_kind"] == "translation_to_smiles":
            pred_text = parse_wrapped_string(response_text, "smiles")
            pred_smiles = canonical_smiles(pred_text)
            row["predicted_smiles"] = pred_smiles
            row["is_correct"] = pred_smiles == canonical_smiles(query["expected_smiles"])
            if not row["is_correct"]:
                row["error"] = "smiles_mismatch"
            return row

        if query["evaluation_kind"] == "shortest_path":
            pred_int = parse_wrapped_integer(response_text)
            row["predicted_integer"] = pred_int
            row["is_correct"] = int(pred_int) == int(query["expected_integer"])
            if not row["is_correct"]:
                row["error"] = "integer_mismatch"
            return row

        if query["evaluation_kind"] == "constrained_generation":
            mol = parse_molecule_from_response(response_text, query["representation"])
            pred_smiles = Chem.MolToSmiles(mol, canonical=True)
            row["predicted_smiles"] = pred_smiles
            ok, failures = check_constrained_molecule(mol, query["constraints"])
            row["is_correct"] = ok
            if not ok:
                row["error"] = "constraint_failure"
                row["failure_details"] = "; ".join(failures)
            return row

        row["error"] = f"unsupported_evaluation_kind:{query['evaluation_kind']}"
        return row
    except Exception as exc:
        row["error"] = f"{exc.__class__.__name__}: {exc}"
        return row


def summarize_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for family in list(FAMILY_LABELS.keys()):
        if family == "overall":
            family_df = df
        else:
            family_df = df[df["family"] == family]
        if family_df.empty:
            continue
        for representation in ("moljson", "commonchem"):
            sub = family_df[family_df["representation"] == representation]
            successes = int(sub["is_correct"].sum())
            total = int(len(sub))
            if total == 0:
                continue
            lo, hi = wilson_interval(successes, total)
            records.append(
                {
                    "family": family,
                    "representation": representation,
                    "successes": successes,
                    "total": total,
                    "accuracy": (successes / total) if total else np.nan,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_paired(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for family in list(FAMILY_LABELS.keys()):
        if family == "overall":
            family_df = df
        else:
            family_df = df[df["family"] == family]
        pivot = (
            family_df.pivot_table(
                index="pair_id",
                columns="representation",
                values="is_correct",
                aggfunc="first",
            )
            .dropna()
            .astype(int)
        )
        if pivot.empty:
            continue
        commonchem = pivot["commonchem"].to_numpy(dtype=int)
        moljson = pivot["moljson"].to_numpy(dtype=int)
        diff = float(commonchem.mean() - moljson.mean())
        diff_lo, diff_hi = paired_bootstrap_ci(commonchem, moljson)
        mcnemar = exact_mcnemar(commonchem, moljson)
        records.append(
            {
                "family": family,
                "n_pairs": int(len(pivot)),
                "commonchem_accuracy": float(commonchem.mean()),
                "moljson_accuracy": float(moljson.mean()),
                "accuracy_diff_commonchem_minus_moljson": diff,
                "diff_ci_low": diff_lo,
                "diff_ci_high": diff_hi,
                "discordant_commonchem_only_correct": int(mcnemar["b10"]),
                "discordant_moljson_only_correct": int(mcnemar["b01"]),
                "mcnemar_p_value": float(mcnemar["p_value"]),
            }
        )
    return pd.DataFrame.from_records(records)


def make_plot(summary_df: pd.DataFrame, *, plot_path: Path, title: str) -> None:
    plot_df = summary_df[summary_df["family"] != "overall"].copy()
    families = [family for family in FAMILY_LABELS if family != "overall" and family in set(plot_df["family"])]
    x = np.arange(len(families))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, representation in [(-width / 2, "moljson"), (width / 2, "commonchem")]:
        sub = plot_df[plot_df["representation"] == representation].set_index("family").loc[families]
        y = sub["accuracy"].to_numpy(dtype=float)
        lower = y - sub["ci_low"].to_numpy(dtype=float)
        upper = sub["ci_high"].to_numpy(dtype=float) - y
    ax.bar(
            x + offset,
            y,
            width=width,
            label=REPRESENTATION_LABELS[representation],
            color="#4c78a8" if representation == "moljson" else "#f58518",
            yerr=np.vstack([lower, upper]),
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS[f] for f in families], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def format_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def format_p(x: float) -> str:
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def write_blog_post(
    queries: list[dict[str, Any]],
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    *,
    blog_post_path: Path,
    title: str,
    execution_mode: str,
    model: str,
    usage_summary: dict[str, Any] | None,
) -> None:
    pair_count = len({query["pair_id"] for query in queries})
    call_count = len(queries)
    complete_count = int(results_df["has_response"].sum())
    successful_parses = int(results_df["is_correct"].sum())

    def pair_count_for(families: set[str], *, input_format: str | None = None) -> int:
        return len(
            {
                query["pair_id"]
                for query in queries
                if query["family"] in families
                and (input_format is None or query.get("metadata", {}).get("input_format") == input_format)
            }
        )

    family_pair_counts = {
        "translation_to_graph": pair_count_for({"translation_to_graph", "translation_smiles_to_graph"}),
        "translation_to_graph_smiles": pair_count_for({"translation_to_graph"}, input_format="smiles"),
        "translation_to_graph_iupac": pair_count_for({"translation_to_graph"}, input_format="iupac"),
        "translation_graph_to_smiles": pair_count_for({"translation_graph_to_smiles"}),
        "shortest_path": pair_count_for({"shortest_path"}),
        "constrained_generation": pair_count_for({"constrained_generation"}),
    }
    family_order = [family for family in FAMILY_LABELS if family != "overall" and family in set(results_df["family"])]

    summary_rows: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        if row["family"] == "overall":
            continue
        summary_rows.append(
            {
                "Task family": FAMILY_LABELS.get(row["family"], row["family"]),
                "Format": REPRESENTATION_LABELS[row["representation"]],
                "Accuracy": f"{row['successes']}/{row['total']} ({format_pct(row['accuracy'])})",
                "95% CI": f"{format_pct(row['ci_low'])} to {format_pct(row['ci_high'])}",
            }
        )

    paired_rows: list[dict[str, Any]] = []
    for _, row in paired_df.iterrows():
        paired_rows.append(
            {
                "Task family": FAMILY_LABELS.get(row["family"], row["family"]),
                "Pairs": int(row["n_pairs"]),
                "CommonChem": format_pct(row["commonchem_accuracy"]),
                "MolJSON": format_pct(row["moljson_accuracy"]),
                "Diff (CC-MJ)": format_pct(row["accuracy_diff_commonchem_minus_moljson"]),
                "95% paired CI": f"{format_pct(row['diff_ci_low'])} to {format_pct(row['diff_ci_high'])}",
                "McNemar p": format_p(row["mcnemar_p_value"]),
            }
        )

    overall = paired_df[paired_df["family"] == "overall"].iloc[0]

    if execution_mode == "responses_api":
        mode_text = (
            f"Every query was executed through the OpenAI Responses API using `{model}` with strict JSON-schema outputs."
        )
    else:
        mode_text = (
            f"Every query was executed with a separate `{model}` subagent. This is a paired comparison, but not an exact reproduction of the paper's structured-output Responses API setup."
        )

    usage_text = ""
    if usage_summary:
        usage_text = (
            "\nAPI usage for this run:\n\n"
            f"- Completed calls: {usage_summary.get('completed_ok', 0)}\n"
            f"- Failed calls: {usage_summary.get('completed_error', 0)}\n"
            f"- Input tokens: {int(usage_summary.get('input_tokens', 0)):,}\n"
            f"- Output tokens: {int(usage_summary.get('output_tokens', 0)):,}\n"
            f"- Total tokens: {int(usage_summary.get('total_tokens', 0)):,}\n"
            f"- Wall-clock runtime: {float(usage_summary.get('elapsed_s', 0.0)) / 60.0:.1f} minutes\n"
        )

    if call_count == 500 and family_pair_counts["translation_to_graph"] == 40 and family_pair_counts["constrained_generation"] == 120:
        subset_text = f"""The 500-call subset was designed to maximize early signal while keeping the comparison paired.

- {family_pair_counts["translation_to_graph_smiles"]} `SMILES -> graph` pairs from `translation_large`, stratified across heavy-atom bins and ring counts
- {family_pair_counts["translation_to_graph_iupac"]} `IUPAC -> graph` pairs from `translation_large`, using the same structure strata
- {family_pair_counts["translation_graph_to_smiles"]} `graph -> SMILES` pairs from `translation_large`, with extra weight on graph interpretation
- {family_pair_counts["shortest_path"]} shortest-path pairs covering path lengths from 2 to 18 bonds, then filled with diverse extra cases
- {family_pair_counts["constrained_generation"]} constrained-generation pairs, with 24 tasks from each benchmark subcategory

Translation and shortest-path diversity were built from molecule size, ring count, and path-length coverage. Constrained-generation diversity was chosen by farthest-point sampling over witness-molecule and constraint features."""
    else:
        subset_lines = []
        if family_pair_counts["translation_to_graph"]:
            subset_lines.append(f"- {family_pair_counts['translation_to_graph']} text-to-graph translation pairs")
        if family_pair_counts["translation_graph_to_smiles"]:
            subset_lines.append(f"- {family_pair_counts['translation_graph_to_smiles']} graph-to-SMILES translation pairs")
        if family_pair_counts["shortest_path"]:
            subset_lines.append(f"- {family_pair_counts['shortest_path']} shortest-path pairs")
        if family_pair_counts["constrained_generation"]:
            subset_lines.append(f"- {family_pair_counts['constrained_generation']} constrained-generation pairs")
        subset_text = "The subset keeps the comparison paired while covering multiple benchmark families.\n\n" + "\n".join(subset_lines)

    interpretation_bullets = [
        "- if CommonChem beats MolJSON overall, the hypothesis gets early support under this serving setup",
        "- if MolJSON still wins, then MolJSON's schema design is likely doing more work than prior ecosystem familiarity",
        "- if the two are close, the practical conclusion is that CommonChem is competitive enough to justify a larger run",
    ]
    if execution_mode == "subagent":
        interpretation_bullets.append(
            "- because this run did not use strict schema enforcement, serialization drift matters more than it would in the paper's setup"
        )
    else:
        interpretation_bullets.append(
            "- because this run uses strict structured outputs, remaining differences are less likely to be explained by trivial JSON-shape drift"
        )

    limitations = [
        "- This is a subset, not the full benchmark from the paper.",
        f"- This run uses `{model}`, not the paper's `GPT-5` model label.",
        "- I excluded IUPAC-output evaluation from this comparison to stay focused on graph-like behavior and avoid external IUPAC parsing dependencies.",
        "- The CommonChem side uses a minimal RDKit-derived serializer and strict schema tailored to this benchmark's no-stereochemistry setting.",
    ]
    if execution_mode == "subagent":
        limitations.insert(
            1,
            "- Queries were executed via subagents rather than the Responses API, so strict schema enforcement and exact API token accounting were unavailable.",
        )
    else:
        limitations.insert(
            1,
            "- The benchmark provides MolJSON schemas directly; the CommonChem schema here is a benchmark-oriented strict schema built from the CommonChem core representation rather than an official paper-supplied evaluation schema.",
        )

    text = f"""# {title}

## Executive summary

I ran a paired head-to-head comparison of CommonChem and MolJSON on a selected subset of the MolJSON benchmark using `{model}` on {date.today().isoformat()}.

This run covered {pair_count} paired benchmark prompts ({call_count} total model calls), split across:

- {family_pair_counts["translation_to_graph"]} text-to-graph translation pairs
- {family_pair_counts["translation_graph_to_smiles"]} `graph -> SMILES` translation pairs
- {family_pair_counts["shortest_path"]} shortest-path reasoning pairs
- {family_pair_counts["constrained_generation"]} constrained-generation pairs

{mode_text}

The overall paired result was:

- CommonChem accuracy: {format_pct(float(overall["commonchem_accuracy"]))}
- MolJSON accuracy: {format_pct(float(overall["moljson_accuracy"]))}
- Difference (CommonChem minus MolJSON): {format_pct(float(overall["accuracy_diff_commonchem_minus_moljson"]))}
- 95% paired bootstrap CI: {format_pct(float(overall["diff_ci_low"]))} to {format_pct(float(overall["diff_ci_high"]))}
- Exact McNemar p-value: {format_p(float(overall["mcnemar_p_value"]))}
{usage_text}

## Why this subset

{subset_text}

## Method

- Benchmark source: `MolJSON-data`
- CommonChem source format: generated from the benchmark SMILES using RDKit
- MolJSON source format: generated with the reference MolJSON package
- Evaluation:
  - translation to graph: parse output to RDKit, convert to canonical SMILES, compare to ground truth
  - translation to SMILES: canonicalize predicted SMILES, compare to ground truth
  - shortest path: exact integer match
  - constrained generation: parse output to RDKit, then verify all graph constraints directly
- Statistics:
  - Wilson confidence intervals for per-format accuracy
  - paired bootstrap confidence intervals for accuracy differences
  - exact McNemar test on discordant pairs

## Results by task

{markdown_table(pd.DataFrame(summary_rows), ["Task family", "Format", "Accuracy", "95% CI"])}

## Paired comparisons

{markdown_table(pd.DataFrame(paired_rows), ["Task family", "Pairs", "CommonChem", "MolJSON", "Diff (CC-MJ)", "95% paired CI", "McNemar p"])}

## Interpretation

The headline question was whether CommonChem would match or beat MolJSON because CommonChem is older and likely more represented in public training data and RDKit-adjacent material.

This run suggests:

{chr(10).join(interpretation_bullets)}

The paired results matter more than the raw percentages because every CommonChem question had a matched MolJSON counterpart over the same underlying molecule or constraint set.

## Limitations

{chr(10).join(limitations)}

## Takeaway

This run produced {complete_count} completed calls and {successful_parses} correct answers across both formats.

For decision-making today, the useful readout is the paired overall comparison:

- CommonChem: {format_pct(float(overall["commonchem_accuracy"]))}
- MolJSON: {format_pct(float(overall["moljson_accuracy"]))}
- Difference: {format_pct(float(overall["accuracy_diff_commonchem_minus_moljson"]))}

If the goal is a publishable or externally shareable claim, the next step is straightforward: rerun the same harness on a larger slice, then move to the full benchmark once the direction of effect is stable.
"""
    blog_post_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    results_csv = Path(args.results_csv)
    summary_csv = Path(args.summary_csv)
    paired_csv = Path(args.paired_csv)
    blog_post_path = Path(args.blog_post)
    plot_path = Path(args.plot_path)

    queries = load_manifest(manifest_path)
    selected_ids = load_selected_ids(Path(args.ids_file)) if args.ids_file else None
    if selected_ids is not None:
        queries = [query for query in queries if query["id"] in selected_ids]
    usage_summary = load_usage_summary(Path(args.usage_summary_json)) if args.usage_summary_json else None

    results = [evaluate_row(query) for query in queries]
    results_df = pd.DataFrame.from_records(results)
    summary_df = summarize_accuracy(results_df)
    paired_df = summarize_paired(results_df)

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    paired_df.to_csv(paired_csv, index=False)
    if not args.skip_plot:
        try:
            make_plot(summary_df, plot_path=plot_path, title=args.title)
        except Exception as exc:
            print(f"Plot generation failed: {exc}")
    write_blog_post(
        queries,
        results_df,
        summary_df,
        paired_df,
        blog_post_path=blog_post_path,
        title=args.title,
        execution_mode=args.execution_mode,
        model=args.model,
        usage_summary=usage_summary,
    )

    print(summary_df)
    print(paired_df)
    print(f"Wrote {blog_post_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
