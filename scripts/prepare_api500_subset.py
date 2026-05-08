from __future__ import annotations

import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from benchmark_utils import expand_ring_size_spec, mol_to_commonchem, prompt_object_text


SEED = 20260506
TRANSLATION_SMILES_TO_GRAPH_PAIRS = 20
TRANSLATION_IUPAC_TO_GRAPH_PAIRS = 20
TRANSLATION_GRAPH_TO_SMILES_PAIRS = 40
SHORTEST_PATH_PAIRS = 50
CONSTRAINED_PAIRS = 120
CONSTRAINED_PER_SUBCATEGORY = 24

ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = ROOT.parent if (ROOT.parent / "data").exists() and (ROOT.parent / "outputs").exists() else None
PROJECT_ROOT = RELEASE_ROOT or ROOT
WORKSPACE_ROOT = PROJECT_ROOT.parent if RELEASE_ROOT else ROOT.parents[1]
DATA_ROOT = WORKSPACE_ROOT / "MolJSON-data"
OUT_ROOT = PROJECT_ROOT / "artifacts_api500"
QUERY_SPECS_DIR = OUT_ROOT / "query_specs"
RESPONSES_DIR = OUT_ROOT / "responses"
MANIFEST_PATH = OUT_ROOT / "queries.jsonl"
SELECTED_IDS_PATH = OUT_ROOT / "selected_ids.txt"
SUMMARY_PATH = OUT_ROOT / "subset_summary.json"


def load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    QUERY_SPECS_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)


def response_contract_for(representation: str, evaluation_kind: str) -> str:
    if evaluation_kind == "translation_to_graph" or evaluation_kind == "constrained_generation":
        if representation == "moljson":
            return (
                "Return only a valid JSON object in MolJSON format. "
                "No markdown, no code fences, no comments, no surrounding prose. "
                "Use the MolJSON keys atoms, bonds, charges, and aromatic_n_h."
            )
        if representation == "commonchem":
            return (
                "Return only a valid JSON object in CommonChem format. "
                "No markdown, no code fences, no comments, no surrounding prose. "
                'Use top-level keys "commonchem" and "molecules". Set "commonchem" to 10. '
                'Return exactly one molecule. Atom objects use "z" and "impHs" when nonzero, plus "chg" when nonzero. '
                'Bond objects use "type" and "atoms".'
            )
    if evaluation_kind == "translation_to_smiles":
        return (
            'Return only a valid JSON object of the form {"smiles":"..."} with no markdown, code fences, comments, or prose.'
        )
    if evaluation_kind == "shortest_path":
        return (
            'Return only a valid JSON object of the form {"integer":"N"} where N is the shortest-path length, '
            "with no markdown, code fences, comments, or prose."
        )
    raise ValueError(f"unsupported evaluation kind: {evaluation_kind}")


def rdkit_stats(smiles: str) -> dict[str, int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid smiles: {smiles}")
    return {
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "ring_count": int(mol.GetRingInfo().NumRings()),
    }


def heavy_atom_bin(heavy_atoms: int) -> str:
    if heavy_atoms <= 13:
        return "10_13"
    if heavy_atoms <= 17:
        return "14_17"
    if heavy_atoms <= 21:
        return "18_21"
    if heavy_atoms <= 25:
        return "22_25"
    return "26_30"


def translation_stratum(info: dict[str, Any]) -> tuple[str, int]:
    return (heavy_atom_bin(int(info["heavy_atoms"])), int(info["ring_count"]))


def farthest_point_sample(
    rows: list[dict[str, Any]],
    *,
    n: int,
    feature_fn,
    seed_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []

    feats = np.asarray([feature_fn(row) for row in rows], dtype=float)
    if feats.ndim != 2:
        raise ValueError("feature_fn must return a fixed-length vector")
    mins = feats.min(axis=0)
    maxs = feats.max(axis=0)
    span = np.where((maxs - mins) > 0, maxs - mins, 1.0)
    feats = (feats - mins) / span

    selected_idx: list[int] = []
    if seed_rows:
        row_to_idx = {id(row): i for i, row in enumerate(rows)}
        for row in seed_rows:
            idx = row_to_idx.get(id(row))
            if idx is not None and idx not in selected_idx:
                selected_idx.append(idx)

    if not selected_idx:
        selected_idx = [0]

    chosen = list(selected_idx)
    while len(chosen) < min(n, len(rows)):
        selected_feats = feats[chosen]
        min_dists: list[float] = []
        for idx in range(len(rows)):
            if idx in chosen:
                min_dists.append(-1.0)
                continue
            d = np.linalg.norm(selected_feats - feats[idx], axis=1)
            min_dists.append(float(np.min(d)))
        next_idx = int(np.argmax(min_dists))
        if next_idx in chosen:
            break
        chosen.append(next_idx)

    return [rows[idx] for idx in chosen[:n]]


def choose_translation_rows(
    rows: list[dict[str, Any]],
    *,
    per_stratum: int,
    used_smiles: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[translation_stratum(row["__meta_info"])].append(row)

    selected: list[dict[str, Any]] = []
    for stratum in sorted(grouped):
        candidates = [
            row for row in grouped[stratum] if row["__meta_info"]["smiles"] not in used_smiles
        ]
        if len(candidates) < per_stratum:
            candidates = grouped[stratum]
        ordered = sorted(candidates, key=lambda row: row["uuid"])
        picks = ordered if len(ordered) <= per_stratum else rng.sample(ordered, per_stratum)
        for row in picks:
            if row not in selected:
                selected.append(row)
                used_smiles.add(row["__meta_info"]["smiles"])
    return selected


def build_translation_queries(rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl_gz(DATA_ROOT / "questions" / "translation_large.jsonl.gz")
    enriched: list[dict[str, Any]] = []
    for row in rows:
        smiles = str(row["meta"]["molecule"]["smiles"])
        info = {"smiles": smiles, **rdkit_stats(smiles)}
        row = dict(row)
        row["__meta_info"] = info
        enriched.append(row)

    by_direction: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_direction[(row["input_format"], row["output_format"])].append(row)

    used_smiles: set[str] = set()
    selected_smiles_to_graph = choose_translation_rows(
        by_direction[("smiles", "graph")],
        per_stratum=1,
        used_smiles=used_smiles,
        rng=rng,
    )
    selected_iupac_to_graph = choose_translation_rows(
        by_direction[("iupac", "graph")],
        per_stratum=1,
        used_smiles=used_smiles,
        rng=rng,
    )
    selected_graph_to_smiles = choose_translation_rows(
        by_direction[("graph", "smiles")],
        per_stratum=2,
        used_smiles=used_smiles,
        rng=rng,
    )

    if len(selected_smiles_to_graph) != TRANSLATION_SMILES_TO_GRAPH_PAIRS:
        raise RuntimeError(f"expected {TRANSLATION_SMILES_TO_GRAPH_PAIRS} smiles->graph pairs")
    if len(selected_iupac_to_graph) != TRANSLATION_IUPAC_TO_GRAPH_PAIRS:
        raise RuntimeError(f"expected {TRANSLATION_IUPAC_TO_GRAPH_PAIRS} iupac->graph pairs")
    if len(selected_graph_to_smiles) != TRANSLATION_GRAPH_TO_SMILES_PAIRS:
        raise RuntimeError(f"expected {TRANSLATION_GRAPH_TO_SMILES_PAIRS} graph->smiles pairs")

    queries: list[dict[str, Any]] = []

    def add_graph_output_pair(row: dict[str, Any], idx: int, input_format: str) -> None:
        smiles = row["__meta_info"]["smiles"]
        if input_format == "smiles":
            commonchem_prompt = f"Convert the molecule from smiles to commonchem: {smiles}"
        elif input_format == "iupac":
            commonchem_prompt = f"Convert the molecule from iupac to commonchem: {row['meta']['input']}"
        else:
            raise ValueError(f"unsupported input format: {input_format}")
        shared_meta = {
            "source_dataset": "translation_large",
            "source_uuid": row["uuid"],
            "source_smiles": smiles,
            "input_format": input_format,
            "heavy_atoms": row["__meta_info"]["heavy_atoms"],
            "ring_count": row["__meta_info"]["ring_count"],
        }
        base_id = f"translation_to_graph_{input_format}_{idx:03d}"
        queries.append(
            {
                "id": f"{base_id}_moljson",
                "pair_id": base_id,
                "family": "translation_to_graph",
                "representation": "moljson",
                "evaluation_kind": "translation_to_graph",
                "prompt": row["prompt"],
                "response_contract": response_contract_for("moljson", "translation_to_graph"),
                "expected_smiles": smiles,
                "metadata": shared_meta,
            }
        )
        queries.append(
            {
                "id": f"{base_id}_commonchem",
                "pair_id": base_id,
                "family": "translation_to_graph",
                "representation": "commonchem",
                "evaluation_kind": "translation_to_graph",
                "prompt": commonchem_prompt,
                "response_contract": response_contract_for("commonchem", "translation_to_graph"),
                "expected_smiles": smiles,
                "metadata": shared_meta,
            }
        )

    def add_graph_input_pair(row: dict[str, Any], idx: int) -> None:
        smiles = row["__meta_info"]["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid smiles: {smiles}")
        commonchem_obj = mol_to_commonchem(mol)
        shared_meta = {
            "source_dataset": "translation_large",
            "source_uuid": row["uuid"],
            "source_smiles": smiles,
            "input_format": "graph",
            "heavy_atoms": row["__meta_info"]["heavy_atoms"],
            "ring_count": row["__meta_info"]["ring_count"],
        }
        base_id = f"translation_graph_to_smiles_{idx:03d}"
        queries.append(
            {
                "id": f"{base_id}_moljson",
                "pair_id": base_id,
                "family": "translation_graph_to_smiles",
                "representation": "moljson",
                "evaluation_kind": "translation_to_smiles",
                "prompt": row["prompt"],
                "response_contract": response_contract_for("moljson", "translation_to_smiles"),
                "expected_smiles": smiles,
                "metadata": shared_meta,
            }
        )
        queries.append(
            {
                "id": f"{base_id}_commonchem",
                "pair_id": base_id,
                "family": "translation_graph_to_smiles",
                "representation": "commonchem",
                "evaluation_kind": "translation_to_smiles",
                "prompt": f"Convert the molecule from commonchem to smiles: {prompt_object_text(commonchem_obj)}",
                "response_contract": response_contract_for("commonchem", "translation_to_smiles"),
                "expected_smiles": smiles,
                "metadata": shared_meta,
            }
        )

    for idx, row in enumerate(selected_smiles_to_graph, start=1):
        add_graph_output_pair(row, idx, "smiles")
    for idx, row in enumerate(selected_iupac_to_graph, start=1):
        add_graph_output_pair(row, idx, "iupac")
    for idx, row in enumerate(selected_graph_to_smiles, start=1):
        add_graph_input_pair(row, idx)

    summary = {
        "translation_to_graph_smiles_pairs": len(selected_smiles_to_graph),
        "translation_to_graph_iupac_pairs": len(selected_iupac_to_graph),
        "translation_graph_to_smiles_pairs": len(selected_graph_to_smiles),
    }
    return queries, summary


def build_shortest_path_queries(rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl_gz(DATA_ROOT / "questions" / "shortest_path_questions.jsonl.gz")
    graph_rows = [row for row in rows if row["input_format"] == "graph"]

    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in graph_rows:
        smiles = str(row["meta"]["molecule"]["smiles"])
        row = dict(row)
        row["__meta_info"] = {"smiles": smiles, **rdkit_stats(smiles)}
        by_length[int(row["answer"])].append(row)

    selected: list[dict[str, Any]] = []
    used_uuids: set[str] = set()

    for path_length in range(2, 19):
        candidates = sorted(by_length[path_length], key=lambda row: row["uuid"])
        low = [row for row in candidates if row["__meta_info"]["ring_count"] <= 1 and row["uuid"] not in used_uuids]
        high = [row for row in candidates if row["__meta_info"]["ring_count"] >= 2 and row["uuid"] not in used_uuids]
        if low:
            selected.append(low[0])
            used_uuids.add(low[0]["uuid"])
        elif candidates:
            for row in candidates:
                if row["uuid"] not in used_uuids:
                    selected.append(row)
                    used_uuids.add(row["uuid"])
                    break
        if high:
            selected.append(high[0])
            used_uuids.add(high[0]["uuid"])
        else:
            for row in candidates:
                if row["uuid"] not in used_uuids:
                    selected.append(row)
                    used_uuids.add(row["uuid"])
                    break

    remaining = [row for rows_ in by_length.values() for row in rows_ if row["uuid"] not in used_uuids]
    extra_needed = SHORTEST_PATH_PAIRS - len(selected)
    if extra_needed < 0:
        raise RuntimeError("over-selected shortest path rows")
    if extra_needed > 0:
        seed_rows = list(selected)
        selected.extend(
            farthest_point_sample(
                remaining,
                n=extra_needed,
                feature_fn=lambda row: [
                    int(row["answer"]),
                    row["__meta_info"]["heavy_atoms"],
                    row["__meta_info"]["ring_count"],
                ],
                seed_rows=seed_rows,
            )
        )

    if len(selected) != SHORTEST_PATH_PAIRS:
        raise RuntimeError(f"expected {SHORTEST_PATH_PAIRS} shortest-path pairs")

    queries: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        smiles = row["__meta_info"]["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid smiles: {smiles}")
        commonchem_obj = mol_to_commonchem(mol)
        commonchem_prompt = (
            "Determine the number of bonds along the shortest path connecting the two halogen atoms. "
            "Count each bond equally, including those directly attached to the halogen atoms.\n\n"
            f"{prompt_object_text(commonchem_obj)}\n\n"
            "Give your answer as an integer. Do not write any comments."
        )
        shared_meta = {
            "source_dataset": "shortest_path_questions",
            "source_uuid": row["uuid"],
            "source_smiles": smiles,
            "path_length": int(row["answer"]),
            "heavy_atoms": row["__meta_info"]["heavy_atoms"],
            "ring_count": row["__meta_info"]["ring_count"],
        }
        base_id = f"shortest_path_{idx:03d}"
        queries.append(
            {
                "id": f"{base_id}_moljson",
                "pair_id": base_id,
                "family": "shortest_path",
                "representation": "moljson",
                "evaluation_kind": "shortest_path",
                "prompt": row["prompt"],
                "response_contract": response_contract_for("moljson", "shortest_path"),
                "expected_integer": int(row["answer"]),
                "metadata": shared_meta,
            }
        )
        queries.append(
            {
                "id": f"{base_id}_commonchem",
                "pair_id": base_id,
                "family": "shortest_path",
                "representation": "commonchem",
                "evaluation_kind": "shortest_path",
                "prompt": commonchem_prompt,
                "response_contract": response_contract_for("commonchem", "shortest_path"),
                "expected_integer": int(row["answer"]),
                "metadata": shared_meta,
            }
        )

    summary = {"shortest_path_pairs": len(selected)}
    return queries, summary


def constrained_feature(row: dict[str, Any]) -> list[float]:
    constraints = row["constraints"]
    witness = str(row["witness"])
    stats = rdkit_stats(witness)
    ring_sizes = expand_ring_size_spec(constraints.get("ring_sizes"))
    shortest_paths = constraints.get("shortest_halogen_paths") or []
    path_values = []
    for spec in shortest_paths:
        path_values.append(float(spec["distance_bonds"]))
    while len(path_values) < 3:
        path_values.append(0.0)
    return [
        float(stats["heavy_atoms"]),
        float(stats["ring_count"]),
        float(constraints.get("rings", 0)),
        float(constraints.get("fused_ring_systems", 0)),
        float(constraints.get("spiro_centers", 0)),
        float(len(ring_sizes)),
        float(sum(ring_sizes) if ring_sizes else 0),
        float(constraints.get("halogens_bonded_to_ring_atoms", False)),
        *path_values[:3],
    ]


def build_constrained_queries() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl_gz(DATA_ROOT / "questions" / "constrained_generation_tasks.jsonl.gz")
    by_subcategory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subcategory[str(row.get("subcategory") or row.get("sub_category"))].append(dict(row))

    selected_rows: list[dict[str, Any]] = []
    for subcategory in sorted(by_subcategory):
        candidates = sorted(by_subcategory[subcategory], key=lambda row: row["uuid"])
        picks = farthest_point_sample(candidates, n=CONSTRAINED_PER_SUBCATEGORY, feature_fn=constrained_feature)
        if len(picks) != CONSTRAINED_PER_SUBCATEGORY:
            raise RuntimeError(f"expected {CONSTRAINED_PER_SUBCATEGORY} rows for {subcategory}")
        selected_rows.extend(picks)

    if len(selected_rows) != CONSTRAINED_PAIRS:
        raise RuntimeError(f"expected {CONSTRAINED_PAIRS} constrained-generation pairs")

    queries: list[dict[str, Any]] = []
    for idx, row in enumerate(selected_rows, start=1):
        subcategory = str(row.get("subcategory") or row.get("sub_category"))
        shared_meta = {
            "source_dataset": "constrained_generation_tasks",
            "source_uuid": row["uuid"],
            "subcategory": subcategory,
            "witness": row["witness"],
            "witness_heavy_atoms": rdkit_stats(str(row["witness"]))["heavy_atoms"],
        }
        base_id = f"constrained_{idx:03d}"
        queries.append(
            {
                "id": f"{base_id}_moljson",
                "pair_id": base_id,
                "family": "constrained_generation",
                "representation": "moljson",
                "evaluation_kind": "constrained_generation",
                "prompt": row["prompt"],
                "response_contract": response_contract_for("moljson", "constrained_generation"),
                "constraints": row["constraints"],
                "metadata": shared_meta,
            }
        )
        queries.append(
            {
                "id": f"{base_id}_commonchem",
                "pair_id": base_id,
                "family": "constrained_generation",
                "representation": "commonchem",
                "evaluation_kind": "constrained_generation",
                "prompt": row["prompt"],
                "response_contract": response_contract_for("commonchem", "constrained_generation"),
                "constraints": row["constraints"],
                "metadata": shared_meta,
            }
        )

    summary = {
        "constrained_pairs": len(selected_rows),
        "constrained_per_subcategory": CONSTRAINED_PER_SUBCATEGORY,
    }
    return queries, summary


def write_query_specs(queries: list[dict[str, Any]]) -> None:
    selected_ids: list[str] = []
    with MANIFEST_PATH.open("w", encoding="utf-8") as manifest:
        for query in queries:
            spec_path = QUERY_SPECS_DIR / f"{query['id']}.json"
            response_path = RESPONSES_DIR / f"{query['id']}.txt"
            spec = dict(query)
            spec["spec_path"] = str(spec_path)
            spec["response_path"] = str(response_path)
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            manifest.write(json.dumps(spec, ensure_ascii=True) + "\n")
            selected_ids.append(query["id"])
    SELECTED_IDS_PATH.write_text("\n".join(selected_ids) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    rng = random.Random(SEED)

    translation_queries, translation_summary = build_translation_queries(rng)
    shortest_path_queries, shortest_path_summary = build_shortest_path_queries(rng)
    constrained_queries, constrained_summary = build_constrained_queries()

    queries = translation_queries + shortest_path_queries + constrained_queries
    if len(queries) != 500:
        raise RuntimeError(f"expected 500 queries, got {len(queries)}")
    write_query_specs(queries)

    summary = {
        "seed": SEED,
        "query_count": len(queries),
        "pair_count": len({query["pair_id"] for query in queries}),
        **translation_summary,
        **shortest_path_summary,
        **constrained_summary,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote manifest to {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
