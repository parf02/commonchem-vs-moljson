from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

try:
    from moljson import MolFromJSON, MolToJSON  # type: ignore  # noqa: E402
except ImportError:
    EXPERIMENT_ROOT = Path(__file__).resolve().parent
    WORKSPACE_ROOT = EXPERIMENT_ROOT.parents[1]
    MOLJSON_SRC = WORKSPACE_ROOT / "MolJSON" / "src"
    if str(MOLJSON_SRC) not in sys.path:
        sys.path.insert(0, str(MOLJSON_SRC))
    from moljson import MolFromJSON, MolToJSON  # type: ignore  # noqa: E402


VALID_HALOGENS = ("F", "Cl", "Br")
LOWER_TO_HALOGEN = {h.lower(): h for h in VALID_HALOGENS}
JSON_DECODER = json.JSONDecoder()


def safe_json_loads(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty text")

    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = min([i for i in (raw.find("{"), raw.find("[")) if i != -1], default=-1)
    if start == -1:
        raise ValueError("no json object found")

    obj, _ = JSON_DECODER.raw_decode(raw[start:])
    return obj


def canonical_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def mol_to_commonchem(mol: Chem.Mol) -> dict[str, Any]:
    kek = Chem.Mol(mol)
    Chem.Kekulize(kek, clearAromaticFlags=True)

    atoms: list[dict[str, Any]] = []
    for atom in kek.GetAtoms():
        atom_obj: dict[str, Any] = {"z": int(atom.GetAtomicNum())}
        total_h = int(atom.GetTotalNumHs())
        if total_h:
            atom_obj["impHs"] = total_h
        formal_charge = int(atom.GetFormalCharge())
        if formal_charge:
            atom_obj["chg"] = formal_charge
        isotope = int(atom.GetIsotope())
        if isotope:
            atom_obj["isotope"] = isotope
        radicals = int(atom.GetNumRadicalElectrons())
        if radicals:
            atom_obj["nRad"] = radicals
        atoms.append(atom_obj)

    bonds: list[dict[str, Any]] = []
    for bond in kek.GetBonds():
        order = int(round(bond.GetBondTypeAsDouble()))
        bonds.append(
            {
                "type": order,
                "atoms": [int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())],
            }
        )

    return {"commonchem": 10, "molecules": [{"atoms": atoms, "bonds": bonds}]}


def commonchem_to_mol(obj: dict[str, Any]) -> Chem.Mol:
    def int_or_default(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        return int(value)

    commonchem_version = obj.get("commonchem")
    if commonchem_version not in (10, "10", {"version": 10}, {"version": "10"}):
        raise ValueError("unsupported CommonChem version")

    molecules = obj.get("molecules")
    if not isinstance(molecules, list) or len(molecules) != 1:
        raise ValueError("expected one molecule")

    mol_obj = molecules[0]
    if not isinstance(mol_obj, dict):
        raise ValueError("molecule must be an object")

    atoms_obj = mol_obj.get("atoms")
    if not isinstance(atoms_obj, list):
        raise ValueError("molecule atoms missing")

    rw = Chem.RWMol()
    for atom_obj in atoms_obj:
        if not isinstance(atom_obj, dict):
            raise ValueError("atom must be an object")
        atomic_num = int(atom_obj["z"])
        atom = Chem.Atom(atomic_num)
        atom.SetFormalCharge(int_or_default(atom_obj.get("chg", 0)))
        isotope = int_or_default(atom_obj.get("isotope", 0))
        if isotope:
            atom.SetIsotope(isotope)
        radicals = int_or_default(atom_obj.get("nRad", 0))
        if radicals:
            atom.SetNumRadicalElectrons(radicals)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(int_or_default(atom_obj.get("impHs", 0)))
        rw.AddAtom(atom)

    bond_map = {
        0: Chem.BondType.ZERO,
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    for bond_obj in mol_obj.get("bonds", []) or []:
        if not isinstance(bond_obj, dict):
            raise ValueError("bond must be an object")
        bond_type = int(bond_obj["type"])
        atom_indices = bond_obj["atoms"]
        if not isinstance(atom_indices, list) or len(atom_indices) != 2:
            raise ValueError("bond atoms must be a length-2 array")
        rw.AddBond(int(atom_indices[0]), int(atom_indices[1]), bond_map[bond_type])

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def prompt_object_text(obj: Any) -> str:
    return str(obj)


def expand_ring_size_spec(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        expanded: list[int] = []
        for size, count in raw.items():
            expanded.extend([int(size)] * int(count))
        return sorted(expanded)
    if isinstance(raw, list):
        return sorted(int(x) for x in raw)
    raise ValueError(f"unsupported ring_sizes spec type: {type(raw).__name__}")


def count_fused_ring_systems(mol: Chem.Mol) -> int:
    bond_rings = [set(r) for r in mol.GetRingInfo().BondRings()]
    if not bond_rings:
        return 0

    adj = [set() for _ in range(len(bond_rings))]
    for i in range(len(bond_rings)):
        for j in range(i + 1, len(bond_rings)):
            if bond_rings[i].intersection(bond_rings[j]):
                adj[i].add(j)
                adj[j].add(i)

    participating = {i for i, neigh in enumerate(adj) if neigh}
    if not participating:
        return 0

    visited: set[int] = set()
    systems = 0
    for start in sorted(participating):
        if start in visited:
            continue
        systems += 1
        stack = [start]
        visited.add(start)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
    return systems


def classify_ring_topology(mol: Chem.Mol) -> str | None:
    atom_rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    bond_rings = [set(r) for r in mol.GetRingInfo().BondRings()]
    if int(rdMolDescriptors.CalcNumSpiroAtoms(mol)) != 0:
        return None

    if len(atom_rings) == 2 and len(bond_rings) == 2:
        shared_bonds = bond_rings[0].intersection(bond_rings[1])
        shared_atoms = atom_rings[0].intersection(atom_rings[1])
        if shared_bonds:
            return "two_fused_rings_shared_edge"
        if shared_atoms:
            return None

        inter_ring_bonds = 0
        ring0 = atom_rings[0]
        ring1 = atom_rings[1]
        for bond in mol.GetBonds():
            a = int(bond.GetBeginAtomIdx())
            b = int(bond.GetEndAtomIdx())
            if (a in ring0 and b in ring1) or (a in ring1 and b in ring0):
                inter_ring_bonds += 1
        if inter_ring_bonds == 0:
            return "two_separate_rings"
        if inter_ring_bonds == 1:
            return "two_separate_rings_single_bond"
    return None


def shortest_halogen_distance(mol: Chem.Mol, h1: str, h2: str) -> int | None:
    idx1 = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == h1]
    idx2 = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == h2]
    if len(idx1) != 1 or len(idx2) != 1:
        return None
    path = Chem.rdmolops.GetShortestPath(mol, idx1[0], idx2[0])
    if not path:
        return None
    return len(path) - 1


def project_constraints(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "connected_components",
        "rings",
        "fused_ring_systems",
        "ring_topology",
        "spiro_centers",
        "halogens_bonded_to_ring_atoms",
        "ring_sizes",
        "halogen_counts",
        "shortest_halogen_paths",
    ):
        if key in raw:
            out[key] = raw[key]
    return out


def check_constrained_molecule(mol: Chem.Mol, constraints: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []

    if "connected_components" in constraints:
        got = len(Chem.GetMolFrags(mol))
        if got != int(constraints["connected_components"]):
            failures.append(f"connected_components={got}")

    if "rings" in constraints:
        got = int(mol.GetRingInfo().NumRings())
        if got != int(constraints["rings"]):
            failures.append(f"rings={got}")

    if "fused_ring_systems" in constraints:
        got = count_fused_ring_systems(mol)
        if got != int(constraints["fused_ring_systems"]):
            failures.append(f"fused_ring_systems={got}")

    if "ring_topology" in constraints:
        got = classify_ring_topology(mol)
        if got != constraints["ring_topology"]:
            failures.append(f"ring_topology={got}")

    if "spiro_centers" in constraints:
        got = int(rdMolDescriptors.CalcNumSpiroAtoms(mol))
        if got != int(constraints["spiro_centers"]):
            failures.append(f"spiro_centers={got}")

    if "halogens_bonded_to_ring_atoms" in constraints:
        required = bool(constraints["halogens_bonded_to_ring_atoms"])
        got = False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in VALID_HALOGENS:
                continue
            if any(nei.IsInRing() for nei in atom.GetNeighbors()):
                got = True
                break
        if got != required:
            failures.append(f"halogens_bonded_to_ring_atoms={got}")

    if "ring_sizes" in constraints:
        ring_sizes = sorted(len(r) for r in mol.GetRingInfo().AtomRings())
        expected = expand_ring_size_spec(constraints["ring_sizes"])
        if ring_sizes != expected:
            failures.append(f"ring_sizes={ring_sizes}")

    if "halogen_counts" in constraints:
        expected_counts = {str(k): int(v) for k, v in constraints["halogen_counts"].items()}
        got_counts = {h: 0 for h in VALID_HALOGENS}
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym in got_counts:
                got_counts[sym] += 1
        for halogen, expected in expected_counts.items():
            if got_counts.get(halogen, 0) != expected:
                failures.append(f"{halogen}_count={got_counts.get(halogen, 0)}")

    for spec in constraints.get("shortest_halogen_paths", []) or []:
        between = spec.get("between", [])
        if not isinstance(between, list) or len(between) != 2:
            failures.append("bad_shortest_halogen_paths_spec")
            continue
        h1 = LOWER_TO_HALOGEN.get(str(between[0]).lower())
        h2 = LOWER_TO_HALOGEN.get(str(between[1]).lower())
        if not h1 or not h2:
            failures.append("unknown_halogen_in_spec")
            continue
        got = shortest_halogen_distance(mol, h1, h2)
        expected = int(spec["distance_bonds"])
        if got != expected:
            failures.append(f"{h1}_{h2}_distance={got}")

    return (not failures), failures


def parse_molecule_from_response(response_text: str, representation: str) -> Chem.Mol:
    payload = safe_json_loads(response_text)
    if representation == "moljson":
        if not isinstance(payload, dict):
            raise ValueError("MolJSON payload must be an object")
        mol = MolFromJSON(payload)
        if mol is None:
            raise ValueError("MolFromJSON returned None")
        return mol
    if representation == "commonchem":
        if not isinstance(payload, dict):
            raise ValueError("CommonChem payload must be an object")
        return commonchem_to_mol(payload)
    raise ValueError(f"unsupported representation: {representation}")


def parse_wrapped_string(response_text: str, key: str) -> str:
    payload = safe_json_loads(response_text)
    if not isinstance(payload, dict):
        raise ValueError("wrapped payload must be an object")
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"missing string key: {key}")
    return value


def parse_wrapped_integer(response_text: str) -> int:
    payload = safe_json_loads(response_text)
    if not isinstance(payload, dict):
        raise ValueError("integer payload must be an object")
    value = payload.get("integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value.strip())
    raise ValueError("missing integer field")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (
        z
        * math.sqrt((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total))
        / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 20260505,
) -> tuple[float, float]:
    if len(a) != len(b):
        raise ValueError("paired arrays must have the same length")
    if len(a) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        diffs[i] = float(np.mean(a[sample_idx] - b[sample_idx]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return (float(lo), float(hi))


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if len(a) != len(b):
        raise ValueError("paired arrays must have the same length")
    b01 = int(np.sum((a == 0) & (b == 1)))
    b10 = int(np.sum((a == 1) & (b == 0)))
    discordant = b01 + b10
    if discordant == 0:
        return {"b01": b01, "b10": b10, "p_value": 1.0}

    smaller = min(b01, b10)
    cdf = 0.0
    for k in range(0, smaller + 1):
        cdf += math.comb(discordant, k) * (0.5 ** discordant)
    p_value = min(1.0, 2.0 * cdf)
    return {"b01": b01, "b10": b10, "p_value": p_value}
