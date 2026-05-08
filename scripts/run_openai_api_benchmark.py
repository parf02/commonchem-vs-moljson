from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parents[1]
MOLJSON_SRC = WORKSPACE_ROOT / "MolJSON" / "src"
if str(MOLJSON_SRC) not in sys.path:
    sys.path.insert(0, str(MOLJSON_SRC))

from moljson import GetSchema  # noqa: E402


DEFAULT_MANIFEST = ROOT / "artifacts_api500" / "queries.jsonl"
DEFAULT_USAGE_LOG = ROOT / "artifacts_api500" / "usage.jsonl"
DEFAULT_RUN_SUMMARY = ROOT / "artifacts_api500" / "run_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CommonChem vs MolJSON API benchmark subset.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--usage-log", default=str(DEFAULT_USAGE_LOG))
    parser.add_argument("--run-summary", default=str(DEFAULT_RUN_SUMMARY))
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-missing", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wrapper_schema(key: str, description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            key: {
                "type": "string",
                "description": description,
            }
        },
        "required": [key],
    }


def commonchem_schema() -> dict[str, Any]:
    atom_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "z": {"type": "integer", "minimum": 1, "maximum": 118},
            "impHs": {"type": "integer", "minimum": 1, "maximum": 8},
            "chg": {"type": "integer", "minimum": -5, "maximum": 5},
            "isotope": {"type": "integer", "minimum": 1},
            "nRad": {"type": "integer", "minimum": 1, "maximum": 4},
        },
        "required": ["z"],
    }
    bond_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "type": {"type": "integer", "enum": [0, 1, 2, 3]},
            "atoms": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
            },
        },
        "required": ["type", "atoms"],
    }
    molecule_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "atoms": {"type": "array", "items": atom_schema},
            "bonds": {"type": "array", "items": bond_schema},
        },
        "required": ["atoms", "bonds"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "commonchem": {"type": "integer", "enum": [10]},
            "molecules": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": molecule_schema,
            },
        },
        "required": ["commonchem", "molecules"],
    }


def schema_for_query(query: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    evaluation_kind = str(query["evaluation_kind"])
    representation = str(query["representation"])

    if evaluation_kind in {"translation_to_graph", "constrained_generation"}:
        if representation == "moljson":
            return ("moljson_graph", GetSchema())
        if representation == "commonchem":
            return ("commonchem_graph", commonchem_schema())
        raise ValueError(f"unsupported graph representation: {representation}")

    if evaluation_kind == "translation_to_smiles":
        return (
            "smiles_answer",
            wrapper_schema(
                "smiles",
                "Molecule written as smiles ONLY. Do not ask clarifying questions. Do not write any comments.",
            ),
        )

    if evaluation_kind == "shortest_path":
        return (
            "integer_answer",
            wrapper_schema(
                "integer",
                "Shortest-path length as an integer encoded as a string ONLY. Do not ask clarifying questions. Do not write any comments.",
            ),
        )

    raise ValueError(f"unsupported evaluation kind: {evaluation_kind}")


def response_exists(query: dict[str, Any]) -> bool:
    path = Path(query["response_path"])
    return path.exists() and bool(path.read_text(encoding="utf-8").strip())


def extract_output_text(resp: Any) -> str:
    output_text = (getattr(resp, "output_text", None) or "").strip()
    if output_text:
        return output_text

    if hasattr(resp, "model_dump"):
        payload = resp.model_dump(mode="json")
    elif isinstance(resp, dict):
        payload = resp
    else:
        payload = {}

    output_items = payload.get("output") or []
    for item in output_items:
        for content in item.get("content") or []:
            text = str(content.get("text") or "").strip()
            if text:
                return text
    return ""


def extract_usage(resp: Any) -> dict[str, Any]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if isinstance(usage, dict):
        return usage
    return {}


def extract_response_id(resp: Any) -> str | None:
    response_id = getattr(resp, "id", None)
    if isinstance(response_id, str) and response_id:
        return response_id
    return None


def record_for_query(
    query: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    response_id: str | None,
    usage: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    return {
        "id": query["id"],
        "pair_id": query["pair_id"],
        "family": query["family"],
        "representation": query["representation"],
        "evaluation_kind": query["evaluation_kind"],
        "response_path": query["response_path"],
        "model": model,
        "reasoning_effort": reasoning_effort,
        "response_id": response_id,
        "usage": usage,
        "error": error,
    }


def is_retryable(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    transient_markers = ("ratelimit", "timeout", "apierror", "serviceunavailable", "apiconnection")
    if any(marker in name for marker in transient_markers):
        return True
    return any(marker in msg for marker in ("429", "rate limit", "timeout", "temporarily", "try again", "503", "502", "connection"))


def backoff_seconds(attempt: int) -> float:
    return min(12.0, 0.75 * (2 ** max(0, attempt - 1)))


async def append_jsonl(path: Path, row: dict[str, Any], lock: asyncio.Lock) -> None:
    line = json.dumps(row, ensure_ascii=True) + "\n"
    async with lock:
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(_append_text, path, line)


def _append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


async def write_response(path: Path, text: str) -> None:
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_text, text, "utf-8")


async def run_one(
    client: Any,
    query: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    timeout_s: int,
    max_retries: int,
    usage_log: Path,
    usage_lock: asyncio.Lock,
) -> dict[str, Any]:
    schema_name, schema = schema_for_query(query)
    prompt = str(query["prompt"]).strip()
    response_path = Path(query["response_path"])

    last_error: str | None = None
    for attempt in range(1, max_retries + 2):
        try:
            resp = await asyncio.wait_for(
                client.responses.create(
                    model=model,
                    reasoning={"effort": reasoning_effort},
                    input=prompt,
                    text={
                        "verbosity": "low",
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        },
                    },
                    store=False,
                ),
                timeout=timeout_s,
            )
            output_text = extract_output_text(resp)
            if not output_text:
                raise ValueError("empty output_text")
            await write_response(response_path, output_text)
            record = record_for_query(
                query,
                model=model,
                reasoning_effort=reasoning_effort,
                response_id=extract_response_id(resp),
                usage=extract_usage(resp),
                error=None,
            )
            await append_jsonl(usage_log, record, usage_lock)
            return record
        except Exception as exc:
            last_error = str(exc).strip() or exc.__class__.__name__
            if attempt <= max_retries and is_retryable(exc):
                await asyncio.sleep(backoff_seconds(attempt))
                continue

    record = record_for_query(
        query,
        model=model,
        reasoning_effort=reasoning_effort,
        response_id=None,
        usage={},
        error=last_error,
    )
    await append_jsonl(usage_log, record, usage_lock)
    return record


async def run_all(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    from openai import AsyncOpenAI

    manifest_path = Path(args.manifest)
    usage_log = Path(args.usage_log)
    run_summary = Path(args.run_summary)
    queries = load_manifest(manifest_path)
    if args.limit > 0:
        queries = queries[: args.limit]

    if args.only_missing:
        queries = [query for query in queries if not response_exists(query)]

    if not queries:
        print("No queries to run.")
        return 0

    client = AsyncOpenAI(api_key=api_key)
    usage_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, int(args.concurrency)))

    async def guarded(query: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await run_one(
                client,
                query,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout_s=int(args.timeout_s),
                max_retries=int(args.max_retries),
                usage_log=usage_log,
                usage_lock=usage_lock,
            )

    results = await asyncio.gather(*(guarded(query) for query in queries))
    success_count = sum(1 for row in results if not row["error"])
    error_count = len(results) - success_count
    summary = {
        "manifest": str(manifest_path),
        "query_count": len(results),
        "success_count": success_count,
        "error_count": error_count,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "usage_log": str(usage_log),
    }
    run_summary.parent.mkdir(parents=True, exist_ok=True)
    run_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if error_count == 0 else 1


def main() -> int:
    args = parse_args()
    return asyncio.run(run_all(args))


if __name__ == "__main__":
    raise SystemExit(main())
