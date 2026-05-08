from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = ROOT.parent if (ROOT.parent / "data").exists() and (ROOT.parent / "outputs").exists() else None
PROJECT_ROOT = RELEASE_ROOT or ROOT

MOLJSON_SRC = PROJECT_ROOT / "MolJSON" / "src"
if MOLJSON_SRC.exists() and str(MOLJSON_SRC) not in sys.path:
    sys.path.insert(0, str(MOLJSON_SRC))

from moljson import GetSchema  # noqa: E402


COMMONCHEM_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "commonchem_strict_schema.json"
if RELEASE_ROOT is None:
    DEFAULT_MANIFEST = ROOT / "artifacts_api500" / "queries.jsonl"
    DEFAULT_LOG_CSV = ROOT / "artifacts_api500" / "api_run_log.csv"
    DEFAULT_SUMMARY_JSON = ROOT / "artifacts_api500" / "api_run_summary.json"
else:
    DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "queries.jsonl"
    DEFAULT_LOG_CSV = PROJECT_ROOT / "outputs" / "api_run_log.csv"
    DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "outputs" / "api_run_summary.json"

CSV_FIELDS = [
    "id",
    "pair_id",
    "family",
    "representation",
    "evaluation_kind",
    "model",
    "effort",
    "response_id",
    "status",
    "error",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "duration_s",
    "response_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CommonChem vs MolJSON API benchmark subset.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--log-csv", "--usage-log", dest="log_csv", default=str(DEFAULT_LOG_CSV))
    parser.add_argument("--summary-json", "--run-summary", dest="summary_json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--effort", "--reasoning-effort", dest="reasoning_effort", default="low")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", "--only-missing", dest="only_missing", action="store_true")
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


def fallback_commonchem_schema() -> dict[str, Any]:
    atom_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "z": {"type": "integer", "minimum": 0, "maximum": 118},
            "impHs": {"type": "integer", "minimum": 0, "maximum": 8},
            "chg": {"type": "integer", "minimum": -5, "maximum": 5},
            "isotope": {"type": "integer", "minimum": 0, "maximum": 400},
            "nRad": {"type": "integer", "minimum": 0, "maximum": 4},
        },
        "required": ["z", "impHs", "chg", "isotope", "nRad"],
    }
    bond_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "type": {"type": "integer", "enum": [1, 2, 3]},
            "atoms": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
            },
        },
        "required": ["type", "atoms"],
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
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "atoms": {"type": "array", "items": atom_schema},
                        "bonds": {"type": "array", "items": bond_schema},
                    },
                    "required": ["atoms", "bonds"],
                },
            },
        },
        "required": ["commonchem", "molecules"],
    }


def commonchem_schema() -> dict[str, Any]:
    if COMMONCHEM_SCHEMA_PATH.exists():
        return json.loads(COMMONCHEM_SCHEMA_PATH.read_text(encoding="utf-8"))
    return fallback_commonchem_schema()


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

    for item in payload.get("output") or []:
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


def token_value(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key)
    return int(value) if isinstance(value, (int, float)) else 0


def record_for_query(
    query: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    response_id: str | None,
    usage: dict[str, Any],
    error: str | None,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "id": query["id"],
        "pair_id": query["pair_id"],
        "family": query["family"],
        "representation": query["representation"],
        "evaluation_kind": query["evaluation_kind"],
        "model": model,
        "effort": reasoning_effort,
        "response_id": response_id or "",
        "status": "ok" if not error else "error",
        "error": error or "",
        "input_tokens": token_value(usage, "input_tokens"),
        "output_tokens": token_value(usage, "output_tokens"),
        "total_tokens": token_value(usage, "total_tokens"),
        "duration_s": f"{duration_s:.3f}",
        "response_path": query["response_path"],
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


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


async def append_csv(path: Path, row: dict[str, Any], lock: asyncio.Lock) -> None:
    async with lock:
        await asyncio.to_thread(_append_csv_row, path, row)


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
    log_csv: Path,
    log_lock: asyncio.Lock,
) -> dict[str, Any]:
    schema_name, schema = schema_for_query(query)
    prompt = str(query["prompt"]).strip()
    response_path = Path(query["response_path"])

    last_error: str | None = None
    for attempt in range(1, max_retries + 2):
        started = time.perf_counter()
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
                duration_s=time.perf_counter() - started,
            )
            await append_csv(log_csv, record, log_lock)
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
                duration_s=time.perf_counter() - started,
            )
            await append_csv(log_csv, record, log_lock)
            return record

    raise AssertionError("unreachable")


def summarize_results(
    results: list[dict[str, Any]],
    *,
    manifest_path: Path,
    log_csv: Path,
    model: str,
    reasoning_effort: str,
    only_missing: bool,
    elapsed_s: float,
) -> dict[str, Any]:
    completed_ok = sum(1 for row in results if row["status"] == "ok")
    completed_error = len(results) - completed_ok
    input_tokens = sum(int(row["input_tokens"]) for row in results)
    output_tokens = sum(int(row["output_tokens"]) for row in results)
    total_tokens = sum(int(row["total_tokens"]) for row in results)
    return {
        "manifest": str(manifest_path),
        "log_csv": str(log_csv),
        "model": model,
        "effort": reasoning_effort,
        "resume": bool(only_missing),
        "query_count": len(results),
        "completed_ok": completed_ok,
        "completed_error": completed_error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed_s, 3),
    }


async def run_all(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    log_csv = Path(args.log_csv)
    summary_json = Path(args.summary_json)

    queries = load_manifest(manifest_path)
    if args.limit > 0:
        queries = queries[: args.limit]
    if args.only_missing:
        queries = [query for query in queries if not response_exists(query)]
    if not queries:
        print("No queries to run.")
        return 0

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    log_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, int(args.concurrency)))
    started = time.perf_counter()

    async def guarded(query: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await run_one(
                client,
                query,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout_s=int(args.timeout_s),
                max_retries=int(args.max_retries),
                log_csv=log_csv,
                log_lock=log_lock,
            )

    results = await asyncio.gather(*(guarded(query) for query in queries))
    summary = summarize_results(
        results,
        manifest_path=manifest_path,
        log_csv=log_csv,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        only_missing=args.only_missing,
        elapsed_s=time.perf_counter() - started,
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["completed_error"] == 0 else 1


def main() -> int:
    args = parse_args()
    return asyncio.run(run_all(args))


if __name__ == "__main__":
    raise SystemExit(main())
