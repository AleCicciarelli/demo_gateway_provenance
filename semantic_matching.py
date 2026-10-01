"""Standalone CSV -> model decisions -> ID mapping prototype.

No gateway import, SQL rewriting, or implicit selection of bucket files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Callable

from row_probability import PROBABILITY_COLUMN

from semantic_match_prompt import DEFAULT_MATCH_DEFINITION, PROMPT_VERSION, build_match_prompt


MATCH_PROBABILITIES = {"match": 0.9, "uncertain": 0.3}

def read_source(path: Path, key: str, fields: list[str]) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    headers = reader.fieldnames or []
    if len(headers) != len(set(headers)):
        raise ValueError(f"Duplicate CSV columns: {path}")
    if not fields or key in fields or not set([key, *fields]) <= set(headers):
        raise ValueError(f"Select an existing key and non-key descriptive fields in {path}")
    rows, seen = [], set()
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row in {path}")
        identity = row[key]
        if not identity.strip() or identity in seen:
            raise ValueError(f"Key {key} must be nonempty and unique in {path}")
        seen.add(identity)
        rows.append({"id": identity, "attributes": {field: row[field] for field in fields}})
    return rows, hashlib.sha256(raw).hexdigest()


def parse_decisions(
    response: str, expected: set[str], *, allow_missing: bool = False,
    defer_duplicates: bool = False,
) -> list[dict]:
    """Validate decisions; optionally omit repeated IDs so they can be retried."""
    data = json.loads(response)
    if not isinstance(data, dict) or not isinstance(data.get("decisions"), list):
        raise ValueError("Model must return a decisions array")
    seen, duplicates = set(), set()
    for item in data["decisions"]:
        if not isinstance(item, dict):
            raise ValueError("Each decision must be an object")
        pair_id = item.get("pair_id")
        if not isinstance(pair_id, str) or pair_id not in expected:
            raise ValueError(f"Unknown pair_id {pair_id!r}; expected {', '.join(sorted(expected))}")
        if pair_id in seen:
            if not defer_duplicates:
                raise ValueError(f"Duplicate pair_id {pair_id!r}")
            duplicates.add(pair_id)
        if item.get("decision") not in ("match", "no_match", "uncertain"):
            raise ValueError("Invalid decision")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("Each decision needs a nonempty reason")
        seen.add(pair_id)
    seen -= duplicates
    if seen != expected and not allow_missing:
        raise ValueError("Model omitted candidate pairs: " + ", ".join(sorted(expected - seen)))
    return [item for item in data["decisions"] if item["pair_id"] not in duplicates]


def model_from_env(provider: str, model: str) -> Callable[[str], str]:
    """Use the gateway's environment conventions without loading its services."""
    if provider not in ("ollama", "openai", "openai-compatible"):
        raise ValueError("Unsupported provider")

    def call(prompt: str) -> str:
        import requests

        if provider == "ollama":
            response = requests.post(
                os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "format": "json",
                      "options": {"temperature": 0, "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192"))}},
                timeout=(5, float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "700"))),
            )
            response.raise_for_status()
            return response.json()["response"]
        base = os.getenv("LLM_API_BASE", "").rstrip("/")
        if not base:
            raise ValueError("LLM_API_BASE is required for openai-compatible calls")
        headers = {}
        if os.getenv("LLM_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["LLM_API_KEY"]
        response = requests.post(
            base + "/chat/completions", headers=headers,
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "stream": False},
            timeout=(float(os.getenv("LLM_CONNECT_TIMEOUT", "5")),
                     float(os.getenv("LLM_READ_TIMEOUT", "120"))),
            verify=os.getenv("LLM_SSL_VERIFY", "true").lower() not in ("0", "false", "no"),
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    return call


def build_mapping(
    left: Path, right: Path, left_key: str, right_key: str,
    left_fields: list[str], right_fields: list[str], match_definition: str | None,
    output: Path, model_call: Callable[[str], str] | None,
    *, batch_size: int = 5, max_pairs: int = 10000,
    max_prompt_chars: int = 24000, cardinality: str = "one_to_one",
    model_info: dict | None = None, progress: Callable[[dict], None] | None = None,
) -> dict:
    """Compare all pairs. None model_call writes prompts only, for dry runs.

    A fresh run directory holds prompts, responses, audit, and matches.csv.
    On failure, retain diagnostics but never publish a partial mapping.
    """
    if batch_size < 1 or max_pairs < 1 or max_prompt_chars < 1:
        raise ValueError("Positive limits are required")
    match_definition = (match_definition or "").strip() or DEFAULT_MATCH_DEFINITION
    if cardinality not in ("one_to_one", "many_to_many"):
        raise ValueError("Unsupported cardinality")
    a, a_hash = read_source(left, left_key, left_fields)
    b, b_hash = read_source(right, right_key, right_fields)
    if len(a) * len(b) > max_pairs:
        raise ValueError("Candidate count exceeds max_pairs; narrow inputs or explicitly raise the limit")

    pairs, identities = [], {}
    for ra in a:
        for rb in b:
            pid = f"p{len(pairs)}"
            pairs.append({"pair_id": pid, "left": ra["attributes"], "right": rb["attributes"]})
            identities[pid] = {"idA": ra["id"], "idB": rb["id"]}
    # Bound prompt size as well as pair count; never silently truncate rows.
    batches, current = [], []
    for pair in pairs:
        proposal = current + [pair]
        if len(proposal) > batch_size or len(build_match_prompt(match_definition, proposal)) > max_prompt_chars:
            if current:
                batches.append(current)
            current = [pair]
        else:
            current = proposal
        if len(build_match_prompt(match_definition, current)) > max_prompt_chars:
            raise ValueError("One candidate exceeds max_prompt_chars; select fewer fields or raise the limit")
    if current:
        batches.append(current)

    output.mkdir(parents=True, exist_ok=False)
    audit = {"status": "running", "prompt_version": PROMPT_VERSION,
             "left": {"path": str(left.resolve()), "key": left_key, "fields": left_fields, "sha256": a_hash},
             "right": {"path": str(right.resolve()), "key": right_key, "fields": right_fields, "sha256": b_hash},
             "match_definition": match_definition, "cardinality": cardinality,
             "model": model_info or {}, "candidate_count": len(pairs), "batch_count": len(batches),
             "pair_ids": identities, "decisions": []}
    total_requests = len(batches)
    completed_requests = 0

    def report_progress():
        if progress:
            progress({"total_requests": total_requests, "completed_requests": completed_requests,
                      "planned_requests": len(batches), "candidate_count": len(pairs)})

    try:
        report_progress()
        for index, batch in enumerate(batches):
            prompt = build_match_prompt(match_definition, batch)
            (output / f"prompt_{index:04d}.txt").write_text(prompt, encoding="utf-8")
            if model_call is None:
                continue
            response = model_call(prompt)
            (output / f"response_{index:04d}.txt").write_text(response, encoding="utf-8")
            decisions = parse_decisions(response, {p["pair_id"] for p in batch},
                                        allow_missing=True, defer_duplicates=True)
            audit["decisions"].extend(decisions)
            returned = {d["pair_id"] for d in decisions}
            total_requests += sum(pair["pair_id"] not in returned for pair in batch)
            completed_requests += 1
            report_progress()
            # Retry omitted or repeated IDs individually, once each. Never pick
            # the first/last conflicting decision or publish an incomplete mapping.
            for pair in batch:
                if pair["pair_id"] in returned:
                    continue
                audit.setdefault("retries", []).append({
                    "batch_index": index, "pair_id": pair["pair_id"],
                    "reason": "missing_or_duplicate_decision",
                })
                retry_prompt = build_match_prompt(match_definition, [pair])
                suffix = f"{index:04d}_retry_{pair['pair_id']}"
                (output / f"prompt_{suffix}.txt").write_text(retry_prompt, encoding="utf-8")
                retry_response = model_call(retry_prompt)
                (output / f"response_{suffix}.txt").write_text(retry_response, encoding="utf-8")
                completed_requests += 1
                report_progress()
                try:
                    retry_decisions = parse_decisions(retry_response, {pair["pair_id"]})
                except ValueError as exc:
                    raise ValueError(f"Matching pair {pair['pair_id']} failed after individual retry: {exc}") from exc
                audit["decisions"].extend(retry_decisions)
        if model_call is None:
            audit["status"] = "dry_run"
        else:
            matches = [d for d in audit["decisions"] if d["decision"] == "match"]
            left_counts = Counter(identities[d["pair_id"]]["idA"] for d in matches)
            right_counts = Counter(identities[d["pair_id"]]["idB"] for d in matches)
            accepted = []
            accepted_pairs = []
            pairs_by_id = {pair["pair_id"]: pair for pair in pairs}
            for decision in audit["decisions"]:
                ids = identities[decision["pair_id"]]
                conflict = (cardinality == "one_to_one" and decision["decision"] == "match"
                            and (left_counts[ids["idA"]] > 1 or right_counts[ids["idB"]] > 1))
                decision["disposition"] = "cardinality_conflict" if conflict else decision["decision"]
                if decision["decision"] in MATCH_PROBABILITIES and not conflict:
                    probability = MATCH_PROBABILITIES[decision["decision"]]
                    accepted.append({**ids, PROBABILITY_COLUMN: probability})
                    pair = pairs_by_id[decision["pair_id"]]
                    accepted_pairs.append({
                        "pair_id": decision["pair_id"], "reason": decision["reason"],
                        "decision": decision["decision"], "probability": probability,
                        "left": {"id": ids["idA"], "attributes": pair["left"]},
                        "right": {"id": ids["idB"], "attributes": pair["right"]},
                    })
            with (output / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["idA", "idB", PROBABILITY_COLUMN])
                writer.writeheader()
                writer.writerows(accepted)
            audit.update(status="complete", accepted_count=len(accepted), accepted_pairs=accepted_pairs)
    except Exception as exc:
        audit.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", type=Path, default=Path("bucket"))
    for side in ("left", "right"):
        parser.add_argument(f"--{side}", required=True, help="CSV filename relative to bucket")
        parser.add_argument(f"--{side}-key", required=True)
        parser.add_argument(f"--{side}-fields", nargs="+", required=True)
    parser.add_argument("--match-definition", help="Optional override of generic identity or factual relationship matching")
    parser.add_argument("--output-dir", type=Path, required=True, help="New run directory; must not exist")
    parser.add_argument("--provider", choices=["ollama", "openai", "openai-compatible"],
                        default=os.getenv("PLANNER_LLM_PROVIDER") or
                        ("openai-compatible" if os.getenv("LLM_API_MODEL") else "ollama"))
    parser.add_argument("--model", default=os.getenv("LLM_API_MODEL") or os.getenv("PLANNER_LLM_MODEL") or "llama3:8b",
                        help="Model override (default: LLM_API_MODEL, then PLANNER_LLM_MODEL, then llama3:8b)")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-pairs", type=int, default=10000)
    parser.add_argument("--max-prompt-chars", type=int, default=24000)
    parser.add_argument("--cardinality", choices=["one_to_one", "many_to_many"], default="one_to_one")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sources = []
    for name in (args.left, args.right):
        path = (args.bucket / name).resolve()
        if not path.is_relative_to(args.bucket.resolve()):
            parser.error("Input files must be inside bucket")
        sources.append(path)
    audit = build_mapping(
        *sources, args.left_key, args.right_key, args.left_fields, args.right_fields,
        args.match_definition, args.output_dir,
        None if args.dry_run else model_from_env(args.provider, args.model),
        batch_size=args.batch_size, max_pairs=args.max_pairs, max_prompt_chars=args.max_prompt_chars,
        cardinality=args.cardinality, model_info={"provider": args.provider, "model": args.model},
    )
    print(json.dumps({"status": audit["status"], "candidates": audit["candidate_count"],
                      "accepted": audit.get("accepted_count"), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
