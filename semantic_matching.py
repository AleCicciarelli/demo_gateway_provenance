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
import re
from collections import Counter
from pathlib import Path
from typing import Callable

from row_probability import PROBABILITY_COLUMN

from semantic_match_prompt import (DEFAULT_MATCH_DEFINITION, PROMPT_VERSION, build_match_prompt,
                                   BLOCK_PROMPT_VERSION, build_block_match_prompt, with_matching_context)


MATCH_PROBABILITIES = {"match": 0.9}


def match_reason_conflict(reason, match_definition=None):
    """Conservatively flag explicit rejection or unresolved deliberation.

    This is an English text guard, not a factual or semantic proof of a match.
    Ambiguous explanations are withheld rather than promoted to matches.
    """
    if not isinstance(reason, str):
        return None
    # Nonidentity is a contradiction only for identity matching. Association
    # definitions must opt in explicitly; unknown definitions stay conservative.
    association = "relationship type: association" in (match_definition or "").lower()
    patterns = (
        r"\b(?:no[ -]match(?:es|ing)?|non[ -]match(?:es|ing)?|unmatch(?:ed)?)\b",
        r"\b(?:cannot|can't|does not|do not|did not|is not|are not|should not)\s+(?:be\s+)?match(?:ed|ing)?\b",
        r"\bno\s+(?:corresponding|matching)\b",
        r"\bnot\s+(?:the\s+)?(?:a match|supported|the driver)\b",
        r"\b(?:should|must)\s+be\s+omitted\b",
        r"\b(?:wait|let me|re-verify|reverify)\b",
        r"\b(?:uncertain|insufficient evidence|cannot determine)\b",
    )
    identity_reason = reason
    if association:
        identity_reason = re.sub(r"\bnot\s+(?:the\s+)?same\s+(?:entity|entities|entity type|record)\b",
                                 "", identity_reason, flags=re.IGNORECASE)
    if re.search(r"\bnot\s+(?:the\s+)?same\b", identity_reason, re.IGNORECASE):
        return "Match reason rejects entity identity or the specific candidate"
    if any(re.search(pattern, reason, re.IGNORECASE) for pattern in patterns):
        return "Match reason contains rejection, uncertainty, or unresolved deliberation"
    return None


def normalize_decision(item, match_definition=None):
    """Normalize labels and withhold matches with contradictory explanations."""
    value = item.get("decision")
    if value == "match":
        conflict = match_reason_conflict(item.get("reason"), match_definition)
        if conflict:
            return {**item, "decision": "uncertain", "raw_decision": value,
                    "validation_warning": conflict}
    if isinstance(value, str) and value in ("match", "unmatch", "uncertain"):
        return dict(item)
    return {**item, "decision": "uncertain", "raw_decision": value}


def resolve_matching_limits(context_tokens=None, max_output_tokens=None, *, default_output_tokens=None):
    """Adapt automatic output ceilings; validate explicit per-request allocations."""
    context = context_tokens if context_tokens is not None else int(
        os.getenv("MATCH_CONTEXT_TOKENS") or os.getenv("OLLAMA_NUM_CTX", "8192"))
    requested = max_output_tokens if max_output_tokens is not None else (
        default_output_tokens if default_output_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "8192")))
    if context < 1 or requested < 1:
        raise ValueError("Positive limits are required")
    effective = requested if max_output_tokens is not None else min(requested, max(1, context // 4))
    usable = int(context * .8)
    if effective >= usable:
        raise ValueError("LLM_MAX_TOKENS leaves no prompt budget within the context headroom; "
                         "lower LLM_MAX_TOKENS or increase the configured model context")
    return {"context_tokens": context, "max_output_tokens": effective,
            "configured_max_output_tokens": requested, "usable_tokens": usable,
            "automatic_output_budget": max_output_tokens is None}


def read_source(path: Path, key: str, fields: list[str], *, allow_empty_fields=False) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    headers = reader.fieldnames or []
    if len(headers) != len(set(headers)):
        raise ValueError(f"Duplicate CSV columns: {path}")
    if (not fields and not allow_empty_fields) or key in fields or not set([key, *fields]) <= set(headers):
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
    defer_duplicates: bool = False, match_definition: str | None = None,
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
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("Each decision needs a nonempty reason")
        seen.add(pair_id)
    seen -= duplicates
    if seen != expected and not allow_missing:
        raise ValueError("Model omitted candidate pairs: " + ", ".join(sorted(expected - seen)))
    return [normalize_decision(item, match_definition) for item in data["decisions"] if item["pair_id"] not in duplicates]


def is_block_row_assessment(item, left_refs):
    """A null candidate is an explicit row outcome, never a join relationship."""
    if "right_ref" not in item or item["right_ref"] is not None:
        return False
    ref = item.get("left_ref")
    if not isinstance(ref, str) or ref not in left_refs:
        raise ValueError("Unknown row assessment left_ref")
    if item.get("decision") not in ("unmatch", "uncertain"):
        raise ValueError("Null right_ref requires unmatch or uncertain, never match")
    if not isinstance(item.get("reason"), str) or not item["reason"].strip():
        raise ValueError("Each row assessment needs a nonempty reason")
    return True


def parse_block_decisions(response: str, pairs: list[dict], *, match_definition=None) -> list[dict]:
    """Resolve pair decisions; null references represent separate row assessments."""
    data = json.loads(response)
    if not isinstance(data, dict) or not isinstance(data.get("decisions"), list):
        raise ValueError("Model must return a decisions array")
    lookup = {(p["left_ref"], p["right_ref"]): p["pair_id"] for p in pairs}
    left_refs = {p["left_ref"] for p in pairs}
    decisions, seen = [], set()
    for item in data["decisions"]:
        if not isinstance(item, dict):
            raise ValueError("Each decision must be an object")
        if is_block_row_assessment(item, left_refs):
            continue
        left_ref, right_ref = item.get("left_ref"), item.get("right_ref")
        if not isinstance(left_ref, str) or not isinstance(right_ref, str):
            raise ValueError("Each block decision needs left_ref and right_ref")
        refs = (left_ref, right_ref)
        if refs not in lookup:
            raise ValueError(f"Unknown block relationship {refs!r}")
        if refs in seen:
            raise ValueError(f"Duplicate block relationship {refs!r}")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("Each decision needs a nonempty reason")
        seen.add(refs)
        decisions.append({**normalize_decision(item, match_definition), "pair_id": lookup[refs]})
    return decisions


BLOCK_COVERAGE_RETRY = (
    "COVERAGE RETRY: The previous response was invalid or omitted left-row assessments. "
    "Assess every left row in this block. Return candidate pairs or reasoned null-right_ref "
    "unmatch/uncertain row assessments; do not force matches.\n\n"
)


class MissingBlockAssessment(ValueError):
    """A syntactically valid response did not account for all left rows."""


def validate_block_coverage(response, left, right, decisions):
    """Require row assessments without materializing omitted Cartesian pairs."""
    data = json.loads(response)
    assessments = data.get("assessments", [])
    if not isinstance(assessments, list):
        raise ValueError("assessments must be an array")
    assessments = list(assessments)
    for item in data.get("decisions", []):
        if isinstance(item, dict) and is_block_row_assessment(item, left):
            assessments.append({"left_ref": item["left_ref"], "reason": item["reason"],
                                "status": "no_partner" if item["decision"] == "unmatch" else "unresolved"})
    explicit = {}
    for item in assessments:
        if not isinstance(item, dict):
            raise ValueError("Each assessment must be an object")
        if "right_ref" in item or "decision" in item:
            raise ValueError("Pair decisions belong in decisions, not assessments")
        ref = item.get("left_ref")
        if not isinstance(ref, str) or ref not in left or ref in explicit:
            raise ValueError("Unknown or duplicate assessment left_ref")
        if item.get("status") not in ("no_partner", "unresolved"):
            raise ValueError("Assessment status must be no_partner or unresolved")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("Each assessment needs a nonempty reason")
        explicit[ref] = item
    by_left = {ref: [] for ref in left}
    for decision in decisions:
        by_left[decision["left_ref"]].append(decision)
    coverage, missing = [], []
    for ref, items in by_left.items():
        selected = any(item["decision"] in ("match", "uncertain") for item in items)
        rejected_all = bool(right) and len(items) == len(right) and all(item["decision"] == "unmatch" for item in items)
        assessment = explicit.get(ref)
        if assessment and selected:
            raise ValueError("Block assessment contradicts a selected pair for " + ref)
        if assessment and assessment["status"] == "unresolved" and rejected_all:
            raise ValueError("Unresolved assessment contradicts rejection of every candidate for " + ref)
        if selected:
            coverage.append({"left_ref": ref, "status": "selected", "basis": "pair_decisions"})
        elif assessment:
            coverage.append({**assessment, "basis": "block_assessment"})
        elif rejected_all:
            coverage.append({"left_ref": ref, "status": "no_partner", "basis": "explicit_rejections"})
        else:
            missing.append(ref)
    if missing:
        raise MissingBlockAssessment("Missing left-row assessments: " + ", ".join(missing))
    return coverage


def block_assessment_summary(left, right, decisions, coverage):
    counts = Counter(item["decision"] for item in decisions)
    return {"assessed_left_rows": len(coverage),
            "selected_left_rows": sum(item["status"] == "selected" for item in coverage),
            "no_partner_left_rows": sum(item["status"] == "no_partner" for item in coverage),
            "unresolved_left_rows": sum(item["status"] == "unresolved" for item in coverage),
            "explicit_matches": counts["match"], "explicit_rejections": counts["unmatch"],
            "uncertain_pairs": counts["uncertain"],
            "validation_downgrades": sum("validation_warning" in item for item in decisions),
            "omitted_pairs": len(left) * len(right) - len(decisions)}


def aggregate_assessment_summary(requests):
    counts = Counter()
    for request in requests:
        counts.update(request.get("assessment_summary") or {})
    return dict(counts)


def model_from_env(provider: str, model: str, *, context_tokens=None, max_output_tokens=None) -> Callable[[str], str]:
    """Use the gateway's environment conventions without loading its services."""
    if provider not in ("ollama", "openai", "openai-compatible"):
        raise ValueError("Unsupported provider")

    def call(prompt: str) -> str:
        import requests
        limits = resolve_matching_limits(context_tokens, max_output_tokens)

        if provider == "ollama":
            response = requests.post(
                os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "format": "json",
                      "options": {"temperature": 0, "num_ctx": limits["context_tokens"],
                                  "num_predict": limits["max_output_tokens"]}},
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
                  "temperature": 0, "stream": False,
                  "max_tokens": limits["max_output_tokens"]},
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
    *, batch_size: int | None = None, max_pairs: int = 10000,
    context_tokens: int | None = None, max_output_tokens: int | None = None,
    cardinality: str = "one_to_one",
    matching_mode: str | None = None, block_rows: int | None = None,
    model_info: dict | None = None, progress: Callable[[dict], None] | None = None,
    request_started: Callable[[dict], None] | None = None,
    matching_context: Callable[[list[dict]], dict] | None = None,
    candidate_filter: Callable | None = None,
) -> dict:
    """Compare all pairs using row blocks by default; retain legacy pair batching.

    None model_call writes prompts only, for dry runs.

    A fresh run directory holds prompts, responses, audit, and matches.csv.
    On failure, retain diagnostics but never publish a partial mapping.
    """
    limits = resolve_matching_limits(context_tokens, max_output_tokens)
    context_tokens, max_output_tokens = limits["context_tokens"], limits["max_output_tokens"]
    if (batch_size is not None and batch_size < 1) or max_pairs < 1:
        raise ValueError("Positive limits are required")
    matching_mode = matching_mode or os.getenv("MATCH_MODE", "block")
    block_rows = block_rows if block_rows is not None else int(os.getenv("MATCH_BLOCK_ROWS", "8"))
    if matching_mode not in ("row", "block") or block_rows < 1:
        raise ValueError("MATCH_MODE must be row or block and block_rows must be positive")
    def make_prompt(definition, batch):
        if matching_mode == "row":
            prompt = build_match_prompt(definition, batch)
        else:
            prompt = build_block_match_prompt(definition,
                {p["left_ref"]: p["left"] for p in batch},
                {p["right_ref"]: p["right"] for p in batch})
        if matching_context is not None:
            context = matching_context([{**p, **identities[p["pair_id"]]} for p in batch])
            prompt = with_matching_context(prompt, context)
        return prompt
    match_definition = (match_definition or "").strip() or DEFAULT_MATCH_DEFINITION
    if cardinality not in ("one_to_one", "many_to_many"):
        raise ValueError("Unsupported cardinality")
    a, a_hash = read_source(left, left_key, left_fields, allow_empty_fields=matching_context is not None)
    b, b_hash = read_source(right, right_key, right_fields, allow_empty_fields=matching_context is not None)
    restriction = None
    if candidate_filter is not None:
        a, b, restriction = candidate_filter(a, b)
    if len(a) * len(b) > max_pairs:
        raise ValueError("Candidate count exceeds max_pairs; narrow inputs or explicitly raise the limit")

    pairs, identities = [], {}
    for ai, ra in enumerate(a):
        for bi, rb in enumerate(b):
            pid = f"p{len(pairs)}"
            pairs.append({"pair_id": pid, "left": ra["attributes"], "right": rb["attributes"]})
            if matching_mode == "block":
                pairs[-1].update(left_ref=f"l{ai}", right_ref=f"r{bi}")
            identities[pid] = {"idA": ra["id"], "idB": rb["id"]}
    # Estimate from UTF-8 bytes rather than characters to account for Unicode.
    # This is a heuristic, not a model tokenizer; leave 20% context headroom.
    token_budget = limits["usable_tokens"]

    def prompt_tokens(batch):
        return (len(make_prompt(match_definition, batch).encode("utf-8")) + 2) // 3

    def fits(batch):
        return ((batch_size is None or len(batch) <= batch_size)
                and prompt_tokens(batch) + max_output_tokens
                    + ((len(BLOCK_COVERAGE_RETRY.encode("utf-8")) + 2) // 3 if matching_mode == "block" else 0) <= token_budget)

    if matching_mode == "block":
        # Bound both row dimensions and response size. Reserve approximately 96
        # tokens per decision (including a concise reason); actual limits can retry.
        response_pair_cap = max(1, max_output_tokens // 96)
        pair_lookup = {(p["left_ref"], p["right_ref"]): p for p in pairs}
        batches = []

        def add_block(left_indices, right_indices):
            batch = [pair_lookup[(f"l{i}", f"r{j}")]
                     for i in left_indices for j in right_indices]
            if fits(batch) and len(batch) <= response_pair_cap:
                batches.append(batch)
                return
            if len(left_indices) == len(right_indices) == 1:
                raise ValueError("One candidate exceeds the prompt/context budget; select fewer fields or raise the limits")
            if len(left_indices) == len(right_indices) and len(left_indices) > 1:
                middle = len(left_indices) // 2
                for li in (left_indices[:middle], left_indices[middle:]):
                    for ri in (right_indices[:middle], right_indices[middle:]):
                        add_block(li, ri)
            elif len(left_indices) >= len(right_indices) and len(left_indices) > 1:
                middle = len(left_indices) // 2
                add_block(left_indices[:middle], right_indices)
                add_block(left_indices[middle:], right_indices)
            else:
                middle = len(right_indices) // 2
                add_block(left_indices, right_indices[:middle])
                add_block(left_indices, right_indices[middle:])

        for i in range(0, len(a), block_rows):
            for j in range(0, len(b), block_rows):
                add_block(list(range(i, min(i + block_rows, len(a)))),
                          list(range(j, min(j + block_rows, len(b)))))
    else:
        batches, current = [], []
        for pair in pairs:
            proposal = current + [pair]
            if not fits(proposal):
                if current:
                    batches.append(current)
                current = [pair]
            else:
                current = proposal
            if not fits(current):
                raise ValueError("One candidate exceeds the prompt/context budget; select fewer fields or raise the limits")
        if current:
            batches.append(current)

    output.mkdir(parents=True, exist_ok=False)
    audit = {"status": "running", "prompt_version": BLOCK_PROMPT_VERSION if matching_mode == "block" else PROMPT_VERSION,
             "acceptance_policy": "match_only",
             "candidate_restriction": restriction,
             "matching_mode": matching_mode,
             "omitted_relationships": "not_explicitly_assessed" if matching_mode == "block" else "retry",
             "coverage_policy": "each_left_row" if matching_mode == "block" else "each_pair",
             "left": {"path": str(left.resolve()), "key": left_key, "fields": left_fields, "sha256": a_hash},
             "right": {"path": str(right.resolve()), "key": right_key, "fields": right_fields, "sha256": b_hash},
             "match_definition": match_definition, "cardinality": cardinality,
             "model": model_info or {}, "candidate_count": len(pairs), "batch_count": len(batches),
             "batching": {"mode": "dynamic" if batch_size is None else "capped",
                          **limits,
                          "batch_size_cap": batch_size, "block_rows_cap": block_rows if matching_mode == "block" else None,
                          "response_pair_cap": response_pair_cap if matching_mode == "block" else None, "context_tokens": context_tokens,
                          "usable_tokens": token_budget,
                          "max_output_tokens": max_output_tokens,
                          "pair_counts": [len(batch) for batch in batches],
                          "estimated_prompt_tokens": [prompt_tokens(batch) for batch in batches]},
             "pair_ids": identities, "decisions": [], "requests": []}
    coverage_retries = set()
    total_requests = len(batches)
    completed_requests = 0

    def record_request(batch, prompt_file, *, is_retry=False):
        metadata = {"prompt_file": prompt_file, "pair_count": len(batch),
                    "pair_ids": [pair["pair_id"] for pair in batch], "is_retry": is_retry,
                    "max_output_tokens": max_output_tokens, "context_tokens": context_tokens}
        if matching_mode == "block":
            metadata.update(left_row_count=len({p["left_ref"] for p in batch}),
                            right_row_count=len({p["right_ref"] for p in batch}))
        audit["requests"].append(metadata)
        if request_started:
            request_started(dict(metadata))

    def report_progress():
        if progress:
            progress({"total_requests": total_requests, "completed_requests": completed_requests,
                      "planned_requests": len(batches), "candidate_count": len(pairs)})

    try:
        report_progress()
        for index, batch in enumerate(batches):
            prompt = make_prompt(match_definition, batch)
            if id(batch) in coverage_retries:
                prompt = BLOCK_COVERAGE_RETRY + prompt
            (output / f"prompt_{index:04d}.txt").write_text(prompt, encoding="utf-8")
            record_request(batch, f"prompt_{index:04d}.txt", is_retry=id(batch) in coverage_retries)
            if model_call is None:
                continue
            try:
                response = model_call(prompt)
            except Exception as exc:
                if not getattr(exc, "output_token_limit", False) or len(batch) <= 1:
                    raise
                # Retry every pair; never accept a truncated decision list.
                if matching_mode == "block":
                    left_refs = list(dict.fromkeys(p["left_ref"] for p in batch))
                    right_refs = list(dict.fromkeys(p["right_ref"] for p in batch))
                    field, refs = (("left_ref", left_refs) if len(left_refs) >= len(right_refs)
                                   else ("right_ref", right_refs))
                    first = set(refs[:len(refs) // 2])
                    children = [[p for p in batch if p[field] in first],
                                [p for p in batch if p[field] not in first]]
                else:
                    midpoint = len(batch) // 2
                    children = [batch[:midpoint], batch[midpoint:]]
                batches[index + 1:index + 1] = children
                audit["requests"][-1]["status"] = "output_token_limit"
                audit.setdefault("retries", []).append({
                    "batch_index": index, "reason": "output_token_limit",
                    "pair_ids": [pair["pair_id"] for pair in batch],
                })
                total_requests += 2
                completed_requests += 1
                report_progress()
                continue
            (output / f"response_{index:04d}.txt").write_text(response, encoding="utf-8")
            if matching_mode == "block":
                lrows = {p["left_ref"]: p["left"] for p in batch}
                rrows = {p["right_ref"]: p["right"] for p in batch}
                try:
                    decisions = parse_block_decisions(response, batch, match_definition=match_definition)
                    coverage = validate_block_coverage(response, lrows, rrows, decisions)
                except ValueError as exc:
                    incomplete = isinstance(exc, MissingBlockAssessment)
                    audit["requests"][-1].update(status="incomplete_assessment" if incomplete else "invalid_response", error=str(exc))
                    if id(batch) in coverage_retries:
                        raise ValueError(f"Block assessment failed after one retry: {exc}") from exc
                    coverage_retries.add(id(batch))
                    batches.insert(index + 1, batch)
                    audit.setdefault("retries", []).append({"batch_index": index, "reason": "missing_assessments" if incomplete else "invalid_response"})
                    total_requests += 1
                    completed_requests += 1
                    report_progress()
                    continue
                audit["requests"][-1].update(assessments=coverage,
                    assessment_summary=block_assessment_summary(lrows, rrows, decisions, coverage))
                audit["decisions"].extend(decisions)
                completed_requests += 1
                report_progress()
                continue
            decisions = parse_decisions(response, {p["pair_id"] for p in batch},
                                        allow_missing=True, defer_duplicates=True, match_definition=match_definition)
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
                retry_prompt = make_prompt(match_definition, [pair])
                suffix = f"{index:04d}_retry_{pair['pair_id']}"
                (output / f"prompt_{suffix}.txt").write_text(retry_prompt, encoding="utf-8")
                record_request([pair], f"prompt_{suffix}.txt", is_retry=True)
                retry_response = model_call(retry_prompt)
                (output / f"response_{suffix}.txt").write_text(retry_response, encoding="utf-8")
                completed_requests += 1
                report_progress()
                try:
                    retry_decisions = parse_decisions(retry_response, {pair["pair_id"]}, match_definition=match_definition)
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
        audit["assessment_summary"] = aggregate_assessment_summary(audit["requests"])
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
    parser.add_argument("--matching-mode", choices=["block", "row"], help="Default: MATCH_MODE or block")
    parser.add_argument("--block-rows", type=int, help="Rows per side per block (MATCH_BLOCK_ROWS or 8)")
    parser.add_argument("--batch-size", type=int, help="Optional pair cap; default: dynamic batching")
    parser.add_argument("--context-tokens", type=int, help="Context budget (MATCH_CONTEXT_TOKENS, OLLAMA_NUM_CTX, or 8192)")
    parser.add_argument("--max-pairs", type=int, default=10000)
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
        None if args.dry_run else model_from_env(args.provider, args.model, context_tokens=args.context_tokens),
        batch_size=args.batch_size, max_pairs=args.max_pairs,
        matching_mode=args.matching_mode, block_rows=args.block_rows,
        context_tokens=args.context_tokens,
        cardinality=args.cardinality, model_info={"provider": args.provider, "model": args.model},
    )
    print(json.dumps({"status": audit["status"], "candidates": audit["candidate_count"],
                      "accepted": audit.get("accepted_count"), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
