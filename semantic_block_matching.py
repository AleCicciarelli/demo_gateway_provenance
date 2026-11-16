"""Sparse row-block matching for contextual multi-join queries."""
import csv
import json
import os
from collections import Counter

from semantic_matching import (build_mapping, read_source, MATCH_PROBABILITIES, resolve_matching_limits, normalize_decision,
                               validate_block_coverage, block_assessment_summary, aggregate_assessment_summary,
                               MissingBlockAssessment, BLOCK_RETRY_RESERVE_TOKENS, block_retry_prompt,
                               is_block_row_assessment)
from semantic_match_prompt import (build_block_match_prompt, with_matching_context,
                                   DEFAULT_MATCH_DEFINITION, CONTEXT_BLOCK_PROMPT_VERSION)
from row_probability import PROBABILITY_COLUMN


def build_contextual_mapping(*args, **kwargs):
    """Keep explicit legacy row mode; use direct row blocks otherwise."""
    mode = kwargs.get("matching_mode") or os.getenv("MATCH_MODE", "block")
    if mode == "row":
        return build_mapping(*args, **kwargs)
    if mode != "block":
        raise ValueError("MATCH_MODE must be row or block")
    return build_sparse_block_mapping(*args, **kwargs)


def parse_sparse_block_decisions(response, left, right, right_count, *, match_definition=None):
    """Validate only returned relationships against the two row dictionaries."""
    data = json.loads(response)
    if not isinstance(data, dict) or not isinstance(data.get("decisions"), list):
        raise ValueError("Model must return a decisions array")
    decisions, seen = [], set()
    for item in data["decisions"]:
        if not isinstance(item, dict):
            raise ValueError("Each decision must be an object")
        if is_block_row_assessment(item, left):
            continue
        lref, rref = item.get("left_ref"), item.get("right_ref")
        if not isinstance(lref, str) or not isinstance(rref, str) or lref not in left or rref not in right:
            raise ValueError(f"Unknown block relationship {(lref, rref)!r}")
        if (lref, rref) in seen:
            raise ValueError(f"Duplicate block relationship {(lref, rref)!r}")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("Each decision needs a nonempty reason")
        seen.add((lref, rref))
        # Assign an ID only to a selected relationship. No Cartesian lookup.
        decisions.append({**normalize_decision(item, match_definition), "pair_id": f"p{int(lref[1:]) * right_count + int(rref[1:])}"})
    return decisions


def build_sparse_block_mapping(
    left, right, left_key, right_key, left_fields, right_fields, match_definition,
    output, model_call, *, batch_size=None, max_pairs=10000, context_tokens=None,
    max_output_tokens=None, cardinality="one_to_one", matching_mode=None,
    block_rows=None, model_info=None, progress=None, request_started=None,
    matching_context=None, candidate_filter=None,
):
    """Audit returned row-block decisions; publish only explicit matches.

    max_pairs is a legacy pair-mode option, not a total relationship-space cap.
    Blocks split for input context, explicit batch_size, or an actual output-limit
    response. We do not reserve a decision for every potential relationship.
    """
    limits = resolve_matching_limits(context_tokens, max_output_tokens)
    context_tokens, max_output_tokens = limits["context_tokens"], limits["max_output_tokens"]
    block_rows = block_rows if block_rows is not None else int(os.getenv("MATCH_BLOCK_ROWS", "8"))
    if (min(context_tokens, max_output_tokens, block_rows, max_pairs) < 1
            or (batch_size is not None and batch_size < 1)):
        raise ValueError("Positive limits are required")
    if cardinality not in ("one_to_one", "many_to_many"):
        raise ValueError("Unsupported cardinality")
    token_budget = limits["usable_tokens"]
    a, ah = read_source(left, left_key, left_fields, allow_empty_fields=matching_context is not None)
    b, bh = read_source(right, right_key, right_fields, allow_empty_fields=matching_context is not None)
    restriction = None
    if candidate_filter is not None:
        a, b, restriction = candidate_filter(a, b)
    left_rows = {f"l{i}": row for i, row in enumerate(a)}
    right_rows = {f"r{i}": row for i, row in enumerate(b)}
    definition = (match_definition or "").strip() or DEFAULT_MATCH_DEFINITION

    def prompt_for(block):
        lrows, rrows = block
        prompt = build_block_match_prompt(definition,
            {ref: row["attributes"] for ref, row in lrows.items()},
            {ref: row["attributes"] for ref, row in rrows.items()})
        if matching_context is not None:
            prompt = with_matching_context(prompt, matching_context({"left": lrows, "right": rrows}))
        return prompt

    def split(block):
        lrows, rrows = block
        if len(lrows) == len(rrows) == 1:
            raise ValueError("One candidate exceeds the prompt/context budget; select fewer fields or raise the limits")
        if len(lrows) >= len(rrows):
            items = list(lrows.items())
            mid = len(items) // 2
            return [(dict(items[:mid]), rrows), (dict(items[mid:]), rrows)]
        items = list(rrows.items())
        mid = len(items) // 2
        return [(lrows, dict(items[:mid])), (lrows, dict(items[mid:]))]

    blocks = []
    def plan(block):
        tokens = (len(prompt_for(block).encode("utf-8")) + 2) // 3
        if (tokens + max_output_tokens + BLOCK_RETRY_RESERVE_TOKENS <= token_budget and
                (batch_size is None or len(block[0]) * len(block[1]) <= batch_size)):
            blocks.append(block)
        else:
            for child in split(block):
                plan(child)

    la, rb = list(left_rows.items()), list(right_rows.items())
    for i in range(0, len(la), block_rows):
        for j in range(0, len(rb), block_rows):
            plan((dict(la[i:i + block_rows]), dict(rb[j:j + block_rows])))
    output.mkdir(parents=True, exist_ok=False)
    audit = {"status": "running", "matching_mode": "block", "prompt_version": CONTEXT_BLOCK_PROMPT_VERSION,
        "acceptance_policy": "match_only",
        "candidate_restriction": restriction,
        "omitted_relationships": "not_explicitly_assessed", "coverage_policy": "each_left_row", "match_definition": definition, "cardinality": cardinality,
        "left": {"path": str(left.resolve()), "key": left_key, "fields": left_fields, "sha256": ah},
        "right": {"path": str(right.resolve()), "key": right_key, "fields": right_fields, "sha256": bh},
        "model": model_info or {}, "left_row_count": len(a), "right_row_count": len(b),
        "potential_relationship_count": len(a) * len(b), "batch_count": len(blocks),
        "batching": {**limits, "block_rows_cap": block_rows},
        "pair_ids": {}, "decisions": [], "requests": []}
    planned = len(blocks)
    completed = 0
    retry_blocks = set()
    coverage_retries = {}

    def report():
        if progress:
            progress({"total_requests": len(blocks), "planned_requests": planned,
                      "completed_requests": completed, "left_row_count": len(a), "right_row_count": len(b)})

    try:
        report()
        for index, block in enumerate(blocks):
            lrows, rrows = block
            prompt = prompt_for(block)
            if id(block) in coverage_retries:
                prompt = block_retry_prompt(prompt, coverage_retries[id(block)])
            filename = f"prompt_{index:04d}.txt"
            (output / filename).write_text(prompt, encoding="utf-8")
            metadata = {"prompt_file": filename, "left_row_count": len(lrows), "right_row_count": len(rrows),
                        "left_refs": list(lrows), "right_refs": list(rrows), "is_retry": id(block) in retry_blocks,
                        "max_output_tokens": max_output_tokens, "context_tokens": context_tokens}
            audit["requests"].append(metadata)
            if request_started:
                request_started(dict(metadata))
            if model_call is None:
                continue
            try:
                response = model_call(prompt)
            except Exception as exc:
                if not getattr(exc, "output_token_limit", False) or (len(lrows) == len(rrows) == 1):
                    raise
                children = split(block)
                retry_blocks.update(id(child) for child in children)
                blocks[index + 1:index + 1] = children
                metadata["status"] = "output_token_limit"
                audit.setdefault("retries", []).append({"batch_index": index, "reason": "output_token_limit"})
                completed += 1
                report()
                continue
            (output / f"response_{index:04d}.txt").write_text(response, encoding="utf-8")
            try:
                selected = parse_sparse_block_decisions(response, lrows, rrows, len(b), match_definition=definition)
                coverage = validate_block_coverage(response, lrows, rrows, selected)
            except ValueError as exc:
                incomplete = isinstance(exc, MissingBlockAssessment)
                metadata.update(status="incomplete_assessment" if incomplete else "invalid_response", error=str(exc))
                if id(block) in coverage_retries:
                    raise ValueError(f"Block assessment failed after one retry: {exc}") from exc
                coverage_retries[id(block)] = str(exc)
                retry_blocks.add(id(block))
                blocks.insert(index + 1, block)
                audit.setdefault("retries", []).append({"batch_index": index, "reason": "missing_assessments" if incomplete else "invalid_response"})
                completed += 1
                report()
                continue
            metadata.update(assessments=coverage,
                assessment_summary=block_assessment_summary(lrows, rrows, selected, coverage))
            audit["decisions"].extend(selected)
            for decision in selected:
                audit["pair_ids"][decision["pair_id"]] = {
                    "idA": lrows[decision["left_ref"]]["id"], "idB": rrows[decision["right_ref"]]["id"]}
            completed += 1
            report()
        if model_call is None:
            audit["status"] = "dry_run"
        else:
            matches = [d for d in audit["decisions"] if d["decision"] == "match"]
            lc = Counter(d["left_ref"] for d in matches)
            rc = Counter(d["right_ref"] for d in matches)
            accepted, rows = [], []
            for d in audit["decisions"]:
                conflict = cardinality == "one_to_one" and d["decision"] == "match" and (
                    lc[d["left_ref"]] > 1 or rc[d["right_ref"]] > 1)
                d["disposition"] = "cardinality_conflict" if conflict else d["decision"]
                if conflict or d["decision"] != "match":
                    continue
                probability = MATCH_PROBABILITIES[d["decision"]]
                rows.append({**audit["pair_ids"][d["pair_id"]], PROBABILITY_COLUMN: probability})
                accepted.append({"pair_id": d["pair_id"], "reason": d["reason"], "decision": d["decision"],
                    "probability": probability, "left": left_rows[d["left_ref"]], "right": right_rows[d["right_ref"]]})
            with (output / "matches.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["idA", "idB", PROBABILITY_COLUMN])
                writer.writeheader()
                writer.writerows(rows)
            audit.update(status="complete", accepted_count=len(rows), accepted_pairs=accepted)
    except Exception as exc:
        audit.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        audit["assessment_summary"] = aggregate_assessment_summary(audit["requests"])
        (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit
