#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent

os.environ.setdefault("GATEWAY_LOG_PATH", str(REPO_ROOT / "logs" / "provsql_gateway_logs.jsonl"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import gateway  # noqa: E402
    from prompt_internal_knowledge import PROMPT_INTERNAL_KNOWLEDGE_TEMPLATE  # noqa: E402
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python dependency '{missing}'. Run this locally with the project environment, "
        f"for example: .venv/bin/python {Path(__file__).relative_to(REPO_ROOT)}"
    ) from exc


DEFAULT_INPUT = REPO_ROOT / "evaluation" / "questions.json"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "internal_knowledge_outputs.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def load_completed_record_ids(path: Path) -> Set[str]:
    completed: Set[str] = set()
    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[resume] ignoring invalid JSONL line {line_no} in {path}", file=sys.stderr)
                continue
            record_id = record.get("record_id")
            if record_id and record.get("ok") is True:
                completed.add(str(record_id))
    return completed


def iter_eval_items(
    questions: Iterable[Dict[str, Any]],
    query_ids: Optional[Set[str]],
) -> Iterable[Dict[str, Any]]:
    for query in questions:
        query_id = str(query.get("query_id", "")).strip()
        if query_ids and query_id not in query_ids:
            continue
        yield {
            "record_id": f"{query_id}:internal-knowledge",
            "query_id": query_id,
            "question_nl": query.get("question_nl"),
            "question_sql": query.get("question_sql"),
        }


def extract_json_array_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    decoder = json.JSONDecoder()
    first_error: Optional[str] = None
    for start, char in enumerate(text):
        if char != "[":
            continue
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            if first_error is None:
                first_error = str(exc)
            continue
        if isinstance(obj, list):
            return text[start : start + end], None
    return None, first_error or "No JSON array found in model output"


def parse_internal_output(text: str) -> Tuple[bool, Optional[str], Optional[List[Dict[str, Any]]]]:
    json_text, extract_error = extract_json_array_text(text)
    if json_text is None:
        return False, extract_error, None

    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return False, str(exc), None

    if not isinstance(obj, list):
        return False, "Root JSON is not an array", None

    errors = []
    valid_items: List[Dict[str, Any]] = []
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            errors.append(f"Item {i} is not an object")
            continue
        if set(item.keys()) != {"result", "provenance"}:
            errors.append(f"Item {i} must have exactly result and provenance")
            continue
        if not isinstance(item["result"], dict):
            errors.append(f"Item {i}.result must be an object")
            continue
        provenance = item["provenance"]
        if not isinstance(provenance, list):
            errors.append(f"Item {i}.provenance must be a list")
            continue
        provenance_ok = True
        for j, witness in enumerate(provenance):
            if not isinstance(witness, list) or not all(isinstance(pid, str) for pid in witness):
                errors.append(f"Item {i}.provenance[{j}] must be a list of strings")
                provenance_ok = False
                break
        if not provenance_ok:
            continue
        valid_items.append(item)

    if errors:
        shown = errors[:10]
        error_text = "; ".join(shown)
        if len(errors) > len(shown):
            error_text += f"; ... and {len(errors) - len(shown)} more"
        return False, error_text, valid_items

    return True, None, valid_items


def validate_internal_output(text: str) -> Tuple[bool, Optional[str]]:
    valid, error, _ = parse_internal_output(text)
    return valid, error


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def run_one(item: Dict[str, Any], ollama_model: str, temperature: float) -> Dict[str, Any]:
    started_at = utc_now()
    start = time.monotonic()
    prompt = ""
    out_text = ""
    parsed_output = None
    parse_error = None
    error = None

    try:
        question = str(item.get("question_nl") or "").strip()
        if not question:
            raise ValueError("Missing natural-language question")
        prompt = PROMPT_INTERNAL_KNOWLEDGE_TEMPLATE.format(question=question)
        out_text = gateway._call_model_with_retry(
            ollama_model,
            prompt,
            temperature,
            max_tries=2,
            validator=validate_internal_output,
        )
        valid_output, parse_error, parsed_output = parse_internal_output(out_text)
        ok = valid_output
    except Exception as exc:
        ok = False
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    elapsed = time.monotonic() - start
    return {
        "record_id": item["record_id"],
        "query_id": item["query_id"],
        "run_mode": "internal-knowledge",
        "question_nl": item["question_nl"],
        "question_sql": item["question_sql"],
        "ollama_model": ollama_model,
        "temperature": temperature,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(elapsed, 3),
        "ok": ok,
        "error": error,
        "prompt": prompt,
        "output_text": out_text,
        "parsed_output": parsed_output,
        "parse_error": parse_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run natural-language questions through the internal-knowledge TPC-H prompt."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query-id", action="append", help="Run only this query_id. May be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to run.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL_INTERNAL_KNOWLEDGE", os.getenv("OLLAMA_MODEL_BASE", "llama3:8b")),
        help="Actual Ollama model name.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip successful record_ids already present in the output JSONL.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Truncate the output file before running.")
    parser.add_argument("--verbose-gateway", action="store_true", help="Show gateway prompt logs on the console.")
    parser.add_argument(
        "--gateway-log-output",
        type=Path,
        default=None,
        help="Where to write gateway stdout when --verbose-gateway is not set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    gateway_stdout_path = (
        args.gateway_log_output.resolve()
        if args.gateway_log_output
        else output_path.with_suffix(output_path.suffix + ".gateway_stdout.log")
    )

    if args.overwrite:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    completed = load_completed_record_ids(output_path) if args.resume else set()
    query_ids = set(args.query_id) if args.query_id else None
    questions = load_questions(args.input)
    items = list(iter_eval_items(questions, query_ids))
    if args.limit is not None:
        items = items[: args.limit]

    print(f"[internal-eval] input={args.input}")
    print(f"[internal-eval] output={output_path}")
    print(f"[internal-eval] records={len(items)} resume={args.resume}")
    print(f"[internal-eval] ollama_model={args.ollama_model} temperature={args.temperature}")
    if not args.verbose_gateway:
        print(f"[internal-eval] gateway stdout={gateway_stdout_path}")

    ran = 0
    skipped = 0
    failed = 0
    gateway_stdout_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(items, start=1):
        record_id = item["record_id"]
        if record_id in completed:
            skipped += 1
            print(f"[internal-eval] skip {idx}/{len(items)} {record_id}")
            continue

        print(f"[internal-eval] run  {idx}/{len(items)} {record_id}")
        if args.verbose_gateway:
            record = run_one(item, ollama_model=args.ollama_model, temperature=args.temperature)
        else:
            with gateway_stdout_path.open("a", encoding="utf-8") as stdout_file:
                with contextlib.redirect_stdout(stdout_file):
                    record = run_one(item, ollama_model=args.ollama_model, temperature=args.temperature)

        append_jsonl(output_path, record)
        ran += 1
        if not record["ok"]:
            failed += 1
            message = record["error"]["message"] if record["error"] else record.get("parse_error")
            print(f"[internal-eval] fail {record_id}: {message}", file=sys.stderr)
        else:
            parsed = record.get("parsed_output") or []
            print(f"[internal-eval] ok   {record_id} output_rows={len(parsed)}")

    print(f"[internal-eval] done ran={ran} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
