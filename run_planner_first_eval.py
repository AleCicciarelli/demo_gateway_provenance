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
from typing import Any, Dict, Iterable, List, Optional, Set


REPO_ROOT = Path(__file__).resolve().parent

# gateway.py defaults to Docker paths. For local evaluation, prefer the repo data.
os.environ.setdefault("CSV_DIR", str(REPO_ROOT / "rel-f1-csv"))
os.environ.setdefault("FAISS_INDEX_FOLDER", str(REPO_ROOT / "faiss_index_relf1_rows_bge_m3"))
os.environ.setdefault("EMB_MODEL", "BAAI/bge-m3")
os.environ.setdefault("EMB_STRATEGY", "bge-m3")
os.environ.setdefault("GATEWAY_LOG_PATH", str(REPO_ROOT / "logs" / "provsql_gateway_logs.jsonl"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import gateway  # noqa: E402
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python dependency '{missing}'. Run this locally with the project environment, "
        f"for example: .venv/bin/python {Path(__file__).relative_to(REPO_ROOT)}"
    ) from exc


DEFAULT_INPUT = REPO_ROOT / "evaluation_relf1" / "leaf_node_questions.json"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation_relf1" / "planner_first_outputs_70b_4.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def load_completed_record_ids(path: Path, *, include_failed: bool = True) -> Set[str]:
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
            if record_id and (include_failed or record.get("ok") is True):
                completed.add(str(record_id))
    return completed


def iter_eval_items(
    questions: Iterable[Dict[str, Any]],
    mode: str,
    query_ids: Optional[Set[str]],
) -> Iterable[Dict[str, Any]]:
    for query in questions:
        query_id = str(query.get("query_id", "")).strip()
        if query_ids and query_id not in query_ids:
            continue

        if mode in {"root", "both"}:
            yield {
                "record_id": f"{query_id}:root",
                "run_mode": "root",
                "query_id": query_id,
                "question_nl": query.get("question_nl"),
                "question_sql": query.get("question_sql"),
                "leaf_index": None,
                "leaf_table_name": None,
                "leaf_question_nl": None,
                "leaf_question_sql": None,
                "sql_to_run": query.get("question_sql"),
            }

        if mode in {"leaf", "both"}:
            for leaf_index, leaf in enumerate(query.get("leaf_tasks") or []):
                table_name = str(leaf.get("table_name", "")).strip()
                yield {
                    "record_id": f"{query_id}:leaf:{leaf_index}:{table_name}",
                    "run_mode": "leaf",
                    "query_id": query_id,
                    "question_nl": query.get("question_nl"),
                    "question_sql": query.get("question_sql"),
                    "leaf_index": leaf_index,
                    "leaf_table_name": table_name,
                    "leaf_question_nl": leaf.get("question_nl"),
                    "leaf_question_sql": leaf.get("question_sql"),
                    "sql_to_run": leaf.get("question_sql"),
                }


def summarize_leaf_outputs(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not result:
        return {}

    leaf_outputs = result.get("leaf_outputs") or []
    summaries = []
    for leaf in leaf_outputs:
        parsed = leaf.get("parsed_output")
        summaries.append(
            {
                "table_name": leaf.get("table_name"),
                "retrieval_query": leaf.get("retrieval_query"),
                "valid_leaf_output": leaf.get("valid_leaf_output"),
                "parse_error": leaf.get("parse_error"),
                "context_rows": sum(len(rows or {}) for rows in (leaf.get("context_data") or {}).values()),
                "output_rows": len(parsed) if isinstance(parsed, list) else None,
            }
        )

    return {
        "leaf_task_count": len(leaf_outputs),
        "valid_leaf_task_count": sum(1 for item in summaries if item["valid_leaf_output"] is True),
        "leaf_outputs": summaries,
    }


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def run_one(item: Dict[str, Any], ollama_model: str, temperature: float) -> Dict[str, Any]:
    started_at = utc_now()
    start = time.monotonic()
    result = None
    error = None

    try:
        sql = item.get("sql_to_run")
        if not sql:
            raise ValueError("Missing SQL to run")
        result = gateway._run_planner_first(
            sql_query=str(sql),
            ollama_model=ollama_model,
            temperature=temperature,
        )
        ok = True
    except Exception as exc:  # keep going so a long eval run is not lost
        ok = False
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    elapsed = time.monotonic() - start
    return {
        "record_id": item["record_id"],
        "query_id": item["query_id"],
        "run_mode": item["run_mode"],
        "question_nl": item["question_nl"],
        "question_sql": item["question_sql"],
        "leaf_index": item["leaf_index"],
        "leaf_table_name": item["leaf_table_name"],
        "leaf_question_nl": item["leaf_question_nl"],
        "leaf_question_sql": item["leaf_question_sql"],
        "sql_to_run": item["sql_to_run"],
        "ollama_model": ollama_model,
        "temperature": temperature,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(elapsed, 3),
        "ok": ok,
        "error": error,
        "summary": summarize_leaf_outputs(result),
        "planner_first_result": result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation SQL through gateway.py planner-first iterative pipeline."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("root", "leaf", "both"),
        default="root",
        help="root runs each full query SQL; leaf runs each leaf_tasks[*].question_sql.",
    )
    parser.add_argument("--query-id", action="append", help="Run only this query_id. May be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to run.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Actual Ollama model name. Defaults to gateway MODEL_ROUTING['planner-first'].",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip record_ids already present in the output JSONL.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="With --resume, retry existing records whose previous row has ok=false.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Truncate the output file before running.",
    )
    parser.add_argument(
        "--verbose-gateway",
        action="store_true",
        help="Show gateway prompt/retrieval logs on the console.",
    )
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

    completed = (
        load_completed_record_ids(output_path, include_failed=not args.retry_failed)
        if args.resume
        else set()
    )
    query_ids = set(args.query_id) if args.query_id else None
    questions = load_questions(args.input)
    items = list(iter_eval_items(questions, args.mode, query_ids))
    if args.limit is not None:
        items = items[: args.limit]

    ollama_model = args.ollama_model or gateway.MODEL_ROUTING["planner-first"]

    print(f"[eval] input={args.input}")
    print(f"[eval] output={output_path}")
    print(f"[eval] mode={args.mode} records={len(items)} resume={args.resume}")
    if args.resume:
        print(f"[eval] resume_skip_existing={not args.retry_failed} completed={len(completed)}")
    print(f"[eval] ollama_model={ollama_model} temperature={args.temperature}")
    if not args.verbose_gateway:
        print(f"[eval] gateway stdout={gateway_stdout_path}")

    ran = 0
    skipped = 0
    failed = 0

    gateway_stdout_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(items, start=1):
        record_id = item["record_id"]
        if record_id in completed:
            skipped += 1
            print(f"[eval] skip {idx}/{len(items)} {record_id}")
            continue

        print(f"[eval] run  {idx}/{len(items)} {record_id}")
        if args.verbose_gateway:
            record = run_one(item, ollama_model=ollama_model, temperature=args.temperature)
        else:
            with gateway_stdout_path.open("a", encoding="utf-8") as stdout_file:
                with contextlib.redirect_stdout(stdout_file):
                    record = run_one(item, ollama_model=ollama_model, temperature=args.temperature)

        append_jsonl(output_path, record)
        completed.add(record_id)
        ran += 1
        if not record["ok"]:
            failed += 1
            print(f"[eval] fail {record_id}: {record['error']['message']}", file=sys.stderr)
        else:
            summary = record.get("summary") or {}
            valid = summary.get("valid_leaf_task_count")
            total = summary.get("leaf_task_count")
            print(f"[eval] ok   {record_id} leaf_valid={valid}/{total}")

    print(f"[eval] done ran={ran} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
