#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import re
from pathlib import Path
from typing import Any


STOPWORDS = {
    "a", "an", "and", "are", "as", "by", "for", "from", "in", "is", "of",
    "on", "or", "the", "to", "with", "what", "which", "who", "show", "me",
    "rows", "row", "record", "records",
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9_#.-]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def score_document(question_terms: list[str], content: str, metadata: dict[str, Any]) -> float:
    text = content.lower()
    table = str(metadata.get("table") or "").lower()
    primary_key = json.dumps(metadata.get("primary_key") or {}, ensure_ascii=False).lower()

    score = 0.0
    for term in question_terms:
        count = text.count(term)
        if count:
            score += min(count, 10)
        if term == table:
            score += 25
        if term in primary_key:
            score += 8

    # Prefer documents where a queried table name appears in metadata, because the
    # row text itself includes many linked rows from other tables.
    if table and table in question_terms:
        score += 20

    return score


def iter_documents(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            doc = json.loads(line)
            yield line_number, doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick lexical search over row_documents.jsonl.")
    parser.add_argument("question", help="Question or retrieval query to test.")
    parser.add_argument(
        "--documents",
        default="faiss_index_tpch_rows_bge_m3/documents.jsonl",
        help="Path to row documents JSONL.",
    )
    parser.add_argument("--table", default="", help="Optional table filter, e.g. customer.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-docs", type=int, default=0, help="Only scan the first N docs. 0 scans all.")
    parser.add_argument("--preview-chars", type=int, default=1200)
    args = parser.parse_args()

    path = Path(args.documents)
    question_terms = tokenize(args.question)
    table_filter = args.table.strip().lower()
    heap: list[tuple[float, int, dict[str, Any]]] = []
    scanned = 0
    matched_table = 0

    for line_number, doc in iter_documents(path):
        scanned += 1
        if args.max_docs and scanned > args.max_docs:
            break

        metadata = doc.get("metadata") or {}
        if table_filter and str(metadata.get("table") or "").lower() != table_filter:
            continue
        matched_table += 1

        content = str(doc.get("page_content") or "")
        score = score_document(question_terms, content, metadata)
        if score <= 0:
            continue

        item = (score, line_number, doc)
        if len(heap) < args.top_k:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)

    results = sorted(heap, key=lambda item: (-item[0], item[1]))

    print(f"question: {args.question}")
    print(f"terms: {', '.join(question_terms) or '(none)'}")
    print(f"scanned_docs: {min(scanned, args.max_docs) if args.max_docs else scanned}")
    if table_filter:
        print(f"table_filter: {table_filter} ({matched_table} docs considered)")
    print()

    if not results:
        print("No matching documents found.")
        return

    for rank, (score, line_number, doc) in enumerate(results, start=1):
        metadata = doc.get("metadata") or {}
        content = str(doc.get("page_content") or "")
        print(f"=== rank {rank} | score={score:.1f} | jsonl_line={line_number} ===")
        print(f"table: {metadata.get('table')}")
        print(f"row_id: {metadata.get('row_id')}")
        print(f"rownum: {metadata.get('rownum_value')}")
        print(f"primary_key: {metadata.get('primary_key')}")
        print(f"linked_rows: {len(metadata.get('linked_rows') or [])}")
        print()
        print(content[: args.preview_chars])
        if len(content) > args.preview_chars:
            print("...")
        print()


if __name__ == "__main__":
    main()
