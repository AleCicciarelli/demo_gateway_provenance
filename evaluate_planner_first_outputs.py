#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = REPO_ROOT / "evaluation" / "planner_first_outputs_70b_3.jsonl"
DEFAULT_GROUND_TRUTH = REPO_ROOT / "evaluation" / "ground_truth_leaf_tasks.json"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "planner_first_metrics_70b_3.json"
DEFAULT_PLOTS_DIR = REPO_ROOT / "evaluation" / "planner_first_plots_70b_3"


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            records.append(obj)
        return records

    obj = json.loads(text)
    if isinstance(obj, list):
        if not all(isinstance(item, dict) for item in obj):
            raise ValueError(f"Expected {path} to contain a list of JSON objects")
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError(f"Expected {path} to contain a JSON object/list or JSONL records")


def normalize_sql(sql: Optional[str]) -> str:
    if not sql:
        return ""
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def row_id_from_answer_row(row: Any) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    for key in ("__rid__", "row_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    for key, value in row.items():
        if key.endswith("_rownum") and isinstance(value, str) and value:
            return value
    return None


def extract_json_array(text: Any) -> Optional[List[Any]]:
    if not isinstance(text, str):
        return None

    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return obj
    return None


def valid_leaf_items_from_output_text(leaf_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    table_name = str(leaf_output.get("table_name") or "").strip()
    context_rows = ((leaf_output.get("context_data") or {}).get(table_name) or {})
    if not isinstance(context_rows, dict):
        return []

    items = extract_json_array(leaf_output.get("output_text"))
    if not isinstance(items, list):
        return []

    valid_items: List[Dict[str, Any]] = []
    seen_row_ids: Set[str] = set()
    for item in items:
        if not isinstance(item, dict) or set(item.keys()) != {"row_id", "values"}:
            continue

        row_id = item.get("row_id")
        values = item.get("values")
        if not isinstance(row_id, str) or row_id in seen_row_ids:
            continue
        if not isinstance(values, dict):
            continue
        if values != context_rows.get(row_id):
            continue

        seen_row_ids.add(row_id)
        valid_items.append(item)

    return valid_items


def parsed_leaf_items(leaf_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    parsed = leaf_output.get("parsed_output")
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return valid_leaf_items_from_output_text(leaf_output)


def normalize_row(row: Any) -> Dict[str, str]:
    if not isinstance(row, dict):
        return {}
    return {str(key): str(value).strip() for key, value in row.items() if key != "__rid__"}


def expected_rows_by_id(ground_truth: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    rows_by_id: Dict[str, Dict[str, str]] = {}
    for row in ground_truth.get("answer") or []:
        row_id = row_id_from_answer_row(row)
        if row_id:
            rows_by_id[row_id] = normalize_row(row)
    return rows_by_id


def predicted_rows_by_id(record: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    rows_by_id: Dict[str, Dict[str, str]] = {}
    leaf_outputs = ((record.get("planner_first_result") or {}).get("leaf_outputs") or [])
    for leaf_output in leaf_outputs:
        for item in parsed_leaf_items(leaf_output):
            row_id = item.get("row_id")
            values = item.get("values")
            if not isinstance(row_id, str) or not row_id:
                row_id = row_id_from_answer_row(values)
            if row_id:
                rows_by_id[row_id] = normalize_row(values)
    return rows_by_id


def parse_row_ids_from_why_prov(why_prov: Iterable[Any]) -> Set[str]:
    row_ids: Set[str] = set()
    for item in why_prov:
        if not isinstance(item, str):
            continue
        row_ids.update(re.findall(r"[A-Za-z][A-Za-z0-9]*_\d+", item))
    return row_ids


def expected_row_ids(ground_truth: Dict[str, Any]) -> Set[str]:
    rows = ground_truth.get("answer") or []
    ids = {rid for rid in (row_id_from_answer_row(row) for row in rows) if rid}
    if ids:
        return ids
    return parse_row_ids_from_why_prov(ground_truth.get("why_prov") or [])


def predicted_row_ids(record: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    leaf_outputs = ((record.get("planner_first_result") or {}).get("leaf_outputs") or [])
    for leaf_output in leaf_outputs:
        for item in parsed_leaf_items(leaf_output):
            row_id = item.get("row_id")
            if isinstance(row_id, str) and row_id:
                ids.add(row_id)
                continue
            row_id = row_id_from_answer_row(item.get("values"))
            if row_id:
                ids.add(row_id)
    return ids


def compare_answer_content(
    expected_rows: Dict[str, Dict[str, str]],
    predicted_rows: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    hallucinated_row_ids = sorted(set(predicted_rows) - set(expected_rows))
    missing_row_ids = sorted(set(expected_rows) - set(predicted_rows))
    value_mismatches = []
    matched_row_ids = sorted(set(expected_rows) & set(predicted_rows))

    for row_id in matched_row_ids:
        expected = expected_rows[row_id]
        predicted = predicted_rows[row_id]
        wrong_columns = {}
        missing_columns = sorted(set(expected) - set(predicted))
        extra_columns = sorted(set(predicted) - set(expected))

        for column in sorted(set(expected) & set(predicted)):
            if expected[column] != predicted[column]:
                wrong_columns[column] = {
                    "expected": expected[column],
                    "predicted": predicted[column],
                }

        if wrong_columns or missing_columns or extra_columns:
            value_mismatches.append(
                {
                    "row_id": row_id,
                    "wrong_columns": wrong_columns,
                    "missing_columns": missing_columns,
                    "extra_columns": extra_columns,
                }
            )

    hallucinated_value_count = len(hallucinated_row_ids) + len(value_mismatches)
    correct_answer_row_count = len(matched_row_ids) - len(value_mismatches)
    content_tp = correct_answer_row_count
    content_fp = len(hallucinated_row_ids) + len(value_mismatches)
    content_fn = len(missing_row_ids) + len(value_mismatches)
    content_precision = safe_div(content_tp, content_tp + content_fp)
    content_recall = safe_div(content_tp, content_tp + content_fn)
    content_row_f1 = safe_div(2 * content_precision * content_recall, content_precision + content_recall)
    content_row_accuracy = safe_div(
        correct_answer_row_count,
        correct_answer_row_count + len(value_mismatches) + len(missing_row_ids) + len(hallucinated_row_ids),
    )
    return {
        "answer_exact_match": not hallucinated_row_ids and not missing_row_ids and not value_mismatches,
        "correct_answer_row_count": correct_answer_row_count,
        "content_tp": content_tp,
        "content_fp": content_fp,
        "content_fn": content_fn,
        "content_precision": content_precision,
        "content_recall": content_recall,
        "content_row_f1": content_row_f1,
        "content_row_accuracy": content_row_accuracy,
        "hallucinated_row_ids": hallucinated_row_ids,
        "hallucinated_row_count": len(hallucinated_row_ids),
        "missing_answer_row_ids": missing_row_ids,
        "missing_answer_row_count": len(missing_row_ids),
        "value_mismatches": value_mismatches,
        "value_mismatch_count": len(value_mismatches),
        "hallucinated_value_count": hallucinated_value_count,
    }


def make_leaf_key(query_id: str, leaf_index: Optional[int], table_name: str, question_sql: str) -> Tuple[str, str, str, str]:
    leaf_task_id = f"{query_id}_leaf{int(leaf_index) + 1:02d}" if leaf_index is not None else ""
    return (query_id, leaf_task_id, table_name.strip(), normalize_sql(question_sql))


def build_ground_truth_index(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for record in records:
        key = (
            str(record.get("query_id", "")),
            str(record.get("leaf_task_id", "")),
            str(record.get("table_name", "")).strip(),
            normalize_sql(record.get("question_sql")),
        )
        index[key] = record
    return index


def find_ground_truth(
    record: Dict[str, Any],
    gt_index: Dict[Tuple[str, str, str, str], Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    query_id = str(record.get("query_id", ""))
    leaf_index = record.get("leaf_index")
    table_name = str(record.get("leaf_table_name") or "")
    question_sql = str(record.get("leaf_question_sql") or record.get("sql_to_run") or "")
    key = make_leaf_key(query_id, leaf_index, table_name, question_sql)
    if key in gt_index:
        return gt_index[key]

    # Fallback for older/different record ids: match on query, table, and SQL.
    candidates = [
        gt
        for gt_key, gt in gt_index.items()
        if gt_key[0] == query_id and gt_key[2] == table_name.strip() and gt_key[3] == normalize_sql(question_sql)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def score_sets(expected: Set[str], predicted: Set[str]) -> Dict[str, Any]:
    tp = len(expected & predicted)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    row_accuracy = safe_div(tp, tp + fp + fn)
    return {
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "row_f1": f1,
        "row_accuracy": row_accuracy,
        "exact_match": expected == predicted,
        "missing_row_ids": sorted(expected - predicted),
        "extra_row_ids": sorted(predicted - expected),
    }


def leaf_validity(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    leaf_outputs = ((record.get("planner_first_result") or {}).get("leaf_outputs") or [])
    if not leaf_outputs:
        return False, "missing leaf_outputs"
    invalid_errors = []
    for leaf_output in leaf_outputs:
        if leaf_output.get("valid_leaf_output") is not True:
            invalid_errors.append(str(leaf_output.get("parse_error") or "invalid leaf output"))
    return (not invalid_errors, "; ".join(invalid_errors) if invalid_errors else None)


def evaluate(predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
    gt_index = build_ground_truth_index(ground_truth)
    details: List[Dict[str, Any]] = []
    aggregate = defaultdict(float)
    unmatched_predictions = 0

    for record in predictions:
        if record.get("run_mode") != "leaf":
            continue
        gt = find_ground_truth(record, gt_index)
        if gt is None:
            unmatched_predictions += 1
            continue

        expected = expected_row_ids(gt)
        predicted = predicted_row_ids(record)
        answer_content = compare_answer_content(expected_rows_by_id(gt), predicted_rows_by_id(record))
        score = score_sets(expected, predicted)
        valid, validation_error = leaf_validity(record)

        detail = {
            "record_id": record.get("record_id"),
            "query_id": record.get("query_id"),
            "leaf_task_id": gt.get("leaf_task_id"),
            "table_name": gt.get("table_name"),
            "ok": record.get("ok") is True,
            "valid_leaf_output": valid,
            "validation_error": validation_error,
            **score,
            **answer_content,
        }
        details.append(detail)

        for key in ("tp", "fp", "fn", "expected_count", "predicted_count"):
            aggregate[key] += score[key]
        aggregate["correct_answer_row_count"] += answer_content["correct_answer_row_count"]
        aggregate["hallucinated_row_count"] += answer_content["hallucinated_row_count"]
        aggregate["missing_answer_row_count"] += answer_content["missing_answer_row_count"]
        aggregate["value_mismatch_count"] += answer_content["value_mismatch_count"]
        for key in ("content_tp", "content_fp", "content_fn"):
            aggregate[key] += answer_content[key]

    n = len(details)
    micro_precision = safe_div(aggregate["tp"], aggregate["tp"] + aggregate["fp"])
    micro_recall = safe_div(aggregate["tp"], aggregate["tp"] + aggregate["fn"])
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    micro_row_accuracy = safe_div(aggregate["tp"], aggregate["tp"] + aggregate["fp"] + aggregate["fn"])
    micro_content_precision = safe_div(aggregate["content_tp"], aggregate["content_tp"] + aggregate["content_fp"])
    micro_content_recall = safe_div(aggregate["content_tp"], aggregate["content_tp"] + aggregate["content_fn"])
    micro_content_row_f1 = safe_div(
        2 * micro_content_precision * micro_content_recall,
        micro_content_precision + micro_content_recall,
    )
    content_row_accuracy = safe_div(
        aggregate["correct_answer_row_count"],
        aggregate["correct_answer_row_count"]
        + aggregate["value_mismatch_count"]
        + aggregate["missing_answer_row_count"]
        + aggregate["hallucinated_row_count"],
    )

    summary = {
        "prediction_records": len(predictions),
        "leaf_records_evaluated": n,
        "ground_truth_leaf_tasks": len(ground_truth),
        "unmatched_predictions": unmatched_predictions,
        "row_accuracy": micro_row_accuracy,
        "leaf_exact_match_rate": safe_div(sum(1 for item in details if item["exact_match"]), n),
        "ok_rate": safe_div(sum(1 for item in details if item["ok"]), n),
        "valid_leaf_output_rate": safe_div(sum(1 for item in details if item["valid_leaf_output"]), n),
        "content_row_accuracy": content_row_accuracy,
        "content_row_f1": micro_content_row_f1,
        "leaf_answer_exact_match_rate": safe_div(sum(1 for item in details if item["answer_exact_match"]), n),
        "hallucination_free_rate": safe_div(
            sum(
                1
                for item in details
                if item["hallucinated_row_count"] == 0 and item["value_mismatch_count"] == 0
            ),
            n,
        ),
        "hallucinated_row_count": int(aggregate["hallucinated_row_count"]),
        "missing_answer_row_count": int(aggregate["missing_answer_row_count"]),
        "value_mismatch_count": int(aggregate["value_mismatch_count"]),
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "row_f1": micro_f1,
            "row_accuracy": micro_row_accuracy,
            "tp": int(aggregate["tp"]),
            "fp": int(aggregate["fp"]),
            "fn": int(aggregate["fn"]),
            "expected_count": int(aggregate["expected_count"]),
            "predicted_count": int(aggregate["predicted_count"]),
        },
        "content_micro": {
            "precision": micro_content_precision,
            "recall": micro_content_recall,
            "f1": micro_content_row_f1,
            "row_f1": micro_content_row_f1,
            "tp": int(aggregate["content_tp"]),
            "fp": int(aggregate["content_fp"]),
            "fn": int(aggregate["content_fn"]),
        },
        "macro": {
            "precision": safe_div(sum(item["precision"] for item in details), n),
            "recall": safe_div(sum(item["recall"] for item in details), n),
            "f1": safe_div(sum(item["f1"] for item in details), n),
            "row_f1": safe_div(sum(item["row_f1"] for item in details), n),
            "row_accuracy": safe_div(sum(item["row_accuracy"] for item in details), n),
            "content_precision": safe_div(sum(item["content_precision"] for item in details), n),
            "content_recall": safe_div(sum(item["content_recall"] for item in details), n),
            "content_row_f1": safe_div(sum(item["content_row_f1"] for item in details), n),
        },
    }

    return {"summary": summary, "details": details}


def write_csv(path: Path, details: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "record_id",
        "query_id",
        "leaf_task_id",
        "table_name",
        "ok",
        "valid_leaf_output",
        "expected_count",
        "predicted_count",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "row_f1",
        "row_accuracy",
        "exact_match",
        "validation_error",
        "missing_row_ids",
        "extra_row_ids",
        "answer_exact_match",
        "correct_answer_row_count",
        "content_tp",
        "content_fp",
        "content_fn",
        "content_precision",
        "content_recall",
        "content_row_f1",
        "content_row_accuracy",
        "hallucinated_row_count",
        "missing_answer_row_count",
        "value_mismatch_count",
        "hallucinated_row_ids",
        "missing_answer_row_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in details:
            out = dict(row)
            out["missing_row_ids"] = " ".join(out.get("missing_row_ids") or [])
            out["extra_row_ids"] = " ".join(out.get("extra_row_ids") or [])
            out["hallucinated_row_ids"] = " ".join(out.get("hallucinated_row_ids") or [])
            out["missing_answer_row_ids"] = " ".join(out.get("missing_answer_row_ids") or [])
            writer.writerow({field: out.get(field) for field in fields})


def write_plots(path: Path, report: Dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plot generation requires matplotlib. Install it or omit --plots-dir.") from exc

    path.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    micro = summary["micro"]
    macro = summary["macro"]
    details = report["details"]

    metrics = {
        "precision": micro["precision"],
        "recall": micro["recall"],
        "row_f1": micro["row_f1"],
        "row_accuracy": micro["row_accuracy"],
        "content_row_f1": summary["content_row_f1"],
        "content_row_accuracy": summary["content_row_accuracy"],
        "hallucination_free": summary["hallucination_free_rate"],
    }
    plt.figure(figsize=(9, 4.8))
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 1)
    plt.ylabel("score")
    plt.title("Planner-first row metrics")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path / "summary_metrics.png", dpi=160)
    plt.close()

    counts = {"TP": micro["tp"], "FP": micro["fp"], "FN": micro["fn"]}
    plt.figure(figsize=(6, 4))
    plt.bar(counts.keys(), counts.values())
    plt.ylabel("rows")
    plt.title("Row-level confusion counts")
    plt.tight_layout()
    plt.savefig(path / "row_confusion_counts.png", dpi=160)
    plt.close()

    if details:
        leaf_labels = [str(item["leaf_task_id"]) for item in details]
        leaf_scores = [item["row_f1"] for item in details]
        width = max(9, min(24, len(details) * 0.35))
        plt.figure(figsize=(width, 4.8))
        plt.bar(leaf_labels, leaf_scores)
        plt.ylim(0, 1)
        plt.ylabel("row_f1")
        plt.title("Row F1 by leaf task")
        plt.xticks(rotation=70, ha="right")
        plt.tight_layout()
        plt.savefig(path / "row_f1_by_leaf.png", dpi=160)
        plt.close()

        content_leaf_scores = [item["content_row_f1"] for item in details]
        plt.figure(figsize=(width, 4.8))
        plt.bar(leaf_labels, content_leaf_scores)
        plt.ylim(0, 1)
        plt.ylabel("content_row_f1")
        plt.title("Content row F1 by leaf task")
        plt.xticks(rotation=70, ha="right")
        plt.tight_layout()
        plt.savefig(path / "content_row_f1_by_leaf.png", dpi=160)
        plt.close()

        by_query: Dict[str, List[float]] = defaultdict(list)
        for item in details:
            by_query[str(item["query_id"])].append(float(item["row_f1"]))
        query_labels = sorted(by_query)
        query_scores = [safe_div(sum(by_query[query_id]), len(by_query[query_id])) for query_id in query_labels]
        width = max(9, min(24, len(query_labels) * 0.45))
        plt.figure(figsize=(width, 4.8))
        plt.bar(query_labels, query_scores)
        plt.ylim(0, 1)
        plt.ylabel("mean row_f1")
        plt.title("Mean row F1 by query")
        plt.xticks(rotation=65, ha="right")
        plt.tight_layout()
        plt.savefig(path / "mean_row_f1_by_query.png", dpi=160)
        plt.close()


def print_report(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    micro = summary["micro"]
    content_micro = summary["content_micro"]
    macro = summary["macro"]
    print("Planner-first leaf evaluation")
    print(f"  prediction records:       {summary['prediction_records']}")
    print(f"  evaluated leaf records:   {summary['leaf_records_evaluated']}")
    print(f"  ground-truth leaf tasks:  {summary['ground_truth_leaf_tasks']}")
    print(f"  unmatched predictions:    {summary['unmatched_predictions']}")
    print()
    print("Set metrics over row ids")
    print(f"  row accuracy:             {summary['row_accuracy']:.4f}")
    print(f"  micro precision:          {micro['precision']:.4f}")
    print(f"  micro recall:             {micro['recall']:.4f}")
    print(f"  micro row_f1:             {micro['row_f1']:.4f}")
    print(f"  macro row accuracy:       {macro['row_accuracy']:.4f}")
    print(f"  macro precision:          {macro['precision']:.4f}")
    print(f"  macro recall:             {macro['recall']:.4f}")
    print(f"  macro row_f1:             {macro['row_f1']:.4f}")
    print(f"  leaf exact-match rate:    {summary['leaf_exact_match_rate']:.4f}")
    print()
    print("Run quality")
    print(f"  ok rate:                  {summary['ok_rate']:.4f}")
    print(f"  valid leaf-output rate:   {summary['valid_leaf_output_rate']:.4f}")
    print(f"  TP / FP / FN:             {micro['tp']} / {micro['fp']} / {micro['fn']}")
    print()
    print("Answer hallucination checks")
    print(f"  content row accuracy:     {summary['content_row_accuracy']:.4f}")
    print(f"  content micro precision:  {content_micro['precision']:.4f}")
    print(f"  content micro recall:     {content_micro['recall']:.4f}")
    print(f"  content micro row_f1:     {content_micro['row_f1']:.4f}")
    print(f"  hallucination-free rate:  {summary['hallucination_free_rate']:.4f}")
    print(f"  leaf exact-answer rate:   {summary['leaf_answer_exact_match_rate']:.4f}")
    print(f"  hallucinated row ids:     {summary['hallucinated_row_count']}")
    print(f"  rows with wrong values:   {summary['value_mismatch_count']}")
    print(f"  missing answer rows:      {summary['missing_answer_row_count']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate planner-first leaf outputs against ProvSQL ground truth leaf tasks."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Directory where PNG plots are written.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Do not write PNG plots.")
    parser.add_argument("--no-write", action="store_true", help="Only print metrics; do not write report files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = load_json_or_jsonl(args.predictions)
    ground_truth = load_json_or_jsonl(args.ground_truth)
    report = evaluate(predictions, ground_truth)
    print_report(report)

    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print()
        print(f"Wrote JSON report: {args.output}")
        if args.csv_output:
            write_csv(args.csv_output, report["details"])
            print(f"Wrote CSV details: {args.csv_output}")
        if not args.no_plots:
            try:
                write_plots(args.plots_dir, report)
                print(f"Wrote plots: {args.plots_dir}")
            except RuntimeError as exc:
                print(f"Skipped plots: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
