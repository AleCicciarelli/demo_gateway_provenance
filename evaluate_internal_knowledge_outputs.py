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
DEFAULT_PREDICTIONS = REPO_ROOT / "evaluation" / "internal_knowledge_outputs.jsonl"
DEFAULT_GROUND_TRUTH = REPO_ROOT / "evaluation" / "ground_truth_queries.json"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "internal_knowledge_metrics.json"
DEFAULT_CSV = REPO_ROOT / "evaluation" / "internal_knowledge_metrics.csv"
DEFAULT_CSV_DIR = REPO_ROOT / "tpch_no_provsql"
DEFAULT_PLOTS_DIR = REPO_ROOT / "evaluation" / "internal_knowledge_plots"


PRIMARY_KEY_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "region": ("r_regionkey",),
    "nation": ("n_nationkey",),
    "supplier": ("s_suppkey",),
    "customer": ("c_custkey",),
    "orders": ("o_orderkey",),
    "lineitem": ("l_orderkey", "l_linenumber"),
    "part": ("p_partkey",),
    "partsupp": ("ps_partkey", "ps_suppkey"),
}


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


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value).strip()
    return str(value).strip()


def normalize_identifier(identifier: str) -> str:
    return re.sub(r"\s+", "", identifier.strip())


def normalize_sql(sql: Optional[str]) -> str:
    if not sql:
        return ""
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def load_rownum_to_semantic_id(csv_dir: Path) -> Dict[str, str]:
    """Map local CSV row-number ids to semantic primary-key ids.

    Internal-knowledge prompts forbid local row ids because those ids are
    instance metadata. Evaluation therefore converts ground-truth ids such as
    customer_123 to comparable ids such as customer_4567 using primary keys.
    """
    mapping: Dict[str, str] = {}
    for table, pk_columns in PRIMARY_KEY_COLUMNS.items():
        path = csv_dir / f"{table}.csv"
        if not path.exists():
            continue
        rownum_col = f"{table}_rownum"
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                local_id = normalize_identifier(str(row.get(rownum_col, "")))
                if not local_id:
                    continue
                pk_values = [normalize_identifier(str(row.get(col, ""))) for col in pk_columns]
                if any(value == "" for value in pk_values):
                    continue
                mapping[local_id] = f"{table}_{'_'.join(pk_values)}"
    return mapping


def parse_ground_truth_provenance(
    why_prov: Iterable[Any],
    rownum_to_semantic_id: Dict[str, str],
) -> Set[str]:
    ids: Set[str] = set()
    for item in why_prov:
        if not isinstance(item, str):
            continue
        for identifier in re.findall(r"[A-Za-z][A-Za-z0-9]*_\d+(?:_\d+)?", item):
            clean = normalize_identifier(identifier)
            ids.add(rownum_to_semantic_id.get(clean, clean))
    return ids


def parse_prediction_provenance(items: Iterable[Dict[str, Any]]) -> Set[str]:
    ids: Set[str] = set()
    for item in items:
        provenance = item.get("provenance")
        if not isinstance(provenance, list):
            continue
        for witness in provenance:
            if not isinstance(witness, list):
                continue
            for identifier in witness:
                if isinstance(identifier, str):
                    ids.add(normalize_identifier(identifier))
    return ids


def row_id_from_answer_row(row: Any) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    for key in ("__rid__", "row_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return normalize_identifier(value)
    for key, value in row.items():
        if key.endswith("_rownum") and isinstance(value, str) and value:
            return normalize_identifier(value)
    return None


def expected_leaf_provenance_ids(
    ground_truth: Dict[str, Any],
    rownum_to_semantic_id: Dict[str, str],
) -> Set[str]:
    ids: Set[str] = set()
    for row in ground_truth.get("answer") or []:
        row_id = row_id_from_answer_row(row)
        if row_id:
            ids.add(rownum_to_semantic_id.get(row_id, row_id))
    if ids:
        return ids
    return parse_ground_truth_provenance(ground_truth.get("why_prov") or [], rownum_to_semantic_id)


def expected_answers(ground_truth: Dict[str, Any]) -> Set[str]:
    return {normalize_value(value) for value in ground_truth.get("answer") or []}


def predicted_answers(record: Dict[str, Any]) -> Set[str]:
    answers: Set[str] = set()
    parsed = record.get("parsed_output")
    if not isinstance(parsed, list):
        parsed = []

    for item in parsed:
        if not isinstance(item, dict):
            continue
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        if len(result) == 1:
            answers.add(normalize_value(next(iter(result.values()))))
        elif result:
            answers.add(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return answers


def parsed_prediction_items(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    parsed = record.get("parsed_output")
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]

    output_text = record.get("output_text")
    if not isinstance(output_text, str):
        return []

    decoder = json.JSONDecoder()
    for start, char in enumerate(output_text):
        if char != "[":
            continue
        try:
            obj, _ = decoder.raw_decode(output_text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return [item for item in obj if isinstance(item, dict)]
    return []


def score_sets(expected: Set[str], predicted: Set[str]) -> Dict[str, Any]:
    tp = len(expected & predicted)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "exact_match": expected == predicted,
        "missing": sorted(expected - predicted),
        "extra": sorted(predicted - expected),
    }


def build_ground_truth_index(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(record.get("query_id", "")): record for record in records}


def make_leaf_key(
    query_id: str,
    leaf_index: Optional[int],
    table_name: str,
    question_sql: str,
) -> Tuple[str, str, str, str]:
    leaf_task_id = f"{query_id}_leaf{int(leaf_index) + 1:02d}" if leaf_index is not None else ""
    return (query_id, leaf_task_id, table_name.strip(), normalize_sql(question_sql))


def build_leaf_ground_truth_index(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
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


def find_leaf_ground_truth(
    record: Dict[str, Any],
    gt_index: Dict[Tuple[str, str, str, str], Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    query_id = str(record.get("query_id", ""))
    leaf_index = record.get("leaf_index")
    table_name = str(record.get("leaf_table_name") or "")
    question_sql = str(record.get("leaf_question_sql") or "")
    key = make_leaf_key(query_id, leaf_index, table_name, question_sql)
    if key in gt_index:
        return gt_index[key]

    candidates = [
        gt
        for gt_key, gt in gt_index.items()
        if gt_key[0] == query_id and gt_key[2] == table_name.strip() and gt_key[3] == normalize_sql(question_sql)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def evaluate(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    csv_dir: Path,
) -> Dict[str, Any]:
    gt_index = build_ground_truth_index(ground_truth)
    rownum_to_semantic_id = load_rownum_to_semantic_id(csv_dir)
    details: List[Dict[str, Any]] = []
    aggregate = defaultdict(float)
    unmatched_predictions = 0

    for record in predictions:
        if record.get("run_mode") != "internal-knowledge":
            continue

        query_id = str(record.get("query_id", ""))
        gt = gt_index.get(query_id)
        if gt is None:
            unmatched_predictions += 1
            continue

        parsed_items = parsed_prediction_items(record)
        expected_answer_set = expected_answers(gt)
        predicted_answer_set = predicted_answers({**record, "parsed_output": parsed_items})
        expected_prov_set = parse_ground_truth_provenance(gt.get("why_prov") or [], rownum_to_semantic_id)
        predicted_prov_set = parse_prediction_provenance(parsed_items)

        answer_score = score_sets(expected_answer_set, predicted_answer_set)
        provenance_score = score_sets(expected_prov_set, predicted_prov_set)

        detail = {
            "record_id": record.get("record_id"),
            "query_id": query_id,
            "ok": record.get("ok") is True,
            "valid_output": record.get("ok") is True and record.get("parse_error") is None,
            "parse_error": record.get("parse_error"),
            "answer_expected_count": answer_score["expected_count"],
            "answer_predicted_count": answer_score["predicted_count"],
            "answer_tp": answer_score["tp"],
            "answer_fp": answer_score["fp"],
            "answer_fn": answer_score["fn"],
            "answer_precision": answer_score["precision"],
            "answer_recall": answer_score["recall"],
            "answer_f1": answer_score["f1"],
            "answer_accuracy": answer_score["accuracy"],
            "answer_exact_match": answer_score["exact_match"],
            "answer_missing": answer_score["missing"],
            "answer_extra": answer_score["extra"],
            "provenance_expected_count": provenance_score["expected_count"],
            "provenance_predicted_count": provenance_score["predicted_count"],
            "provenance_tp": provenance_score["tp"],
            "provenance_fp": provenance_score["fp"],
            "provenance_fn": provenance_score["fn"],
            "provenance_precision": provenance_score["precision"],
            "provenance_recall": provenance_score["recall"],
            "provenance_f1": provenance_score["f1"],
            "provenance_accuracy": provenance_score["accuracy"],
            "provenance_exact_match": provenance_score["exact_match"],
            "provenance_missing": provenance_score["missing"],
            "provenance_extra": provenance_score["extra"],
        }
        details.append(detail)

        for prefix, score in (("answer", answer_score), ("provenance", provenance_score)):
            for key in ("tp", "fp", "fn", "expected_count", "predicted_count"):
                aggregate[f"{prefix}_{key}"] += score[key]

    n = len(details)

    def micro(prefix: str) -> Dict[str, Any]:
        tp = aggregate[f"{prefix}_tp"]
        fp = aggregate[f"{prefix}_fp"]
        fn = aggregate[f"{prefix}_fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = tp / (tp + fp + fn) if tp + fp + fn else 1.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "expected_count": int(aggregate[f"{prefix}_expected_count"]),
            "predicted_count": int(aggregate[f"{prefix}_predicted_count"]),
        }

    summary = {
        "prediction_records": len(predictions),
        "query_records_evaluated": n,
        "ground_truth_queries": len(ground_truth),
        "unmatched_predictions": unmatched_predictions,
        "ok_rate": sum(1 for item in details if item["ok"]) / n if n else 0.0,
        "valid_output_rate": sum(1 for item in details if item["valid_output"]) / n if n else 0.0,
        "answer_exact_match_rate": sum(1 for item in details if item["answer_exact_match"]) / n if n else 0.0,
        "provenance_exact_match_rate": (
            sum(1 for item in details if item["provenance_exact_match"]) / n if n else 0.0
        ),
        "answer_micro": micro("answer"),
        "provenance_micro": micro("provenance"),
        "answer_macro": {
            "precision": sum(item["answer_precision"] for item in details) / n if n else 0.0,
            "recall": sum(item["answer_recall"] for item in details) / n if n else 0.0,
            "f1": sum(item["answer_f1"] for item in details) / n if n else 0.0,
            "accuracy": sum(item["answer_accuracy"] for item in details) / n if n else 0.0,
        },
        "provenance_macro": {
            "precision": sum(item["provenance_precision"] for item in details) / n if n else 0.0,
            "recall": sum(item["provenance_recall"] for item in details) / n if n else 0.0,
            "f1": sum(item["provenance_f1"] for item in details) / n if n else 0.0,
            "accuracy": sum(item["provenance_accuracy"] for item in details) / n if n else 0.0,
        },
    }

    return {"summary": summary, "details": details}


def evaluate_leaf(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    csv_dir: Path,
) -> Dict[str, Any]:
    gt_index = build_leaf_ground_truth_index(ground_truth)
    rownum_to_semantic_id = load_rownum_to_semantic_id(csv_dir)
    details: List[Dict[str, Any]] = []
    aggregate = defaultdict(float)
    unmatched_predictions = 0

    for record in predictions:
        if record.get("run_mode") != "internal-knowledge-leaf":
            continue

        gt = find_leaf_ground_truth(record, gt_index)
        if gt is None:
            unmatched_predictions += 1
            continue

        parsed_items = parsed_prediction_items(record)
        expected_set = expected_leaf_provenance_ids(gt, rownum_to_semantic_id)
        predicted_set = parse_prediction_provenance(parsed_items)
        score = score_sets(expected_set, predicted_set)

        detail = {
            "record_id": record.get("record_id"),
            "query_id": record.get("query_id"),
            "leaf_task_id": gt.get("leaf_task_id"),
            "table_name": gt.get("table_name"),
            "ok": record.get("ok") is True,
            "valid_leaf_output": record.get("ok") is True and record.get("parse_error") is None,
            "valid_output": record.get("ok") is True and record.get("parse_error") is None,
            "parse_error": record.get("parse_error"),
            "expected_count": score["expected_count"],
            "predicted_count": score["predicted_count"],
            "tp": score["tp"],
            "fp": score["fp"],
            "fn": score["fn"],
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1"],
            "row_f1": score["f1"],
            "row_accuracy": score["accuracy"],
            "exact_match": score["exact_match"],
            "missing_row_ids": score["missing"],
            "extra_row_ids": score["extra"],
            "answer_exact_match": score["exact_match"],
            "correct_answer_row_count": score["tp"],
            "content_tp": score["tp"],
            "content_fp": score["fp"],
            "content_fn": score["fn"],
            "content_precision": score["precision"],
            "content_recall": score["recall"],
            "content_row_f1": score["f1"],
            "content_row_accuracy": score["accuracy"],
            "hallucinated_row_count": score["fp"],
            "missing_answer_row_count": score["fn"],
            "value_mismatch_count": 0,
            "hallucinated_row_ids": score["extra"],
            "missing_answer_row_ids": score["missing"],
        }
        details.append(detail)

        for key in ("tp", "fp", "fn", "expected_count", "predicted_count"):
            aggregate[key] += score[key]
        aggregate["correct_answer_row_count"] += score["tp"]
        aggregate["hallucinated_row_count"] += score["fp"]
        aggregate["missing_answer_row_count"] += score["fn"]
        for key in ("content_tp", "content_fp", "content_fn"):
            aggregate[key] += detail[key]

    n = len(details)
    tp = aggregate["tp"]
    fp = aggregate["fp"]
    fn = aggregate["fn"]
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
    row_accuracy = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    content_tp = aggregate["content_tp"]
    content_fp = aggregate["content_fp"]
    content_fn = aggregate["content_fn"]
    content_precision = content_tp / (content_tp + content_fp) if content_tp + content_fp else 0.0
    content_recall = content_tp / (content_tp + content_fn) if content_tp + content_fn else 0.0
    content_row_f1 = (
        2 * content_precision * content_recall / (content_precision + content_recall)
        if content_precision + content_recall
        else 0.0
    )
    content_row_accuracy = (
        aggregate["correct_answer_row_count"]
        / (
            aggregate["correct_answer_row_count"]
            + aggregate["missing_answer_row_count"]
            + aggregate["hallucinated_row_count"]
        )
        if aggregate["correct_answer_row_count"] + aggregate["missing_answer_row_count"] + aggregate["hallucinated_row_count"]
        else 1.0
    )

    summary = {
        "prediction_records": len(predictions),
        "leaf_records_evaluated": n,
        "ground_truth_leaf_tasks": len(ground_truth),
        "unmatched_predictions": unmatched_predictions,
        "row_accuracy": row_accuracy,
        "ok_rate": sum(1 for item in details if item["ok"]) / n if n else 0.0,
        "valid_leaf_output_rate": sum(1 for item in details if item["valid_leaf_output"]) / n if n else 0.0,
        "valid_output_rate": sum(1 for item in details if item["valid_output"]) / n if n else 0.0,
        "leaf_exact_match_rate": sum(1 for item in details if item["exact_match"]) / n if n else 0.0,
        "content_row_accuracy": content_row_accuracy,
        "content_row_f1": content_row_f1,
        "leaf_answer_exact_match_rate": sum(1 for item in details if item["answer_exact_match"]) / n if n else 0.0,
        "hallucination_free_rate": (
            sum(1 for item in details if item["hallucinated_row_count"] == 0 and item["value_mismatch_count"] == 0) / n
            if n
            else 0.0
        ),
        "hallucinated_row_count": int(aggregate["hallucinated_row_count"]),
        "missing_answer_row_count": int(aggregate["missing_answer_row_count"]),
        "value_mismatch_count": 0,
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "row_f1": micro_f1,
            "row_accuracy": row_accuracy,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "expected_count": int(aggregate["expected_count"]),
            "predicted_count": int(aggregate["predicted_count"]),
        },
        "content_micro": {
            "precision": content_precision,
            "recall": content_recall,
            "f1": content_row_f1,
            "row_f1": content_row_f1,
            "tp": int(content_tp),
            "fp": int(content_fp),
            "fn": int(content_fn),
        },
        "macro": {
            "precision": sum(item["precision"] for item in details) / n if n else 0.0,
            "recall": sum(item["recall"] for item in details) / n if n else 0.0,
            "f1": sum(item["f1"] for item in details) / n if n else 0.0,
            "row_f1": sum(item["row_f1"] for item in details) / n if n else 0.0,
            "row_accuracy": sum(item["row_accuracy"] for item in details) / n if n else 0.0,
            "content_precision": sum(item["content_precision"] for item in details) / n if n else 0.0,
            "content_recall": sum(item["content_recall"] for item in details) / n if n else 0.0,
            "content_row_f1": sum(item["content_row_f1"] for item in details) / n if n else 0.0,
        },
    }

    return {"summary": summary, "details": details}


def write_csv(path: Path, details: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if details and "leaf_task_id" in details[0]:
        fields = [
            "record_id",
            "query_id",
            "leaf_task_id",
            "table_name",
            "ok",
            "valid_leaf_output",
            "parse_error",
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
            "missing_row_ids",
            "extra_row_ids",
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
        return

    fields = [
        "record_id",
        "query_id",
        "ok",
        "valid_output",
        "parse_error",
        "answer_expected_count",
        "answer_predicted_count",
        "answer_tp",
        "answer_fp",
        "answer_fn",
        "answer_precision",
        "answer_recall",
        "answer_f1",
        "answer_accuracy",
        "answer_exact_match",
        "answer_missing",
        "answer_extra",
        "provenance_expected_count",
        "provenance_predicted_count",
        "provenance_tp",
        "provenance_fp",
        "provenance_fn",
        "provenance_precision",
        "provenance_recall",
        "provenance_f1",
        "provenance_accuracy",
        "provenance_exact_match",
        "provenance_missing",
        "provenance_extra",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in details:
            out = dict(row)
            for key in ("answer_missing", "answer_extra", "provenance_missing", "provenance_extra"):
                out[key] = " ".join(out.get(key) or [])
            writer.writerow({field: out.get(field) for field in fields})


def write_plots(path: Path, report: Dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plot generation requires matplotlib. Install it or omit --plots-dir.") from exc

    path.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    if "leaf_records_evaluated" in summary:
        micro = summary["micro"]
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
        plt.title("Internal-knowledge row metrics")
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
            labels = [str(item["leaf_task_id"]) for item in details]
            width = max(9, min(24, len(details) * 0.35))
            plt.figure(figsize=(width, 4.8))
            plt.bar(labels, [item["row_f1"] for item in details])
            plt.ylim(0, 1)
            plt.ylabel("row_f1")
            plt.title("Row F1 by leaf task")
            plt.xticks(rotation=70, ha="right")
            plt.tight_layout()
            plt.savefig(path / "row_f1_by_leaf.png", dpi=160)
            plt.close()

            plt.figure(figsize=(width, 4.8))
            plt.bar(labels, [item["content_row_f1"] for item in details])
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
            query_scores = [
                sum(by_query[query_id]) / len(by_query[query_id])
                if by_query[query_id]
                else 0.0
                for query_id in query_labels
            ]
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
        return

    answer = summary["answer_micro"]
    provenance = summary["provenance_micro"]
    details = report["details"]

    metrics = {
        "answer_precision": answer["precision"],
        "answer_recall": answer["recall"],
        "answer_f1": answer["f1"],
        "answer_exact": summary["answer_exact_match_rate"],
        "prov_precision": provenance["precision"],
        "prov_recall": provenance["recall"],
        "prov_f1": provenance["f1"],
        "prov_exact": summary["provenance_exact_match_rate"],
    }
    plt.figure(figsize=(10, 4.8))
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 1)
    plt.ylabel("score")
    plt.title("Internal-knowledge query metrics")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path / "summary_metrics.png", dpi=160)
    plt.close()

    answer_counts = {"TP": answer["tp"], "FP": answer["fp"], "FN": answer["fn"]}
    plt.figure(figsize=(6, 4))
    plt.bar(answer_counts.keys(), answer_counts.values())
    plt.ylabel("answer values")
    plt.title("Answer confusion counts")
    plt.tight_layout()
    plt.savefig(path / "answer_confusion_counts.png", dpi=160)
    plt.close()

    provenance_counts = {"TP": provenance["tp"], "FP": provenance["fp"], "FN": provenance["fn"]}
    plt.figure(figsize=(6, 4))
    plt.bar(provenance_counts.keys(), provenance_counts.values())
    plt.ylabel("provenance identifiers")
    plt.title("Semantic provenance confusion counts")
    plt.tight_layout()
    plt.savefig(path / "provenance_confusion_counts.png", dpi=160)
    plt.close()

    if details:
        labels = [str(item["query_id"]) for item in details]
        width = max(9, min(24, len(details) * 0.45))

        plt.figure(figsize=(width, 4.8))
        plt.bar(labels, [item["answer_f1"] for item in details])
        plt.ylim(0, 1)
        plt.ylabel("answer_f1")
        plt.title("Answer F1 by query")
        plt.xticks(rotation=65, ha="right")
        plt.tight_layout()
        plt.savefig(path / "answer_f1_by_query.png", dpi=160)
        plt.close()

        plt.figure(figsize=(width, 4.8))
        plt.bar(labels, [item["provenance_f1"] for item in details])
        plt.ylim(0, 1)
        plt.ylabel("provenance_f1")
        plt.title("Semantic provenance F1 by query")
        plt.xticks(rotation=65, ha="right")
        plt.tight_layout()
        plt.savefig(path / "provenance_f1_by_query.png", dpi=160)
        plt.close()


def print_report(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    if "leaf_records_evaluated" in summary:
        micro = summary["micro"]
        content_micro = summary["content_micro"]
        macro = summary["macro"]
        print("Internal-knowledge leaf evaluation")
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
        return

    answer = summary["answer_micro"]
    provenance = summary["provenance_micro"]
    print("Internal-knowledge query evaluation")
    print(f"  prediction records:          {summary['prediction_records']}")
    print(f"  evaluated query records:     {summary['query_records_evaluated']}")
    print(f"  ground-truth queries:        {summary['ground_truth_queries']}")
    print(f"  unmatched predictions:       {summary['unmatched_predictions']}")
    print()
    print("Run quality")
    print(f"  ok rate:                     {summary['ok_rate']:.4f}")
    print(f"  valid output rate:           {summary['valid_output_rate']:.4f}")
    print()
    print("Answer value metrics")
    print(f"  exact-match rate:            {summary['answer_exact_match_rate']:.4f}")
    print(f"  micro precision:             {answer['precision']:.4f}")
    print(f"  micro recall:                {answer['recall']:.4f}")
    print(f"  micro f1:                    {answer['f1']:.4f}")
    print(f"  TP / FP / FN:                {answer['tp']} / {answer['fp']} / {answer['fn']}")
    print()
    print("Semantic provenance metrics")
    print(f"  exact-match rate:            {summary['provenance_exact_match_rate']:.4f}")
    print(f"  micro precision:             {provenance['precision']:.4f}")
    print(f"  micro recall:                {provenance['recall']:.4f}")
    print(f"  micro f1:                    {provenance['f1']:.4f}")
    print(f"  TP / FP / FN:                {provenance['tp']} / {provenance['fp']} / {provenance['fn']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate internal-knowledge outputs against query-level or leaf-task ProvSQL ground truth."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--mode",
        choices=("query", "leaf", "auto"),
        default="auto",
        help="query evaluates run_mode=internal-knowledge; leaf evaluates run_mode=internal-knowledge-leaf.",
    )
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
    mode = args.mode
    if mode == "auto":
        mode = "leaf" if any(record.get("run_mode") == "internal-knowledge-leaf" for record in predictions) else "query"
    report = (
        evaluate_leaf(predictions, ground_truth, args.csv_dir)
        if mode == "leaf"
        else evaluate(predictions, ground_truth, args.csv_dir)
    )
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
