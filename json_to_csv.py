# conversion of planner result json to csv files, one per table, to save in the bucket folder for the AP-explanation service

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List


def _normalize_csv_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def clean_bucket(bucket_dir: Path) -> None:
    bucket_dir.mkdir(parents=True, exist_ok=True)

    for item in bucket_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def planner_result_to_csv_files(
    planner_result: Dict[str, Any],
    output_dir: Path,
    delimiter: str = ",",
    keep_rownum: bool = True,
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_row_ids: Dict[str, set[str]] = defaultdict(set)

    for leaf in planner_result.get("leaf_outputs", []):
        table_name = leaf.get("table_name")
        if not table_name:
            continue

        parsed_output = leaf.get("parsed_output") or []

        for item in parsed_output:
            if not isinstance(item, dict):
                continue

            row_id = item.get("row_id")
            values = item.get("values")

            if not isinstance(values, dict):
                continue

            if row_id and row_id in seen_row_ids[table_name]:
                continue

            row = {
                column: _normalize_csv_value(value)
                for column, value in values.items()
            }

            row.pop("__rid__", None)

            if not keep_rownum:
                row.pop(f"{table_name}_rownum", None)

            tables[table_name].append(row)

            if row_id:
                seen_row_ids[table_name].add(row_id)

    generated_files: List[str] = []

    for table_name, rows in tables.items():
        if not rows:
            continue

        csv_path = output_dir / f"{table_name}.csv"

        columns: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in columns:
                    columns.append(key)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=columns,
                delimiter=delimiter,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

        generated_files.append(csv_path.name)

    return generated_files
