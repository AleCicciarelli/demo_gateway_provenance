#!/usr/bin/env python3
"""Export Wikidata CSV rows using the existing semantic-join document contract.

Uses only the standard library so document preparation needs no GPU/ML install.
Source CSV values remain strings (empty fields become None).
"""
import argparse
import csv
import json
from pathlib import Path


def value_text(value):
    if value is None:
        return "NULL"
    text = str(value).strip()
    return text[:120] + "..." if len(text) > 120 else text


def prepare(dataset, output):
    schema = json.loads((dataset / "metadata/schema.json").read_text())
    paths = {p.stem: p for p in (dataset / "csv").glob("*.csv")}
    if set(paths) != set(schema):
        raise ValueError("CSV tables do not match metadata/schema.json")
    tables = {}
    targets = {}
    for table, spec in schema.items():
        with paths[table].open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"row_id", spec["primary_key"], *spec.get("foreign_keys", {})}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing required columns in {table}")
            rows = [{k: v if v != "" else None for k, v in row.items()} for row in reader]
        for column in {"row_id", spec["primary_key"]}:
            keys = [row[column] for row in rows]
            if None in keys or len(set(keys)) != len(keys):
                raise ValueError(f"Null or duplicate {table}.{column}")
        tables[table] = rows
        targets[table] = {row[spec["primary_key"]]: row for row in rows}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    counts = {}
    with temporary.open("w", encoding="utf-8") as handle:
        for table in sorted(tables):
            spec = schema[table]
            for row in tables[table]:
                links = []
                for column, reference in spec.get("foreign_keys", {}).items():
                    target_table, target_column = reference.split(".")
                    if target_column != schema[target_table]["primary_key"]:
                        raise ValueError(f"Foreign key does not reference a primary key: {reference}")
                    linked = targets[target_table].get(row[column])
                    if linked is None:
                        raise ValueError(f"Unresolved foreign key {table}.{column}={row[column]}")
                    display = [c for c in linked if any(p in c.lower() for p in (
                        "name", "title", "label", "status", "type", "category",
                        "segment", "region", "nation", "country", "date", "brand"
                    ))][:4]
                    links.append({
                        "relation": "foreign_key", "from_table": table,
                        "from_columns": [column], "from_values": [row[column]],
                        "to_table": target_table, "to_columns": [target_column],
                        "linked_values": {f"{target_table}.{c}": linked[c] for c in display},
                    })
                document = {
                    "page_content": "; ".join(
                        f"{column} is {value_text(value)}"
                        for column, value in row.items()
                        if column not in {"row_id", "statement_id"}
                    ),
                    "metadata": {
                        "doc_type": "row", "table": table,
                        "row_id": f"{table}:row_id={row['row_id']}",
                        "rid": row["row_id"], "rownum_column": "row_id",
                        "rownum_value": row["row_id"],
                        "primary_key": {spec["primary_key"]: row[spec["primary_key"]]},
                        "values": dict(row), "linked_rows": links,
                        "textualization_strategy": "semantic-join",
                    },
                }
                handle.write(json.dumps(document, ensure_ascii=False) + "\n")
            counts[table] = len(tables[table])
            print(f"{table}: {counts[table]:,} documents", flush=True)
    temporary.replace(output)
    print(f"Saved {sum(counts.values()):,} documents to {output}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("wikidata_movies"))
    parser.add_argument("--documents-out", type=Path, default=Path(
        "faiss_index_wikidata_movies_rows_bge_m3_semantic_join/row_documents_wikidata_movies.jsonl"
    ))
    args = parser.parse_args()
    prepare(args.dataset_dir, args.documents_out)
