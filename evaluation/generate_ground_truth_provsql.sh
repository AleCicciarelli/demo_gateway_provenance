#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="leaf_node_questions.json"
OUTPUT_QUERIES_FILE="ground_truth_queries.json"
OUTPUT_LEAF_FILE="ground_truth_leaf_tasks.json"

python3 <<'PYTHON_SCRIPT'
import json
import re
import subprocess
import sys

INPUT_FILE = "leaf_node_questions.json"
OUTPUT_QUERIES_FILE = "ground_truth_queries.json"
OUTPUT_LEAF_FILE = "ground_truth_leaf_tasks.json"

# we launch this from a container where psql can connect to tpch:
PSQL_CMD = ["docker", "exec", "-i", "provsql-tpch", "psql", "-U", "postgres", "-d", "tpch"]
WHY_PROV_EXPR = "provsql.sr_why(provsql.provenance(), 'provmap') AS why_prov"


def remove_trailing_semicolon(sql: str) -> str:
    return sql.strip().rstrip(";")


def add_why_prov_to_sql(sql: str) -> str:
    sql = remove_trailing_semicolon(sql)

    pattern = r"^\s*SELECT\s+(DISTINCT\s+)?(.+?)\s+FROM\s+"
    match = re.search(pattern, sql, flags=re.IGNORECASE | re.DOTALL)

    if not match:
        raise ValueError(f"Cannot parse SQL query:\n{sql}")

    distinct_part = match.group(1) or ""
    projection = match.group(2).strip()
    _, end = match.span()

    return (
        f"SELECT {distinct_part}{projection}, "
        f"{WHY_PROV_EXPR} FROM "
        + sql[end:]
        + ";"
    )


def run_sql(sql: str):
    sql = remove_trailing_semicolon(sql)

    cmd = PSQL_CMD + [
        "-q",
        "-X",
        "--csv",
        "-c",
        sql
    ]

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        return {
            "__error__": True,
            "sql": sql,
            "stderr": result.stderr.strip()
        }

    output = result.stdout.strip()

    if not output:
        return []

    import csv
    import io

    reader = csv.DictReader(io.StringIO(output))
    rows = []

    for row in reader:
        rows.append(dict(row))

    return rows

def extract_answer_and_why(rows):
    answers = []
    why_provs = []

    for row in rows:
        row = dict(row)

        why = row.pop("why_prov", None)
        why_provs.append(why)

        # Remove internal ProvSQL column if psql returns it
        row.pop("provsql", None)

        if len(row) == 1:
            answers.append(next(iter(row.values())))
        else:
            answers.append(row)

    return answers, why_provs

def build_query_ground_truth(entry):
    original_sql = entry["question_sql"]
    provsql_sql = add_why_prov_to_sql(original_sql)

    rows = run_sql(provsql_sql)
    answer, why_prov = extract_answer_and_why(rows)

    return {
        "query_id": entry["query_id"],
        "question_nl": entry["question_nl"],
        "question_sql": original_sql,
        "provsql_sql": provsql_sql,
        "answer": answer,
        "why_prov": why_prov
    }


def build_leaf_ground_truth(entry, leaf_cache):
    outputs = []
    query_id = entry["query_id"]

    for index, leaf in enumerate(entry.get("leaf_tasks", []), start=1):
        original_sql = leaf["question_sql"]
        provsql_sql = add_why_prov_to_sql(original_sql)

        cache_key = original_sql.strip().rstrip(";")

        if cache_key not in leaf_cache:
            print(f"  Executing leaf SQL once: {cache_key}")
            rows = run_sql(provsql_sql)

            if isinstance(rows, dict) and rows.get("__error__"):
                leaf_cache[cache_key] = {
                    "answer": [],
                    "why_prov": [],
                    "error": rows["stderr"],
                    "provsql_sql": provsql_sql
                }
            else:
                answer, why_prov = extract_answer_and_why(rows)
                leaf_cache[cache_key] = {
                    "answer": answer,
                    "why_prov": why_prov,
                    "error": None,
                    "provsql_sql": provsql_sql
                }
        else:
            print(f"  Reusing cached leaf SQL: {cache_key}")

        cached = leaf_cache[cache_key]

        leaf_output = {
            "query_id": query_id,
            "leaf_task_id": f"{query_id}_leaf{index:02d}",
            "table_name": leaf.get("table_name"),
            "question_nl": leaf["question_nl"],
            "question_sql": original_sql,
            "provsql_sql": cached["provsql_sql"],
            "answer": cached["answer"],
            "why_prov": cached["why_prov"]
        }

        if cached.get("error"):
            leaf_output["error"] = cached["error"]

        outputs.append(leaf_output)

    return outputs


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        entries = json.load(f)

    ground_truth_queries = []
    ground_truth_leaf_tasks = []

    leaf_cache = {}

    for entry in entries:
        query_id = entry.get("query_id", "unknown")
        print(f"Processing {query_id}...")

        ground_truth_queries.append(build_query_ground_truth(entry))
        ground_truth_leaf_tasks.extend(build_leaf_ground_truth(entry, leaf_cache))

    with open(OUTPUT_QUERIES_FILE, "w", encoding="utf-8") as f:
        json.dump(ground_truth_queries, f, indent=2, ensure_ascii=False, default=str)

    with open(OUTPUT_LEAF_FILE, "w", encoding="utf-8") as f:
        json.dump(ground_truth_leaf_tasks, f, indent=2, ensure_ascii=False, default=str)

    print()
    print(f"Saved {OUTPUT_QUERIES_FILE}")
    print(f"Saved {OUTPUT_LEAF_FILE}")
    print(f"Executed {len(leaf_cache)} unique leaf SQL queries.")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT