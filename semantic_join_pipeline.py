"""Optional CSV semantic bridge and conservative AST-based SQL rewriting."""
from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

import sqlglot
from sqlglot import exp

from semantic_matching import build_mapping
from row_probability import PROBABILITY_COLUMN

# currently, semantic matching is only supported for a single SELECT with two base tables and one inner join.
def inspect_semantic_join(sql: str):
    statements = sqlglot.parse(sql)
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError("Semantic matching supports one SELECT with two tables and one inner join.")
    tree = statements[0]
    joins = tree.args.get("joins") or []
    source = tree.args.get("from_")
    if (tree.args.get("with_") or len(list(tree.find_all(exp.Select))) != 1
            or source is None or not isinstance(source.this, exp.Table) or len(joins) != 1):
        raise ValueError("Semantic matching requires two base tables and one inner join; CTEs/subqueries are unsupported.")
    join = joins[0]
    left, right = source.this, join.this
    if (not isinstance(right, exp.Table) or len(list(tree.find_all(exp.Table))) != 2
            or left.name == right.name or left.alias_or_name == right.alias_or_name
            or join.args.get("side") or join.args.get("method") or join.args.get("using")
            or join.args.get("kind", "").upper() not in ("", "INNER")):
        raise ValueError("Semantic matching currently supports distinct tables joined by INNER JOIN ON only.")
    predicates = []
    def flatten(node):
        if isinstance(node, exp.Paren):
            flatten(node.this)
        elif isinstance(node, exp.And):
            flatten(node.this)
            flatten(node.expression)
        else:
            predicates.append(node)
    flatten(join.args.get("on"))
    candidates = [p for p in predicates if isinstance(p, exp.EQ)
                  and isinstance(p.this, exp.Column) and isinstance(p.expression, exp.Column)
                  and {p.this.table, p.expression.table} == {left.alias_or_name, right.alias_or_name}]
    if len(candidates) != 1 or any(p is None or p.find(exp.Or) for p in predicates):
        raise ValueError("Semantic matching requires exactly one qualified cross-table equality; OR/composite joins are unsupported.")
    return tree, left, right, join, candidates[0]


def descriptive_fields(headers):
    return [name for name in headers if name.lower() not in {"id", "__rid__", "__probability"}
            and not name.lower().endswith(("id", "_rownum", "_key"))]


def prepare_semantic_joins(sql_query, csv_files, bucket_dir, model_call, *, delimiter=",", event=None,
                           source_backed_columns=None):
    """Publish the bridge only after a complete matching run. Never overwrite sources."""
    tree, left, right, join, replaced = inspect_semantic_join(sql_query)
    bucket = Path(bucket_dir)
    # Both sides must preserve real source identifiers. Coincidental overlap with
    # synthetic LLM IDs is not evidence that an ordinary equality is meaningful.
    aliases = {table.alias_or_name: table.name for table in (left, right)}
    source_columns = source_backed_columns or {}
    join_columns = (replaced.this, replaced.expression)
    comparable_ids = all(
        (column.name.lower() == "__rid__" or column.name.lower().endswith(("id", "_key", "_rownum")))
        and column.name in source_columns.get(aliases[column.table], [])
        for column in join_columns
    )
    if comparable_ids:
        for column in join_columns:
            filename = aliases[column.table] + ".csv"
            if filename not in csv_files:
                comparable_ids = False
                break
            with (bucket / filename).open(newline="", encoding="utf-8-sig") as handle:
                if column.name not in next(csv.reader(handle, delimiter=delimiter), []):
                    comparable_ids = False
                    break
    if comparable_ids:
        trace = {"status": "skipped", "reason": "Both join columns preserve identifiers from the same source dataset; using the original ID join.",
                 "original_sql": sql_query, "rewritten_sql": sql_query, "calls": [],
                 "left_table": left.name, "right_table": right.name,
                 "join_predicate": replaced.sql()}
        if event:
            event("matching_skipped", {"semantic_matching": trace})
        return {"sql_query": sql_query, "csv_files": list(csv_files), "trace": trace, "mapping_columns": []}
    headers = {}
    for table in (left, right):
        filename = table.name + ".csv"
        if filename not in csv_files:
            raise ValueError(f"Missing semantic matching input: {filename}")
        with (bucket / filename).open(newline="", encoding="utf-8-sig") as handle:
            headers[table.name] = next(csv.reader(handle, delimiter=delimiter), [])
        key = table.name + "_rownum"
        if key not in headers[table.name]:
            raise ValueError(f"Semantic matching requires the retained row identifier {key}.")
        if not descriptive_fields(headers[table.name]):
            raise ValueError(f"No descriptive matching fields in {filename}; include names or other context in extraction.")
    # Adding idA/idB must not change unqualified-name or projection-alias resolution.
    for column in tree.find_all(exp.Column):
        if not column.table and column.name.lower() in {"ida", "idb"}:
            raise ValueError(f"Qualify column {column.name} before semantic matching; it collides with a mapping column.")
    # SELECT * must not expose the bridge's columns. COUNT(*) stays untouched.
    projections = []
    for item in tree.expressions:
        if isinstance(item, exp.Star):
            if any(item.args.values()):
                raise ValueError("Modified wildcard projections are unsupported in semantic mode.")
            projections.extend(exp.Column(this=exp.Star(), table=exp.to_identifier(t.alias_or_name)) for t in (left, right))
        else:
            projections.append(item)
    tree.set("expressions", projections)
    token = uuid.uuid4().hex[:12]
    mapping_table = left.name + "_" + right.name
    if mapping_table + ".csv" in csv_files:
        raise ValueError(f"Semantic mapping table conflicts with a source CSV: {mapping_table}")
    rownum_column = mapping_table + "_rownum"
    if mapping_table in {left.alias_or_name, right.alias_or_name}:
        raise ValueError(f"Source alias conflicts with semantic mapping table: {mapping_table}")
    run = bucket / ("semantic_audit_" + token)
    trace = {"status": "running", "original_sql": sql_query, "calls": [],
             "left_fields": descriptive_fields(headers[left.name]),
             "right_fields": descriptive_fields(headers[right.name]),
             "replaced_predicate": replaced.sql(), "mapping_file": mapping_table + ".csv", "mapping_table": mapping_table,
             "rownum_column": rownum_column, "left_table": left.name, "right_table": right.name}
    def send(kind):
        if event:
            event(kind, {"semantic_matching": json.loads(json.dumps(trace))})
    def report_progress(progress):
        trace.update(progress)
        send("matching_progress")

    def call(prompt):
        entry = {"prompt": prompt}
        trace["calls"].append(entry)
        send("matching_prompt")
        response = model_call(prompt)
        entry["response"] = response
        send("matching_response")
        return response
    send("matching_start")
    try:
        # The matcher expects comma CSV; normalize copies when AP uses another delimiter.
        inputs = [bucket / (t.name + ".csv") for t in (left, right)]
        if delimiter != ",":
            normalized = bucket / ("semantic_inputs_" + token)
            normalized.mkdir()
            for index, path in enumerate(inputs):
                target = normalized / path.name
                with path.open(newline="", encoding="utf-8-sig") as src, target.open("w", newline="", encoding="utf-8") as dst:
                    csv.writer(dst).writerows(csv.reader(src, delimiter=delimiter))
                inputs[index] = target
        audit = build_mapping(*inputs, left.name + "_rownum", right.name + "_rownum",
                              trace["left_fields"], trace["right_fields"], None, run, call,
                              cardinality="many_to_many", progress=report_progress)
        target = bucket / trace["mapping_file"]
        with (run / "matches.csv").open(newline="", encoding="utf-8") as src, target.open("w", newline="", encoding="utf-8") as dst:
            reader = csv.DictReader(src)
            mapping_rows = [
                {**row, rownum_column: f"{mapping_table}_{index}"}
                for index, row in enumerate(reader, start=1)
            ]
            writer = csv.DictWriter(dst, fieldnames=["idA", "idB", rownum_column, PROBABILITY_COLUMN], delimiter=delimiter)
            writer.writeheader()
            writer.writerows(mapping_rows)
        # The selected equality is replaced. Other ON conditions, filters, grouping, ordering, and limits remain.
        # SELECT * becomes SELECT r.*, c.* so mapping columns do not leak into the answer.
        # PostgreSQL folds unquoted idA/idB to ida/idb; CSV headers retain case.
        replacement = exp.EQ(this=exp.column(right.name + "_rownum", table=right.alias_or_name),
                             expression=exp.column("idB", table=mapping_table, quoted=True))
        replaced.replace(replacement)
        bridge = exp.Join(this=exp.Table(this=exp.to_identifier(mapping_table)),
                          on=exp.EQ(this=exp.column(left.name + "_rownum", table=left.alias_or_name),
                                    expression=exp.column("idA", table=mapping_table, quoted=True)), kind="INNER")
        tree.set("joins", [bridge, join])
        trace.update(status="complete", rewritten_sql=tree.sql(), audit=audit,
                     mapping_csv=target.read_text(), mapping_rows=mapping_rows, accepted_count=audit["accepted_count"])
        (run / "rewrite.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        send("matching_done")
        return {"sql_query": trace["rewritten_sql"], "csv_files": [*csv_files, target.name],
                "trace": trace, "mapping_columns": ["idA", "idB", rownum_column]}
    except Exception as exc:
        trace.update(status="failed", error=str(exc))
        send("matching_failed")
        raise
