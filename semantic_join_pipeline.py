"""CSV semantic bridges and conservative AST-based SQL rewriting for generated joins."""
from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from decimal import Decimal, InvalidOperation
import uuid
from pathlib import Path

import sqlglot
from sqlglot import exp

from semantic_matching import build_mapping
from semantic_block_matching import build_contextual_mapping
from row_probability import PROBABILITY_COLUMN

# Pair-level validation is reused for each edge of a multi-table join chain.
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


def identifier_fields(headers, table_schema=None):
    """Prefer declared keys; use bounded ID naming conventions for unknown fields."""
    schema = table_schema or {}
    primary = schema.get("primary_key") or []
    if isinstance(primary, str):
        primary = [primary]
    row_id = schema.get("row_id") or []
    if isinstance(row_id, str):
        row_id = [row_id]
    declared = {str(name).lower() for name in [*primary, *row_id, *(schema.get("foreign_keys") or {})]}
    known_columns = {str(name).lower() for name in schema.get("columns") or []}
    result = set()
    for name in headers:
        lower = name.lower()
        internal = lower in {"id", "__rid__", "__probability"} or lower.endswith("_rownum")
        fallback = (lower.endswith(("_id", "_key")) or bool(re.search(r"[a-z0-9](?:Id|ID)$", name)))
        if internal or lower in declared or (lower not in known_columns and fallback):
            result.add(name)
    return result


def descriptive_fields(headers, table_schema=None):
    identifiers = identifier_fields(headers, table_schema)
    return [name for name in headers if name not in identifiers]


def _query_matching_context(tree, edges, target_left, target_right, bucket, delimiter,
                            source_backed_columns, llm_tables, schema_info=None):
    """Prepare evidence lookups, never materialize or execute the query's joins.

    Only exact, source-backed single-equality edges can attach evidence. In
    particular, neither the target edge nor generated IDs may supply a link.
    """
    source_columns = source_backed_columns or {}
    endpoints = {target_left.name, target_right.name}
    adjacency, records, indexes, cache, references = {}, {}, {}, {}, {}
    relationships = []
    context_allowed = None
    for join, equality, left, right, _ in edges:
        if {left.name, right.name} == endpoints:
            continue
        if llm_tables is not None and {left.name, right.name} & set(llm_tables):
            continue
        # Residual conditions need actual SQL evaluation; do not claim a link
        # from just one equality of a more restrictive predicate.
        condition = join.args.get("on")
        while isinstance(condition, exp.Paren):
            condition = condition.this
        if condition != equality:
            continue
        aliases = {left.alias_or_name: left.name, right.alias_or_name: right.name}
        a, b = equality.this, equality.expression
        ta, tb = aliases[a.table], aliases[b.table]
        if (a.name not in source_columns.get(ta, [])
                or b.name not in source_columns.get(tb, [])):
            continue
        adjacency.setdefault(ta, []).append((a.name, tb, b.name))
        adjacency.setdefault(tb, []).append((b.name, ta, a.name))
        relationships.append(equality.sql())

    def load(table):
        if table not in records:
            with (Path(bucket) / (table + ".csv")).open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                headers = reader.fieldnames or []
                if len(headers) != len(set(headers)):
                    raise ValueError(f"Duplicate CSV columns: {table}")
                key = table + "_rownum"
                if key not in headers:
                    raise ValueError(f"Context table {table} requires {key}.")
                rows = list(reader)
                if any(None in row or any(v is None for v in row.values()) for row in rows):
                    raise ValueError(f"Malformed context CSV: {table}")
                if any(not row[key].strip() for row in rows) or len({row[key] for row in rows}) != len(rows):
                    raise ValueError(f"Context row identifiers must be nonempty and unique: {table}")
                records[table] = {row[key]: row for row in rows}
                references.update({(table, row[key]): f"context{len(references) + i}"
                                   for i, row in enumerate(rows)})
        return records[table]

    def linked(table, identity):
        key = (table, identity)
        if key in cache:
            return cache[key]
        pending, visited, evidence = [key], {key}, []
        while pending:
            current, rid = pending.pop()
            row = load(current)[rid]
            for column, other, other_column in adjacency.get(current, []):
                if other in endpoints:
                    continue
                index_key = (other, other_column)
                if index_key not in indexes:
                    index = {}
                    for other_id, other_row in load(other).items():
                        if other_column not in other_row:
                            raise ValueError(f"Missing context column {other}.{other_column}")
                        value = other_row[other_column]
                        if value != "":
                            index.setdefault(value, []).append(other_id)
                    indexes[index_key] = index
                if column not in row:
                    raise ValueError(f"Missing context column {current}.{column}")
                for other_id in indexes[index_key].get(row[column], []):
                    if context_allowed is not None and other in context_allowed and other_id not in context_allowed[other]:
                        continue
                    address = (other, other_id)
                    if address in visited:
                        continue
                    visited.add(address)
                    other_row = load(other)[other_id]
                    evidence.append({"ref": references[address],
                        "linked_from": "candidate" if (current, rid) == key else references[(current, rid)],
                        "relationship": f"{current}.{column} = {other}.{other_column}",
                        "table": other, "attributes": {
                        field: other_row[field] for field in descriptive_fields(
                            list(other_row), (schema_info or {}).get(other))}})
                    pending.append(address)
        cache[key] = evidence
        return evidence

    def context(batch):
        candidates = {}
        if isinstance(batch, dict):
            # Direct block mode visits each row once, never creates pair objects.
            addresses = [(ref, table, row["id"])
                         for side, table in (("left", target_left.name), ("right", target_right.name))
                         for ref, row in batch[side].items()]
        else:
            addresses = [(pair.get(side + "_ref", pair["pair_id"] + ":" + side), table, pair[id_key])
                         for pair in batch
                         for side, table, id_key in (("left", target_left.name, "idA"),
                                                    ("right", target_right.name, "idB"))]
        for ref, table, identity in addresses:
            if ref not in candidates:
                evidence = linked(table, identity) if table in adjacency else []
                if evidence:
                    candidates[ref] = evidence
        if not candidates:
            return {}
        return {"target_predicate": next(e.sql() for _, e, l, r, _ in edges
                    if l.name == target_left.name and r.name == target_right.name),
                "source_relationships": relationships,
                "query_filters": tree.args["where"].sql() if tree.args.get("where") else None,
                "candidate_context": candidates}

    def filter_candidates(left_rows, right_rows):
        """Propagate necessary source equalities without constructing joined rows."""
        nonlocal context_allowed
        reachable, pending = set(), [target_left.name, target_right.name]
        while pending:
            table = pending.pop()
            if table in reachable or table not in adjacency:
                continue
            reachable.add(table)
            pending.extend(other for _, other, _ in adjacency[table])
        allowed = {table: set(load(table)) for table in reachable}
        for table, rows in ((target_left.name, left_rows), (target_right.name, right_rows)):
            if table in allowed:
                allowed[table].intersection_update(row["id"] for row in rows)

        def key(value):
            if value is None or value == "":
                return None
            # Conservative over-inclusion for numeric spellings such as 001/1.
            # This never removes an exact text match; final SQL remains authoritative.
            try:
                number = Decimal(str(value))
                if number.is_finite():
                    return ("numeric", number)
            except InvalidOperation:
                pass
            return ("text", str(value))

        changed = True
        while changed:
            changed = False
            for table in reachable:
                for column, other, other_column in adjacency[table]:
                    other_rows = load(other)
                    if any(other_column not in other_rows[rid] for rid in allowed[other]):
                        raise ValueError(f"Missing context column {other}.{other_column}")
                    keys = {key(other_rows[rid][other_column]) for rid in allowed[other]}
                    keys.discard(None)
                    rows = load(table)
                    if any(column not in rows[rid] for rid in allowed[table]):
                        raise ValueError(f"Missing context column {table}.{column}")
                    retained = {rid for rid in allowed[table] if key(rows[rid][column]) in keys}
                    if retained != allowed[table]:
                        allowed[table] = retained
                        changed = True
        selected = [[row for row in rows if table not in allowed or row["id"] in allowed[table]]
                    for table, rows in ((target_left.name, left_rows), (target_right.name, right_rows))]
        context_allowed = allowed
        cache.clear()
        return (*selected, {"left_before": len(left_rows), "left_after": len(selected[0]),
                            "right_before": len(right_rows), "right_after": len(selected[1]),
                            "source_relationships": relationships})

    context.filter_candidates = filter_candidates
    return context


def join_match_definition(left, right, equality, schema_info=None):
    """Describe edge meaning from schema metadata, never from stored source rows."""
    aliases = {table.alias_or_name: table.name for table in (left, right)}
    columns = (equality.this, equality.expression)
    descriptions = []
    kind = "association"
    for source, target in (columns, columns[::-1]):
        table = aliases[source.table]
        target_name = aliases[target.table] + "." + target.name
        spec = ((schema_info or {}).get(table, {}).get("relationship_descriptions") or {}).get(source.name)
        if not isinstance(spec, dict) or spec.get("target") != target_name:
            continue
        description = spec.get("description")
        if not isinstance(description, str) or not description.strip():
            continue
        if spec.get("kind") not in ("association", "identity"):
            raise ValueError("Relationship description needs kind association or identity")
        descriptions.append((spec["kind"], description.strip()))
    if len(set(descriptions)) > 1:
        raise ValueError("Conflicting schema relationship descriptions for " + equality.sql())
    if descriptions:
        kind, question = descriptions[0]
    else:
        question = (
            f"Determine which {right.name} record each {left.name} record refers to through "
            f"{equality.sql()}. Infer and name the specific factual relationship from the table "
            "and attribute meanings before assessing it. Use year/date/location when relevant."
        )
    return (
        f"Relationship type: {kind}. Left records: {left.name}. Right records: {right.name}. "
        + question + " "
        + ("Assess entity identity only, not other factual relationships. " if kind == "identity" else
           "Assess this specific relationship, not whether the records are the same entity. ")
        + "Use supplied descriptions and relevant remembered knowledge; label model knowledge as unverified. "
        "Missing attributes are unknown, not contradictions. "
        + ("" if kind == "identity" else
           "Assess the association at the level of detail supplied by the records. "
           "Use world knowledge to associate geographic or category attributes; a country-only candidate "
           "does not require a venue name when the event can be associated with that country. ")
        + 
        "Do not compare synthetic IDs. Return uncertain when the relationship cannot be settled."
    )


def _prepare_semantic_pair(sql_query, csv_files, bucket_dir, model_call, *, delimiter=",", event=None,
                           source_backed_columns=None, max_output_tokens=None, source_context=None,
                           matching_context=None, context_tokens=None, schema_info=None, publication_dir=None):
    """Publish the bridge only after a complete matching run. Never overwrite sources."""
    tree, left, right, join, replaced = inspect_semantic_join(sql_query)
    bucket = Path(bucket_dir)
    # Both sides must preserve real source identifiers. Coincidental overlap with
    # synthetic LLM IDs is not evidence that an ordinary equality is meaningful.
    aliases = {table.alias_or_name: table.name for table in (left, right)}
    source_columns = source_backed_columns or {}
    join_columns = (replaced.this, replaced.expression)
    comparable_ids = all(
        column.name.lower() != PROBABILITY_COLUMN
        and column.name in identifier_fields([column.name], (schema_info or {}).get(aliases[column.table]))
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
        if not descriptive_fields(headers[table.name], (schema_info or {}).get(table.name)) and matching_context is None:
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
             "left_fields": descriptive_fields(headers[left.name], (schema_info or {}).get(left.name)),
             "right_fields": descriptive_fields(headers[right.name], (schema_info or {}).get(right.name)),
             "replaced_predicate": replaced.sql(), "mapping_file": mapping_table + ".csv", "mapping_table": mapping_table,
             "rownum_column": rownum_column, "left_table": left.name, "right_table": right.name}
    def send(kind):
        if event:
            event(kind, {"semantic_matching": json.loads(json.dumps(trace))})
    def report_progress(progress):
        trace.update(progress)
        send("matching_progress")

    request_metadata = {}

    def request_started(metadata):
        request_metadata.clear()
        request_metadata.update(metadata)

    def call(prompt):
        entry = {"prompt": prompt, **request_metadata}
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
        # Context, when supplied for a multi-join query, uses only other
        # source-backed query edges; never resolve the target relationship.
        if delimiter != ",":
            normalized = bucket / ("semantic_inputs_" + token)
            normalized.mkdir()
            for index, path in enumerate(inputs):
                target = normalized / path.name
                with path.open(newline="", encoding="utf-8-sig") as src, target.open("w", newline="", encoding="utf-8") as dst:
                    csv.writer(dst).writerows(csv.reader(src, delimiter=delimiter))
                inputs[index] = target
        definition = join_match_definition(left, right, replaced, schema_info)
        matcher = build_contextual_mapping if matching_context is not None else build_mapping
        audit = matcher(*inputs, left.name + "_rownum", right.name + "_rownum",
                              trace["left_fields"], trace["right_fields"], definition, run, call,
                              cardinality="many_to_many", progress=report_progress,
                              max_output_tokens=max_output_tokens, context_tokens=context_tokens,
                              request_started=request_started,
                              **({"matching_context": matching_context,
                                  "candidate_filter": matching_context.filter_candidates}
                                 if matching_context is not None else {}))
        target = Path(publication_dir or bucket) / trace["mapping_file"]
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


def _publish_semantic_mappings(staging, bucket, filenames):
    """Publish complete files without overwriting others; undo this run on error."""
    published = []
    try:
        for filename in filenames:
            source, target = Path(staging) / filename, Path(bucket) / filename
            # Staging is in the bucket's filesystem. A hard link publishes a
            # complete file atomically and fails if a destination already exists.
            os.link(source, target)
            published.append((source, target))
    except Exception:
        for source, target in reversed(published):
            if target.exists() and source.samefile(target):
                target.unlink()
        raise


def prepare_semantic_joins(sql_query, csv_files, bucket_dir, model_call, *, delimiter=",", event=None,
                           source_backed_columns=None, max_output_tokens=None, source_context=None,
                           llm_tables=None, validate_only=False, context_tokens=None, schema_info=None):
    """Rewrite supported inner-join chains, preserving joins backed by real IDs."""
    statements = sqlglot.parse(sql_query)
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError("Semantic rewriting requires one SELECT.")
    tree = statements[0]
    joins = tree.args.get("joins") or []
    source = tree.args.get("from_")
    if (tree.args.get("with_") or len(list(tree.find_all(exp.Select))) != 1
            or source is None or not isinstance(source.this, exp.Table)):
        raise ValueError("Semantic rewriting requires base tables; CTEs/subqueries are unsupported.")
    if not joins:
        return {"sql_query": sql_query, "csv_files": list(csv_files), "mapping_columns": [],
                "trace": {"status": "skipped", "reason": "No joins to rewrite.",
                          "original_sql": sql_query, "rewritten_sql": sql_query, "calls": []}}
    tables = [source.this, *(join.this for join in joins)]
    if (not all(isinstance(table, exp.Table) for table in tables)
            or len(list(tree.find_all(exp.Table))) != len(tables)
            or len({table.name for table in tables}) != len(tables)
            or len({table.alias_or_name for table in tables}) != len(tables)):
        raise ValueError("Semantic rewriting requires distinct base tables and aliases.")
    # Keep the existing two-table trace format for compatibility.
    if len(tables) == 2 and (llm_tables is None or any(t.name in llm_tables for t in tables)):
        if validate_only:
            inspect_semantic_join(sql_query)
            return {"validated": True}
        return _prepare_semantic_pair(sql_query, csv_files, bucket_dir, model_call,
            delimiter=delimiter, event=event, source_backed_columns=source_backed_columns,
            max_output_tokens=max_output_tokens, source_context=source_context,
            context_tokens=context_tokens, schema_info=schema_info)
    if any(not c.table and c.name.lower() in {"ida", "idb"} for c in tree.find_all(exp.Column)):
        raise ValueError("Qualify idA/idB columns before semantic rewriting.")
    aliases = {tables[0].alias_or_name: tables[0]}
    edges = []
    # Validate every edge before making model requests or publishing mappings.
    for join in joins:
        right = join.this
        equalities = [node for node in (join.args.get("on") or exp.Null()).find_all(exp.EQ)
                      if isinstance(node.this, exp.Column) and isinstance(node.expression, exp.Column)
                      and right.alias_or_name in {node.this.table, node.expression.table}
                      and any(alias in aliases for alias in (node.this.table, node.expression.table))]
        if len(equalities) != 1:
            raise ValueError("Each semantic join requires one qualified equality to an earlier table.")
        equality = equalities[0]
        left_alias = next(alias for alias in (equality.this.table, equality.expression.table) if alias in aliases)
        left = aliases[left_alias]
        if any(column.table and column.table not in {*aliases, right.alias_or_name}
               for column in join.args["on"].find_all(exp.Column)):
            raise ValueError("A join predicate references a table that is not yet available.")
        pair = exp.Select(expressions=[exp.Star()]).from_(left.copy())
        pair.set("joins", [join.copy()])
        inspect_semantic_join(pair.sql())
        # Residual predicates referring to other earlier tables are preserved in
        # the full query, but not passed to the isolated pair matcher.
        pair.args["joins"][0].set("on", equality.copy())
        mapping = left.name + "_" + right.name
        if mapping + ".csv" in csv_files or mapping in {t.alias_or_name for t in tables}:
            raise ValueError(f"Semantic mapping conflicts with source table or alias: {mapping}")
        edges.append((join, equality, left, right, pair))
        aliases[right.alias_or_name] = right
    projections = []
    for item in tree.expressions:
        if isinstance(item, exp.Star):
            if any(item.args.values()):
                raise ValueError("Modified wildcard projections are unsupported in semantic mode.")
            projections.extend(exp.Column(this=exp.Star(), table=exp.to_identifier(t.alias_or_name)) for t in tables)
        else:
            projections.append(item)
    tree.set("expressions", projections)
    if validate_only:
        return {"validated": True}
    traces, rewritten_joins, files, mapping_columns = [], [], list(csv_files), {}
    eligible = [edge for edge in edges if llm_tables is None
                or edge[2].name in llm_tables or edge[3].name in llm_tables]
    published = False
    active_trace = None

    def aggregate(status="running", current=None):
        items = [*traces, current] if current is not None else traces
        completed = len(traces) + int(current is not None and current["status"] in ("complete", "skipped"))
        snapshot = {"status": status, "original_sql": sql_query, "rewritten_sql": tree.sql(),
                "joins": items, "calls": [call for trace in items for call in trace.get("calls", [])],
                "accepted_count": sum(trace.get("accepted_count", 0) for trace in items),
                "mapping_files": list(mapping_columns) if published else [],
                "edge_count": len(eligible), "completed_edges": completed,
                "publication_status": "published" if published else "staged",
                "reason": "All eligible joins preserve source identifiers." if status == "skipped" else ""}
        if current is not None:
            counters = {key: current[key] for key in ("total_requests", "completed_requests", "planned_requests",
                        "left_row_count", "right_row_count") if key in current}
            snapshot.update(counters)
            snapshot.update(progress_scope="edge", current_edge={"index": len(traces) + 1,
                "left_table": current["left_table"], "right_table": current["right_table"],
                "status": current["status"], **counters})
        else:
            snapshot.update(progress_scope="query",
                total_requests=sum(trace.get("total_requests", 0) for trace in items),
                completed_requests=sum(trace.get("completed_requests", 0) for trace in items),
                planned_requests=sum(trace.get("planned_requests", 0) for trace in items))
        return snapshot

    # Capture the original query relationships before any equality is rewritten.
    contexts = {id(join): _query_matching_context(tree, edges, left, right, bucket_dir,
                    delimiter, source_backed_columns, llm_tables, schema_info)
                for join, _, left, right, _ in edges} if len(joins) >= 2 else {}
    with tempfile.TemporaryDirectory(prefix="semantic_staging_", dir=bucket_dir) as staging:
        try:
            for join, equality, left, right, pair in edges:
                if llm_tables is not None and left.name not in llm_tables and right.name not in llm_tables:
                    rewritten_joins.append(join)
                    continue

                def relay(kind, payload):
                    nonlocal active_trace
                    current = payload["semantic_matching"]
                    active_trace = current
                    snapshot = aggregate("failed" if current["status"] == "failed" else "running", current)
                    if current.get("error"):
                        snapshot["error"] = current["error"]
                    if event:
                        # Completing an edge does not complete the query or publish files.
                        forwarded = "matching_progress" if kind in ("matching_done", "matching_skipped") else kind
                        event(forwarded, {"semantic_matching": snapshot})

                prepared = _prepare_semantic_pair(pair.sql(), files, bucket_dir, model_call,
                    delimiter=delimiter, event=relay, source_backed_columns=source_backed_columns,
                    max_output_tokens=max_output_tokens, source_context=source_context,
                    matching_context=contexts.get(id(join)), context_tokens=context_tokens,
                    schema_info=schema_info, publication_dir=staging)
                trace = prepared["trace"]
                traces.append(trace)
                active_trace = None
                files = prepared["csv_files"]
                if trace["status"] == "skipped":
                    rewritten_joins.append(join)
                    continue
                pair_tree = sqlglot.parse_one(prepared["sql_query"])
                bridge, mapped_join = pair_tree.args["joins"]
                equality.replace(mapped_join.args["on"].copy())
                rewritten_joins.extend([bridge, join])
                mapping_columns[trace["mapping_file"]] = prepared["mapping_columns"]
            tree.set("joins", rewritten_joins)
            _publish_semantic_mappings(staging, bucket_dir, mapping_columns)
            published = True
        except Exception as exc:
            if event:
                failed = aggregate("failed", active_trace)
                failed.update(error=str(exc), publication_status="rolled_back")
                event("matching_failed", {"semantic_matching": failed})
            raise
    trace = aggregate("complete" if mapping_columns else "skipped")
    if event:
        event("matching_done" if mapping_columns else "matching_skipped", {"semantic_matching": trace})
    return {"sql_query": tree.sql(), "csv_files": files, "trace": trace,
            "mapping_columns": [], "mapping_columns_by_file": mapping_columns}


def validate_semantic_query(sql_query):
    """Validate query shape before extraction without reading data or invoking a model."""
    prepare_semantic_joins(sql_query, [], "", None, validate_only=True)
