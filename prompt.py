"""Prompts for RAG pipeline."""

import json
from typing import Any, Dict


_OUTPUT_RULES = """OUTPUT RULES:
- Return only a valid JSON array; no explanations, markdown, comments, or SQL.
- Each item must have exactly "row_id" and "values":
  {"row_id":"<table row dictionary key>","values":{"<source column>":"<exact source value>"}}
- Output rows only from CONTEXT_DATA[TARGET_TABLE]. Return [] if the target table is missing.
- Each item corresponds to one existing source row. "row_id" is its dictionary key.
- "values" must contain all REQUIRED_COLUMNS.
- Copy values exactly from the row identified by "row_id", including spaces and value types.
- Do not invent rows, identifiers, columns, or values.
"""


def required_leaf_columns(task: Dict[str, Any]) -> list[str]:
    """Collect required columns in order; planner columns already include predicates."""
    if task.get("all_columns"):
        return []
    columns: list[str] = []
    for key in (
        "columns",
        "select_columns",
        "join_keys",
        "group_by_columns",
        "aggregate_columns",
    ):
        for column in task.get(key) or []:
            name = str(column).strip()
            if name and name != "*" and name not in columns:
                columns.append(name)
    return columns


def leaf_projection_instruction(task: Dict[str, Any]) -> str:
    if task.get("all_columns"):
        return ""
    return "REQUIRED_COLUMNS:\n" + compact_json(required_leaf_columns(task))


def leaf_output_rules(task: Dict[str, Any]) -> str:
    if task.get("all_columns"):
        return _OUTPUT_RULES.replace('- "values" must contain all REQUIRED_COLUMNS.\n', '')
    return _OUTPUT_RULES


def compact_json(value: Any) -> str:
    """Serialize prompt data without formatting whitespace."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_leaf_prompt(
    task: Dict[str, Any], ctx: Dict[str, Any], mode: str = "first", *,
    pushdown: bool = False,
) -> str:
    """Extract target-table rows, applying local predicates only with pushdown."""
    if mode == "more":
        return """Return only a valid JSON array of additional rows using the same target table, rules, and schema as before.
Do not repeat row_ids already returned. Each item must have exactly "row_id" and "values".
"row_id" is the target-table row dictionary key copy values exactly from that row.
Return [] if there are no more valid rows. No explanations, markdown, or SQL."""

    selection = "- Return exactly one item for every target-table row, preserving source order."
    if pushdown:
        predicates = [
            str(predicate).strip()
            for predicate in task.get("local_predicates") or []
            if str(predicate).strip()
        ]
        selection = f"""LOCAL_PREDICATES:
{compact_json(predicates)}
- Select target-table rows matching all LOCAL_PREDICATES, preserving source order.
- If LOCAL_PREDICATES is empty, return every target-table row.
- If no rows match, return []."""

    return f"""You are a JSON extraction engine.

TARGET_TABLE:
{task["table_name"]}

{leaf_projection_instruction(task)}

SELECTION RULES:
- Read only CONTEXT_DATA[TARGET_TABLE]; ignore every other table.
{selection}
- Do not execute joins, aggregate, group, sort, limit, or compute the final SQL result.

{leaf_output_rules(task)}

CONTEXT_DATA:
{compact_json(ctx)}"""


def build_iterative_join_leaf_prompt(
    task: Dict[str, Any],
    ctx: Dict[str, Any],
    inherited_bindings: Dict[str, Any] | None = None,
    source_row_ids: list[str] | None = None,
) -> str:
    """Select target-table candidates using constraints"""
    constraints = {
        "local_predicates": [
            str(predicate).strip()
            for predicate in task.get("local_predicates") or []
            if str(predicate).strip()
        ],
        "inherited_bindings": inherited_bindings or {},
        "source_row_ids": source_row_ids or [],
    }

    return f"""You are a JSON extraction engine for one iterative join step.

TARGET_TABLE:
{task["table_name"]}

{leaf_projection_instruction(task)}

CONSTRAINTS:
{compact_json(constraints)}

SELECTION RULES:
- Select target-table candidate rows useful for this leaf step.
- You may inspect every context table and compare linked/source rows as supporting evidence.
- Prefer rows satisfying all local_predicates and matching inherited_bindings.
- If an inherited binding column is present in a row, it should match one of the supplied values.
- source_row_ids identify the rows that produced the inherited bindings.
- Empty constraints impose no additional filtering conditions.
- If no row clearly satisfies the constraints, return [].

{leaf_output_rules(task)}

CONTEXT_DATA:
{compact_json(ctx)}"""
