from typing import Dict, Any
import json

def build_leaf_prompt(task: Dict[str, Any], ctx: Dict[str, Any], mode: str = "first") -> str:
    table = task["table_name"]
    id_col = f"{table}_rownum"
    context_json = json.dumps(ctx, ensure_ascii=False, indent=2)
    #columns = task["columns"]
    #preds = task.get("local_predicates", [])

    if mode == "more":
        return (
            "Return ONLY a valid JSON array with additional rows in the exact same schema as before.\n"
            "Do not repeat rows already returned.\n"
            "Each output item MUST include row_id and values.\n"
            "row_id is an output wrapper field, not an input column.\n"
            "Each row_id MUST be a row dictionary key from CONTEXT_DATA.\n"
            "Copy values exactly from the row identified by row_id.\n"
            "If there are no more valid rows, return [].\n"
            "No explanations. No markdown. No SQL."
        )
  # no conditions for now, but they can be added later if needed
  #  columns_block = "\n".join(f"- {c}" for c in columns)

  #  if preds:
  #      filter_lines = ["Rows MUST satisfy ALL these filters exactly:"]
  #      for p in preds:
  #          filter_lines.append(f"- {p}")
  #      filter_block = "\n".join(filter_lines)
  #  else:
  #      filter_block = "There are no extra filters."

  #  schema_example = ", ".join([f'"{c}": "<value>"' for c in columns])
  
    # few shot prompt with 2 examples, one with a valid row and one without any valid rows
    few_shot = """
    EXAMPLES:

    Example 1:

    TARGET_TABLE:
    nation

    CONTEXT_DATA:
    {
      "nation": {
        "nation_2": {
          "nation_rownum": "nation_2",
          "n_name": "ARGENTINA",
          "n_nationkey": "1",
          "__rid__": "nation_2"
        }
      },
      "supplier": {
        "supplier_42": {
          "s_name": "Supplier#000000042",
          "s_nationkey": "22",
          "__rid__": "supplier_42"
        }
      }
    }

    CORRECT_JSON_OUTPUT:
    [
      {
        "row_id": "nation_2",
        "values": {
          "nation_rownum": "nation_2",
          "n_name": "ARGENTINA",
          "n_nationkey": "1",
          "__rid__": "nation_2"
        }
      }
    ]

    Example 2:

    TARGET_TABLE:
    nation

    CONTEXT_DATA:
    {
      "supplier": {
        "supplier_42": {
          "s_name": "Supplier#000000042",
          "s_nationkey": "22",
          "__rid__": "supplier_42"
        }
      }
    }

    CORRECT_JSON_OUTPUT:
    []
    """

    return f"""
    You are a JSON extraction engine.

    Your task is simple:

    Return candidate source rows from exactly this table:

    TARGET_TABLE:
    {table}

    You must read only:

    CONTEXT_DATA["{table}"]

    Ignore every other table in CONTEXT_DATA.

    If CONTEXT_DATA does not contain the key "{table}", return [].

    Do NOT evaluate SQL joins.
    Do NOT apply WHERE filters or pushed predicates.
    Do NOT aggregate, group, sort, limit, project, or compute final query results.
    A deterministic component will handle those operations later.

    --------------------------------------------------

    OUTPUT RULES:

    Return ONLY valid JSON.

    The output must be a JSON array.

    For every row inside CONTEXT_DATA["{table}"], output exactly one object:

    {{
      "row_id": "<the row dictionary key, e.g. {table}_123>",
      "values": <the full row object copied exactly>
    }}

    "row_id" is an output wrapper field.
    It may not exist as a column inside the row.
    Its value must equal the CONTEXT_DATA dictionary key for that row.
    For this dataset, that same identifier also appears inside the row as "{id_col}" and "__rid__".

    For example, if CONTEXT_DATA["{table}"] contains this entry:

    "{table}_22": {{
      "{id_col}": "{table}_22",
      "some_column": "some value",
      "__rid__": "{table}_22"
    }}

    then the output item must be:

    {{
      "row_id": "{table}_22",
      "values": {{
        "{id_col}": "{table}_22",
        "some_column": "some value",
        "__rid__": "{table}_22"
      }}
    }}

    The "values" object must contain the FULL row.
    The top-level object must NOT be the row itself.
    The only top-level keys are "row_id" and "values".
    Do not remove columns.
    Do not select columns.
    Do not rename columns.
    Do not change values.
    Do not trim spaces.
    Do not change order intentionally.
    Do not add explanations.

    --------------------------------------------------

    CRITICAL RULES:

    1. Use only rows under CONTEXT_DATA["{table}"].
    2. Ignore all other tables.
    3. Each output item corresponds to exactly one input row.
    4. "row_id" must be the dictionary key of the row, such as "{table}_123".
    5. "values" must be the complete row object.
    6. Copy strings exactly, including spaces.
    7. Return [] if the target table is missing.
    8. Return JSON only. No markdown. No comments. No text.
    9. Do not invent rows, identifiers, columns, or values.

    --------------------------------------------------

    {few_shot}

    Now extract the rows.

    CONTEXT_DATA:
{context_json}
    """.strip()


def build_iterative_join_leaf_prompt(
    task: Dict[str, Any],
    ctx: Dict[str, Any],
    inherited_bindings: Dict[str, Any] | None = None,
    source_row_ids: list[str] | None = None,
) -> str:
    table = task["table_name"]
    id_col = f"{table}_rownum"
    context_json = json.dumps(ctx, ensure_ascii=False, indent=2)
    predicates = [
        str(predicate).strip()
        for predicate in task.get("local_predicates") or []
        if str(predicate).strip()
    ]
    bindings = inherited_bindings or {}
    sources = source_row_ids or []

    constraints = []
    if predicates:
        constraints.append("LOCAL_PREDICATES:")
        constraints.extend(f"- {predicate}" for predicate in predicates)
    if bindings:
        constraints.append("INHERITED_JOIN_BINDINGS:")
        for column, values in bindings.items():
            if isinstance(values, list):
                values_text = ", ".join(str(value) for value in values)
            else:
                values_text = str(values)
            constraints.append(f"- {column}: {values_text}")
    if sources:
        constraints.append("SOURCE_ROWS_THAT_PRODUCED_BINDINGS:")
        constraints.append("- " + ", ".join(sources))

    constraints_block = "\n".join(constraints) if constraints else "No local predicates or inherited bindings."

    return f"""
You are a JSON extraction engine for one iterative join step.

TARGET_TABLE:
{table}

CONTEXT_DATA may contain several tables.
You may inspect every table in CONTEXT_DATA to understand links, source rows, and nearby evidence.
However, your output rows must come only from the TARGET_TABLE.
If CONTEXT_DATA does not contain "{table}", return [].

Your job is to select candidate source rows from the target table that are useful for this leaf step.
Use the constraints below, the inherited bindings, and the other retrieved tables as evidence to choose rows from the target table.

CONSTRAINTS:
{constraints_block}

Selection rules:
- Prefer rows that satisfy all LOCAL_PREDICATES.
- Prefer rows whose columns match the INHERITED_JOIN_BINDINGS.
- If a listed inherited binding column is present in a row, the row should match one of the listed values.
- Use rows from other tables only as evidence for finding the right target-table rows.
- It is valid and useful to compare target-table rows with linked/source rows from other tables.
- If no row clearly satisfies the constraints, return [].
- Do not compute the final SQL answer.
- Do not execute joins, aggregation, grouping, ordering, limit, or projection.
- Do not invent rows outside CONTEXT_DATA.
- Do not output rows from non-target tables.

Output rules:
- Return ONLY valid JSON.
- The output must be a JSON array.
- Each output item must have exactly "row_id" and "values".
- "row_id" must be a row dictionary key from CONTEXT_DATA["{table}"], such as "{table}_123".
- "values" must be the full row object copied exactly.
- Copy strings exactly, including spaces.
- Do not remove, rename, trim, or modify columns.
- Do not add explanations, markdown, comments, or extra keys.

Example item:
{{
  "row_id": "{table}_22",
  "values": {{
    "{id_col}": "{table}_22",
    "some_column": "some value",
    "__rid__": "{table}_22"
  }}
}}

Now select matching candidate rows.

CONTEXT_DATA:
{context_json}
    """.strip()
