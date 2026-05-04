from typing import Dict, Any
import json

def build_leaf_prompt(task: Dict[str, Any], ctx: Dict[str, Any], mode: str = "first") -> str:
    table = task["table_name"]
    #columns = task["columns"]
    #preds = task.get("local_predicates", [])

    if mode == "more":
        return (
            "Return ONLY a valid JSON array with additional rows in the exact same schema as before.\n"
            "Do not repeat rows already returned.\n"
            "Each output item MUST include row_id and values.\n"
            "Each row_id MUST exist in CONTEXT_DATA.\n"
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

    Return ALL rows from exactly this table:

    TARGET_TABLE:
    {table}

    You must read only:

    CONTEXT_DATA["{table}"]

    Ignore every other table in CONTEXT_DATA.

    If CONTEXT_DATA does not contain the key "{table}", return [].

    --------------------------------------------------

    OUTPUT RULES:

    Return ONLY valid JSON.

    The output must be a JSON array.

    For every row inside CONTEXT_DATA["{table}"], output exactly one object:

    {{
      "row_id": "<the row_id key>",
      "values": <the full row object copied exactly>
    }}

    The "values" object must contain the FULL row.
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
    4. "row_id" must be the dictionary key of the row.
    5. "values" must be the complete row object.
    6. Copy strings exactly, including spaces.
    7. Return [] if the target table is missing.
    8. Return JSON only. No markdown. No comments. No text.

    --------------------------------------------------

    {few_shot}

    Now extract the rows.

    CONTEXT_DATA:
    {json.dumps(ctx, ensure_ascii=False)}
    """.strip()