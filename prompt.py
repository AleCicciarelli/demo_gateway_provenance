from typing import Dict, Any
import json

def build_leaf_prompt(task: Dict[str, Any], ctx: Dict[str, Any], mode: str = "first") -> str:
    table = task["table_name"]
    columns = task["columns"]
    preds = task.get("local_predicates", [])

    if mode == "more":
        return (
            "Return ONLY a valid JSON array with additional rows in the exact same schema as before.\n"
            "Do not repeat rows already returned.\n"
            "Use ONLY rows from the target table.\n"
            "If there are no more rows, return [].\n"
            "No explanations. No markdown. No SQL."
        )

    columns_block = "\n".join(f"- {c}" for c in columns)

    if preds:
        filter_lines = ["Rows MUST satisfy ALL these filters exactly:"]
        for p in preds:
            filter_lines.append(f"- {p}")
        filter_block = "\n".join(filter_lines)
    else:
        filter_block = "There are no extra filters."

    return f"""
You are executing a deterministic leaf scan over structured rows already present in CONTEXT_DATA.

IMPORTANT:
CONTEXT_DATA may contain rows from multiple tables.
However, for this task you MUST return rows ONLY from the target table "{table}".

Your job is NOT to explain anything.
Your job is NOT to write SQL.
Your job is NOT to summarize the context.
Your job is ONLY to extract matching rows from the target table.

TARGET TABLE:
{table}

REQUIRED OUTPUT:
Return ONLY a valid JSON array.
Each array element MUST have exactly this structure:

{{
  "row_id": "<table_name>_<row_number>",
  "values": {{
    {", ".join([f'"{c}": ...' for c in columns])}
  }}
}}

STRICT RULES:
- Output rows ONLY from table "{table}".
- Ignore rows from all other tables, even if they appear in CONTEXT_DATA.
- Use ONLY row_ids that already appear in CONTEXT_DATA.
- Do NOT invent row_ids.
- Do NOT output any text before or after the JSON array.
- Do NOT output markdown.
- Do NOT output SQL.
- Do NOT add extra keys.
- The "values" object must contain EXACTLY these columns and no others:

{columns_block}

FILTERS:
{filter_block}

Before returning a row, verify:
1. the row_id exists in CONTEXT_DATA
2. the row belongs to table "{table}"
3. the row satisfies ALL filters
4. the values come from that exact row
5. the values object contains exactly the requested columns

If no row matches, return [].

CONTEXT_DATA:
{json.dumps(ctx, ensure_ascii=False)}
""".strip()