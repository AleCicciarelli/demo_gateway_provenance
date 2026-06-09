# complete pipeline for planner-first explanation, including csv generation and calling the explanation service, and rendering the final markdown response

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from json_to_csv import clean_bucket, planner_result_to_csv_files
from explanation_client import ExplanationClient


def render_explanation_markdown(
    sql_query: str,
    generated_csv_files: List[str],
    explanation_output: Any,
) -> str:
    markdown = "### Full Pipeline Result\n\n"

    markdown += "#### Query\n\n"
    markdown += f"```sql\n{sql_query}\n```\n\n"

    markdown += "#### Generated CSV files\n\n"
    if generated_csv_files:
        for file in generated_csv_files:
            markdown += f"- `{file}`\n"
    else:
        markdown += "No CSV files generated.\n"

    markdown += "\n#### Explanation Service Output\n\n"

    if isinstance(explanation_output, dict):
        markdown += "```json\n"
        markdown += json.dumps(explanation_output, ensure_ascii=False, indent=2)
        markdown += "\n```\n"
    else:
        markdown += str(explanation_output)

    return markdown


def run_planner_first_explanation_pipeline(
    sql_query: str,
    planner_result: Dict[str, Any],
    bucket_dir: Path,
    explanation_client: ExplanationClient,
    delimiter: str = ",",
    keep_rownum: bool = False,
) -> Dict[str, Any]:
    clean_bucket(bucket_dir)

    generated_csv_files = planner_result_to_csv_files(
        planner_result=planner_result,
        output_dir=bucket_dir,
        delimiter=delimiter,
        keep_rownum=keep_rownum,
    )

    explanation_output = explanation_client.run_explanation(
        sql_query=sql_query,
        csv_files=generated_csv_files,
        delimiter=delimiter,
    )

    response_text = render_explanation_markdown(
        sql_query=sql_query,
        generated_csv_files=generated_csv_files,
        explanation_output=explanation_output,
    )

    return {
        "generated_csv_files": generated_csv_files,
        "explanation_output": explanation_output,
        "response_text": response_text,
    }