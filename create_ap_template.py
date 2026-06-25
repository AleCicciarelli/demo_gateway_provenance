# ap_template.py

from __future__ import annotations

import uuid
from typing import Any, Dict, List
import json


AP_SCHEMA = "https://datagems-dev.scayle.es/moma2/v1/schemas/ap/ap-common.schema.json"


def build_ap_csv_template(
    sql_query: str,
    csv_files: List[str],
    delimiter: str = ",",
) -> Dict[str, Any]:
    ap_id = str(uuid.uuid4())
    operator_id = str(uuid.uuid4())
    csv_set_id = str(uuid.uuid4())

    nodes = [
        {
            "id": ap_id,
            "labels": ["Analytical_Pattern"],
            "properties": {
                "description": "Analytical Pattern to query a CSV dataset using Provenance annotations",
                "name": "Query CSV Dataset AP",
                "process": "query",
                "publishedDate": "2025-06-30",
                "startTime": "10:00:00",
            },
        },
        {
            "id": operator_id,
            "labels": ["Operator", "Provenance_SQL_Operator"],
            "properties": {
                "description": "Query the CSV dataset with provenance explanations",
                "name": "Provenance Query Operator",
                "command": "query",
                "queryType": "SELECT",
                "publishedDate": "2025-06-30",
                "query": sql_query,
                "startTime": "10:00:00",
                "step": 1,
                "type": "SQL Query",
            },
        },
        {
            "id": csv_set_id,
            "labels": ["CSV_Set", "Data"],
            "properties": {
                "delimiter": delimiter,
            },
        },
    ]

    edges = [
        {
            "from": ap_id,
            "labels": ["consist_of"],
            "to": operator_id,
        },
        {
            "from": csv_set_id,
            "labels": ["input"],
            "to": operator_id,
        },
    ]

    for csv_file in csv_files:
        csv_id = str(uuid.uuid4())

        nodes.append(
            {
                "id": csv_id,
                "labels": ["CSV", "Data", "cr:FileObject"],
                "properties": {
                    "contentSize": "1000 B",
                    "contentUrl": f"s3:/{csv_file}",
                    "description": "",
                    "encodingFormat": "text/csv",
                    "name": csv_file,
                    "sha256": "",
                    "type": "cr:FileObject",
                },
            }
        )

        # Nota: uso la direzione come nel tuo template:
        # CSV_Set --containedIn--> CSV
        edges.append(
            {
                "from": csv_set_id,
                "labels": ["containedIn"],
                "to": csv_id,
            }
        )

    return {
        "$schema": AP_SCHEMA,
        "edges": edges,
        "nodes": nodes,
    }

sql_query = """
SELECT c.c_name, n.n_name FROM customer c JOIN nation n ON c.c_nationkey = n.n_nationkey LIMIT 5
""".strip()

csv_files = [
    "customer.csv",
    "nation.csv",
]

payload = build_ap_csv_template(
    sql_query=sql_query,
    csv_files=csv_files,
    delimiter=",",
)

print(json.dumps(payload, indent=2, ensure_ascii=False))