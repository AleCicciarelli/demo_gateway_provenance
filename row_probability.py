"""Resolve pipeline estimates without changing query-column values."""
import math
import os

PROBABILITY_COLUMN = "__probability"


def valid_probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def resolve_row_probabilities(leaves):
    """Return probabilities by (table, row identity), merging duplicates conservatively."""
    levels = {level: float(os.getenv(f"LLM_PROBABILITY_{level.upper()}", default))
              for level, default in {"high": "0.9", "medium": "0.6", "low": "0.3"}.items()}
    if not all(valid_probability(value) for value in levels.values()):
        raise ValueError("LLM probability mappings must be finite numbers in [0, 1]")
    resolved = {}
    for leaf in leaves:
        table = leaf.get("table_name")
        if not table:
            continue
        evidence = {}
        for annotation in leaf.get("annotations") or []:
            if annotation.get("type") == "rag_similarity":
                for entry in annotation.get("evidence") or []:
                    if entry.get("row_id"):
                        evidence.setdefault(entry["row_id"], []).append(entry.get("score"))
        for item in leaf.get("parsed_output") or []:
            if not isinstance(item, dict) or not isinstance(item.get("values"), dict):
                continue
            row_id = item.get("row_id")
            key = (table, row_id or id(item))
            pipeline = leaf.get("pipeline")
            assessments = [a for a in item.get("confidence_annotations") or []
                           if a.get("type") == "llm_row_confidence"]
            if pipeline == "sql-table" or leaf.get("prompt") == "SQL TABLE MODE":
                probability, metric = 1.0, "deterministic_execution"
            elif pipeline == "llm-internal" or assessments or any(
                    a.get("type") == "llm_confidence" for a in leaf.get("annotations") or []):
                scores = [levels.get(a.get("level")) for a in assessments]
                probability = min(scores) if scores and all(valid_probability(s) for s in scores) else None
                metric = "llm_self_assessment"
            else:
                scores = evidence.get(row_id, [])
                probability = min(scores) if scores and all(valid_probability(s) for s in scores) else None
                metric = "faiss_relevance"
            source = {"table": table, "row_id": row_id, "pipeline": pipeline,
                      "probability": probability, "metric": metric}
            if key in resolved:
                previous = resolved[key]
                source["probability"] = (min(previous["probability"], probability)
                    if previous["probability"] is not None and probability is not None else None)
                if previous["metric"] != metric:
                    source["metric"] = "mixed_pipeline"
            resolved[key] = source
    return resolved


def protect_probability_projection(sql_query, csv_columns):
    """Expand SELECT wildcards against data columns, excluding added metadata."""
    if not csv_columns:
        return sql_query
    import sqlglot
    from sqlglot import exp
    from sqlglot.optimizer.qualify import qualify
    from sqlglot.schema import MappingSchema

    query = sqlglot.parse_one(sql_query, read="postgres")
    if not any(expression.is_star for select in query.find_all(exp.Select)
               for expression in select.expressions):
        return sql_query
    schema = MappingSchema(
        {filename.removesuffix(".csv"): {column: "TEXT" for column in columns}
         for filename, columns in csv_columns.items()}, dialect="postgres", normalize=False,
    )
    query = qualify(query, dialect="postgres", schema=schema,
                    validate_qualify_columns=False)
    if any(expression.is_star for select in query.find_all(exp.Select)
           for expression in select.expressions):
        raise ValueError("Cannot safely expand SELECT wildcard using exported CSV columns")
    return query.sql(dialect="postgres")
