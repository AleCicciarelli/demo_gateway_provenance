#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:  # Compatibility with the existing gateway image.
    from langchain_community.embeddings import HuggingFaceEmbeddings

TEXTUALIZATION_STRATEGIES = ("rich", "semantic-join")


def is_provenance_column(col: str) -> bool:
    return re.fullmatch(r".*_rownum", str(col).lower().strip()) is not None


def clean_value(v: Any) -> Any:
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def value_to_text(v: Any, max_len: int = 120) -> str:
    v = clean_value(v)
    if v is None:
        return "NULL"
    s = str(v).strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def table_name_from_file(path: Path) -> str:
    return path.stem.lower()


def read_tables(csv_dir: Path, sep: str = ",") -> dict[str, pd.DataFrame]:
    tables = {}

    for path in sorted(csv_dir.glob("*.csv")):
        table = table_name_from_file(path)
        df = pd.read_csv(path, sep=sep, engine="python")
        df.columns = [str(c).strip() for c in df.columns]
        tables[table] = df
        print(f"Loaded {table}: {len(df)} rows, {len(df.columns)} columns")

    return tables


def get_primary_key_columns(schema_profile: dict[str, Any], table: str) -> list[str]:
    candidates = schema_profile["tables"][table].get("primary_key_candidates", [])
    for cand in candidates:
        cols = [c for c in cand["columns"] if not is_provenance_column(c)]
        if cols:
            return cols
    return []


def find_rownum_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if is_provenance_column(col):
            return col
    return None


def make_row_id(table: str, row: pd.Series, pk_cols: list[str], rownum_col: str | None) -> str:
    if rownum_col is not None:
        return f"{table}:{rownum_col}={value_to_text(row[rownum_col])}"

    if pk_cols:
        pk_text = ",".join(f"{c}={value_to_text(row[c])}" for c in pk_cols)
        return f"{table}:{pk_text}"

    return f"{table}:index={row.name}"


def build_target_indexes(
    tables: dict[str, pd.DataFrame],
    foreign_keys: list[dict[str, Any]],
) -> dict[tuple[str, tuple[str, ...]], dict[tuple[str, ...], dict[str, Any]]]:
    """
    Build lookup:
    (target_table, target_columns) -> key_tuple -> target row dict
    """
    indexes = {}

    needed_targets = set()
    for fk in foreign_keys:
        needed_targets.add((fk["to_table"], tuple(fk["to_columns"])))

    for target_table, target_cols in needed_targets:
        if target_table not in tables:
            continue

        df = tables[target_table]
        lookup = {}

        for _, row in df.iterrows():
            key = tuple(value_to_text(row[c]) for c in target_cols)
            lookup[key] = {c: clean_value(row[c]) for c in df.columns}

        indexes[(target_table, target_cols)] = lookup

    return indexes


def choose_display_columns(row: dict[str, Any], max_cols: int = 4) -> list[str]:
    """
    Generic heuristic: select readable columns from linked row.
    """
    preferred_patterns = [
        "name", "title", "label", "status", "type",
        "category", "segment", "region", "nation",
        "country", "date", "brand"
    ]

    cols = list(row.keys())
    non_prov = [c for c in cols if not is_provenance_column(c)]

    preferred = []
    for c in non_prov:
        n = c.lower()
        if any(p in n for p in preferred_patterns):
            preferred.append(c)

    if preferred:
        return preferred[:max_cols]

    fallback = []
    for c in non_prov:
        v = row[c]
        if v is None:
            continue
        s = str(v)
        if len(s) <= 80:
            fallback.append(c)

    return fallback[:max_cols]


def outgoing_fks_for_table(schema_profile: dict[str, Any], table: str) -> list[dict[str, Any]]:
    return [
        fk for fk in schema_profile.get("foreign_key_candidates", [])
        if fk["from_table"] == table
        and float(fk.get("name_similarity", 0.0)) > 0.0
    ]


def build_page_content(
    table: str,
    row: pd.Series,
    pk_cols: list[str],
    outgoing_fks: list[dict[str, Any]],
    target_indexes: dict[tuple[str, tuple[str, ...]], dict[tuple[str, ...], dict[str, Any]]],
    max_value_cols: int = 40,
) -> tuple[str, list[dict[str, Any]]]:
    lines = []

    lines.append(f"Row from table {table}.")
    lines.append(f"This row represents one record from {table}.")
    lines.append("")

    if pk_cols:
        lines.append("Primary key:")
        for col in pk_cols:
            lines.append(f"{table}.{col} = {value_to_text(row[col])}.")
        lines.append("")

    lines.append("Column values:")

    count = 0
    for col in row.index:
        if is_provenance_column(col):
            continue
        value = value_to_text(row[col])

        # Skip very long null/noisy values if needed
        lines.append(f"{table}.{col} = {value}.")
        count += 1

        if count >= max_value_cols:
            break

    lines.append("")

    linked_rows = []

    if outgoing_fks:
        lines.append("Foreign key relations:")

    for fk in outgoing_fks:
        from_cols = fk["from_columns"]
        to_table = fk["to_table"]
        to_cols = tuple(fk["to_columns"])

        source_key = tuple(value_to_text(row[c]) for c in from_cols)
        target_lookup = target_indexes.get((to_table, to_cols), {})
        linked = target_lookup.get(source_key)

        from_expr = ", ".join(
            f"{table}.{c} = {value_to_text(row[c])}"
            for c in from_cols
        )
        to_expr = ", ".join(
            f"{to_table}.{c} = {v}"
            for c, v in zip(to_cols, source_key)
        )

        lines.append(f"{from_expr} references {to_expr}.")

        link_meta = {
            "relation": "foreign_key",
            "from_table": table,
            "from_columns": from_cols,
            "from_values": list(source_key),
            "to_table": to_table,
            "to_columns": list(to_cols),
            "linked_values": {},
        }

        if linked is not None:
            display_cols = choose_display_columns(linked)
            lines.append(f"Linked {to_table} row:")

            for c in display_cols:
                lines.append(f"{to_table}.{c} = {value_to_text(linked[c])}.")
                link_meta["linked_values"][f"{to_table}.{c}"] = clean_value(linked[c])

        linked_rows.append(link_meta)

    return "\n".join(lines), linked_rows


def build_semantic_join_page_content(
    row: pd.Series,
    max_value_cols: int = 40,
) -> str:
    """Textualize one record as compact natural-language field assertions."""
    fields = []
    for col in row.index:
        if is_provenance_column(col):
            continue
        fields.append(f"{col} is {value_to_text(row[col])}")
        if len(fields) >= max_value_cols:
            break
    return "; ".join(fields)


def build_documents(
    tables: dict[str, pd.DataFrame],
    schema_profile: dict[str, Any],
    textualization_strategy: str = "rich",
) -> list[Document]:
    if textualization_strategy not in TEXTUALIZATION_STRATEGIES:
        raise ValueError(
            f"Unknown textualization strategy '{textualization_strategy}'. "
            f"Choose one of: {', '.join(TEXTUALIZATION_STRATEGIES)}"
        )

    docs = []
    foreign_keys = schema_profile.get("foreign_key_candidates", [])
    target_indexes = build_target_indexes(tables, foreign_keys)

    for table, df in tables.items():
        if table not in schema_profile["tables"]:
            print(f"Skipping {table}: not found in schema profile")
            continue

        pk_cols = get_primary_key_columns(schema_profile, table)
        rownum_col = find_rownum_column(df)
        outgoing_fks = outgoing_fks_for_table(schema_profile, table)

        for _, row in df.iterrows():
            row_id = make_row_id(table, row, pk_cols, rownum_col)

            page_content, linked_rows = build_page_content(
                table=table,
                row=row,
                pk_cols=pk_cols,
                outgoing_fks=outgoing_fks,
                target_indexes=target_indexes,
            )
            if textualization_strategy == "semantic-join":
                page_content = build_semantic_join_page_content(row)

            values = {
                col: clean_value(row[col])
                for col in df.columns
                if not is_provenance_column(col)
            }

            primary_key = {
                col: clean_value(row[col])
                for col in pk_cols
            }

            metadata = {
                "doc_type": "row",
                "table": table,
                "row_id": row_id,
                "rownum_column": rownum_col,
                "rownum_value": clean_value(row[rownum_col]) if rownum_col else None,
                "primary_key": primary_key,
                "values": values,
                "linked_rows": linked_rows,
                "textualization_strategy": textualization_strategy,
            }

            docs.append(Document(page_content=page_content, metadata=metadata))

    return docs


def save_jsonl(docs: list[Document], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", required=True)
    parser.add_argument("--schema_profile", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--documents_out", default="row_documents.jsonl")
    parser.add_argument("--faiss_out", default="faiss_index_bge_m3_rows")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--textualization-strategy",
        choices=TEXTUALIZATION_STRATEGIES,
        default="rich",
        help=(
            "Row text format. 'rich' preserves the current table/PK/FK format; "
            "'semantic-join' emits compact 'column is value; ...' records."
        ),
    )
    parser.add_argument("--documents-only", action="store_true")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    schema_profile = json.loads(Path(args.schema_profile).read_text(encoding="utf-8"))

    tables = read_tables(csv_dir, sep=args.sep)
    docs = build_documents(
        tables,
        schema_profile,
        textualization_strategy=args.textualization_strategy,
    )

    print(f"Built {len(docs)} row documents")

    documents_out = Path(args.documents_out)
    save_jsonl(docs, documents_out)
    print(f"Saved row documents to {documents_out}")

    if args.documents_only:
        return

    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": args.device},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(args.faiss_out)

    print(f"Saved FAISS index to {args.faiss_out}")


if __name__ == "__main__":
    main()
