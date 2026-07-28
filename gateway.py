#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import asdict, dataclass
import html
import json
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import csv
import shutil
from pathlib import Path
from collections import defaultdict
import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from embedding_strategies import EmbeddingStrategies
from explanation_client import ExplanationClient
from explanation_pipeline import run_planner_first_explanation_pipeline
from json_to_csv import clean_bucket, planner_result_to_csv_files
from faiss_index_manager import FaissIndexManager
from iterative_join_pipeline import run_iterative_join_pipeline
from prompt import build_iterative_join_leaf_prompt, build_leaf_prompt
from prompt_internal_knowledge import (
    PROMPT_INTERNAL_KNOWLEDGE_TEMPLATE,
    get_internal_knowledge_prompt_template,
)
from tpch_schema_info import SCHEMA_INFO as TPCH_SCHEMA_INFO
from planner import  build_query_plan
import uuid

app = FastAPI()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# =========================
# Config
# =========================
LOG_PATH = os.getenv("GATEWAY_LOG_PATH", "/app/logs/provsql_gateway_logs.jsonl")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "700"))
LLM_API_BASE = os.getenv("LLM_API_BASE", "").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_MODEL = os.getenv("LLM_API_MODEL", "")
LLM_API_MODEL_ALIASES = [
    item.strip()
    for item in os.getenv("LLM_API_MODEL_ALIASES", "").split(",")
    if item.strip()
]
LLM_SSL_VERIFY = _env_bool("LLM_SSL_VERIFY", True)
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", str(OLLAMA_REQUEST_TIMEOUT)))
PLANNER_LLM_PROVIDER = os.getenv("PLANNER_LLM_PROVIDER", "ollama").strip().lower()
PLANNER_LLM_FALLBACK_ENABLED = _env_bool("PLANNER_LLM_FALLBACK_ENABLED", True)
PLANNER_LLM_FALLBACK_MODEL = os.getenv("PLANNER_LLM_FALLBACK_MODEL", "llama3:8b")
PLANNER_LLM_MODEL = os.getenv(
    "PLANNER_LLM_MODEL",
    LLM_API_MODEL if PLANNER_LLM_PROVIDER in {"openai", "openai-compatible"} and LLM_API_MODEL else "llama3:8b",
)
DEFAULT_DATASET = os.getenv("DEFAULT_DATASET", os.getenv("DATASET", "tpch")).strip().lower()
CSV_DIR = os.getenv("CSV_DIR", "/app/tpch_no_provsql")
MAX_CONTEXT_ROWS = int(os.getenv("MAX_CONTEXT_ROWS", "37"))
MAX_TABLES = int(os.getenv("MAX_TABLES", "5"))
INCLUDE_CORRELATED_CONTEXT_ROWS = _env_bool("INCLUDE_CORRELATED_CONTEXT_ROWS", True)
MAX_CORRELATED_CONTEXT_ROWS = int(os.getenv("MAX_CORRELATED_CONTEXT_ROWS", "24"))
GATEWAY_RESPONSE_FORMAT = os.getenv("GATEWAY_RESPONSE_FORMAT", "json").strip().lower()
GATEWAY_MARKDOWN_INCLUDE_RAW = os.getenv("GATEWAY_MARKDOWN_INCLUDE_RAW", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
GATEWAY_MARKDOWN_MAX_ROWS = int(os.getenv("GATEWAY_MARKDOWN_MAX_ROWS", "50"))
GATEWAY_MARKDOWN_MAX_CELL_CHARS = int(os.getenv("GATEWAY_MARKDOWN_MAX_CELL_CHARS", "120"))

OLLAMA_MODEL_BASE = os.getenv("OLLAMA_MODEL_BASE", "llama3:8b")
OLLAMA_MODEL_FT_NL = os.getenv("OLLAMA_MODEL_FT_NL", "llama3-8b-dpo2-sft1-nl:latest")
OLLAMA_MODEL_FT_SQL = os.getenv("OLLAMA_MODEL_FT_SQL", "llama3-8b-dpo1-sft2-sql:latest")

FAISS_INDEX_FOLDER = os.getenv("FAISS_INDEX_FOLDER", "/app/faiss_index_tpch")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMB_STRATEGY = os.getenv("EMB_STRATEGY", "auto")
EMB_DEVICE = os.getenv("EMB_DEVICE", "auto")
INDEX_TABLES = os.getenv("INDEX_TABLES", "")
INDEX_SET = set(t.strip() for t in INDEX_TABLES.split(",") if t.strip())
BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "500"))

RELF1_CSV_DIR = os.getenv("RELF1_CSV_DIR", "/app/rel-f1-csv")
RELF1_FAISS_INDEX_FOLDER = os.getenv("RELF1_FAISS_INDEX_FOLDER", "/app/faiss_index_relf1_rows_bge_m3")
RELF1_EMB_MODEL = os.getenv("RELF1_EMB_MODEL", "BAAI/bge-m3")
RELF1_EMB_STRATEGY = os.getenv("RELF1_EMB_STRATEGY", "bge-m3")
RELF1_INDEX_TABLES = os.getenv("RELF1_INDEX_TABLES", "")
RELF1_INDEX_SET = set(t.strip() for t in RELF1_INDEX_TABLES.split(",") if t.strip())
RELF1_SCHEMA_PROFILE = os.getenv("RELF1_SCHEMA_PROFILE", "/app/rel-f1-csv/schema_profile_relf1.json")

EXPLAIN_MODEL = os.getenv("OLLAMA_MODEL_EXPLAIN", "deepseek-r1:70b")
EXPLAIN_MAX_TRIES = int(os.getenv("EXPLAIN_MAX_TRIES", "2"))

RETRIEVER_K = int(os.getenv("RETRIEVER_K", "12"))
MAX_ITERATIVE_RETRIEVALS = int(os.getenv("MAX_ITERATIVE_RETRIEVALS", "2"))

# For ap-explanation service:
EXPLANATION_URL = os.getenv("EXPLANATION_URL", "http://explanation_app:5000")
EXPLANATION_ENDPOINT = os.getenv("EXPLANATION_ENDPOINT", "/api/v1/aps/explanation/why",)
EXPLANATION_BUCKET_DIR = Path(os.getenv("EXPLANATION_BUCKET_DIR", "/shared_bucket"))
EXPLANATION_REQUEST_TIMEOUT = float(os.getenv("EXPLANATION_REQUEST_TIMEOUT", "300"))

EXPLANATION_CSV_DELIMITER = os.getenv("EXPLANATION_CSV_DELIMITER", ",")
EXPLANATION_KEEP_ROWNUM = _env_bool("EXPLANATION_KEEP_ROWNUM", True)
EXPLANATION_RESET_FORMULA_STATE = _env_bool("EXPLANATION_RESET_FORMULA_STATE", False)
EXPLANATION_POSTGRES_HOST = os.getenv("EXPLANATION_POSTGRES_HOST", "postgres-provsql")
EXPLANATION_POSTGRES_PORT = int(os.getenv("EXPLANATION_POSTGRES_PORT", "5432"))
EXPLANATION_POSTGRES_DB = os.getenv("EXPLANATION_POSTGRES_DB", "mathe")
EXPLANATION_POSTGRES_USER = os.getenv("EXPLANATION_POSTGRES_USER", "provdemo")
EXPLANATION_POSTGRES_PASSWORD = os.getenv("EXPLANATION_POSTGRES_PASSWORD", "provdemo")

# una pipeline explanation alla volta per evitare che due pipeline concorrenti scrivano i csv nella stessa cartella del bucket
_EXPLANATION_PIPELINE_LOCK = threading.Lock()
RAG_PIPELINE_ID = "rag"
LLM_INTERNAL_PIPELINE_ID = "llm-internal"
SQL_TABLE_PIPELINE_ID = "sql-table"
# Legacy constants remain internal compatibility shims for older API clients.
PLANNER_ONLY_MODEL_ID = RAG_PIPELINE_ID
PLANNER_ONLY_PUSHDOWN_MODEL_ID = "rag-pushdown"
PLANNER_ONLY_EXPLANATION_MODEL_ID = "rag-explanation"
ITERATIVE_PIPELINE_MODEL_ID = "rag-iterative"
PLANNER_ONLY_ALIASES = {
    "planner only": RAG_PIPELINE_ID,
    "planner-only": RAG_PIPELINE_ID,
    "planner-first": RAG_PIPELINE_ID,
    "planner_first": RAG_PIPELINE_ID,
    "planner only pushdown": PLANNER_ONLY_PUSHDOWN_MODEL_ID,
    "planner-only-pushdown": PLANNER_ONLY_PUSHDOWN_MODEL_ID,
    "planner-first-pushdown": PLANNER_ONLY_PUSHDOWN_MODEL_ID,
    "planner_first_pushdown": PLANNER_ONLY_PUSHDOWN_MODEL_ID,
    "planner only-explanation": PLANNER_ONLY_EXPLANATION_MODEL_ID,
    "planner-only-explanation": PLANNER_ONLY_EXPLANATION_MODEL_ID,
    "planner-first-explanation": PLANNER_ONLY_EXPLANATION_MODEL_ID,
    "planner_first_explanation": PLANNER_ONLY_EXPLANATION_MODEL_ID,
    "iterative join aware": ITERATIVE_PIPELINE_MODEL_ID,
    "iterative-join-aware": ITERATIVE_PIPELINE_MODEL_ID,
    "internal-knowledge": LLM_INTERNAL_PIPELINE_ID,
    "internal knowledge": LLM_INTERNAL_PIPELINE_ID,
    "llm internal": LLM_INTERNAL_PIPELINE_ID,
    "sql table": SQL_TABLE_PIPELINE_ID,
}


def _canonical_pipeline_id(pipeline: Optional[str]) -> str:
    pipeline_id = (pipeline or PLANNER_ONLY_MODEL_ID).strip().lower()
    return PLANNER_ONLY_ALIASES.get(pipeline_id, pipeline_id)


def _canonical_model_id(model: str) -> str:
    model_id = (model or "").strip()
    return PLANNER_ONLY_ALIASES.get(model_id.lower(), model_id)


# Mapping "UI model id" -> "Ollama model name".
MODEL_ROUTING: Dict[str, str] = {
    "base-llama3-8b": os.getenv("OLLAMA_MODEL_BASE", "llama3:8b"),
    "best-ft-llama3-8b-nl": os.getenv("OLLAMA_MODEL_FT_NL", "llama3-8b-dpo2-sft1-nl:latest"),
    "best-ft-llama3-8b-sql": os.getenv("OLLAMA_MODEL_FT_SQL", "llama3-8b-dpo1-sft2-sql:latest"),
    PLANNER_ONLY_MODEL_ID: PLANNER_LLM_MODEL
    if PLANNER_LLM_PROVIDER in {"openai", "openai-compatible"}
    else os.getenv("OLLAMA_MODEL_PLANNER_FIRST", PLANNER_LLM_MODEL),
    PLANNER_ONLY_PUSHDOWN_MODEL_ID: PLANNER_LLM_MODEL
    if PLANNER_LLM_PROVIDER in {"openai", "openai-compatible"}
    else os.getenv("OLLAMA_MODEL_PLANNER_FIRST_PUSHDOWN", PLANNER_LLM_MODEL),
    PLANNER_ONLY_EXPLANATION_MODEL_ID: PLANNER_LLM_MODEL
    if PLANNER_LLM_PROVIDER in {"openai", "openai-compatible"}
    else os.getenv("OLLAMA_MODEL_PLANNER_FIRST_EXPLANATION", PLANNER_LLM_MODEL),
    LLM_INTERNAL_PIPELINE_ID: os.getenv("OLLAMA_MODEL_INTERNAL_KNOWLEDGE", "llama3:8b"),
    SQL_TABLE_PIPELINE_ID: os.getenv("OLLAMA_MODEL_BASE", "llama3:8b"),
    ITERATIVE_PIPELINE_MODEL_ID: os.getenv("OLLAMA_MODEL_ITERATIVE_PIPELINE", PLANNER_LLM_MODEL),
}
for alias, canonical in PLANNER_ONLY_ALIASES.items():
    MODEL_ROUTING.setdefault(alias, MODEL_ROUTING[canonical])

# Exposing only the UI model ids in the /v1/models endpoint.
EXPOSED_MODEL_IDS = [
    "base-llama3-8b",
    "best-ft-llama3-8b-nl",
    "best-ft-llama3-8b-sql",
    RAG_PIPELINE_ID,
    LLM_INTERNAL_PIPELINE_ID,
    SQL_TABLE_PIPELINE_ID,
]
EXPOSED_MODELS = [{"id": mid, "object": "model"} for mid in EXPOSED_MODEL_IDS]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    csv_dir: str
    faiss_index_folder: str
    emb_model: str
    emb_strategy: str
    index_set: set[str]
    schema_info: Dict[str, Any]


@dataclass
class DatasetRuntime:
    csv_cache: Dict[str, List[Dict[str, Any]]]
    csv_loaded: bool
    csv_rid_index: Dict[str, Dict[str, int]]
    global_rid_index: Dict[str, Tuple[str, int]]
    embeddings: Optional[EmbeddingStrategies]
    faiss_manager: Optional[FaissIndexManager]

def _schema_info_from_profile(profile_path: str) -> Dict[str, Any]:
    path = Path(profile_path)
    if not path.exists():
        return {}

    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[dataset] Could not read schema profile {profile_path}: {exc}", flush=True)
        return {}

    tables = profile.get("tables") or {}
    foreign_key_candidates = profile.get("foreign_key_candidates") or []
    schema_info: Dict[str, Any] = {}

    for table, info in tables.items():
        columns = [
            col
            for col, col_info in (info.get("columns") or {}).items()
            if not col.endswith("_rownum") and not bool(col_info.get("is_provenance"))
        ]
        column_types = {
            str(col): str(col_info.get("type") or "").strip().lower()
            for col, col_info in (info.get("columns") or {}).items()
            if not str(col).endswith("_rownum")
            and isinstance(col_info, dict)
            and not bool(col_info.get("is_provenance"))
        }
        row_id = next(
            (col for col in (info.get("columns") or {}) if col.endswith("_rownum")),
            f"{table}_rownum",
        )
        primary_key: Any = None
        pk_candidates = info.get("primary_key_candidates") or []
        if pk_candidates:
            pk_cols = [
                col
                for col in (pk_candidates[0].get("columns") or [])
                if not str(col).endswith("_rownum")
            ]
            if len(pk_cols) == 1:
                primary_key = pk_cols[0]
            elif pk_cols:
                primary_key = pk_cols

        schema_info[table] = {
            "columns": columns,
            "column_types": column_types,
            "primary_key": primary_key,
            "row_id": row_id,
            "foreign_keys": {},
        }

    for fk in foreign_key_candidates:
        from_table = fk.get("from_table")
        if from_table not in schema_info:
            continue
        from_cols = fk.get("from_columns") or []
        to_table = fk.get("to_table")
        to_cols = fk.get("to_columns") or []
        for from_col, to_col in zip(from_cols, to_cols):
            schema_info[from_table]["foreign_keys"][from_col] = f"{to_table}.{to_col}"

    return schema_info


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "tpch": DatasetConfig(
        name="tpch",
        csv_dir=CSV_DIR,
        faiss_index_folder=FAISS_INDEX_FOLDER,
        emb_model=EMB_MODEL,
        emb_strategy=EMB_STRATEGY,
        index_set=INDEX_SET,
        schema_info=TPCH_SCHEMA_INFO,
    ),
    "relf1": DatasetConfig(
        name="relf1",
        csv_dir=RELF1_CSV_DIR,
        faiss_index_folder=RELF1_FAISS_INDEX_FOLDER,
        emb_model=RELF1_EMB_MODEL,
        emb_strategy=RELF1_EMB_STRATEGY,
        index_set=RELF1_INDEX_SET,
        schema_info=_schema_info_from_profile(RELF1_SCHEMA_PROFILE),
    ),
}
DATASET_ALIASES = {
    "rel-f1": "relf1",
    "rel_f1": "relf1",
    "f1": "relf1",
    "formula1": "relf1",
    "tpc-h": "tpch",
    "tpc_h": "tpch",
}

_DATASET_RUNTIMES: Dict[str, DatasetRuntime] = {}
_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _identifier_needs_quotes(identifier: str) -> bool:
    return not bool(_IDENTIFIER_RE.match(identifier)) or identifier.upper() in {
        "ALL",
        "ANALYSE",
        "ANALYZE",
        "AND",
        "ANY",
        "ARRAY",
        "AS",
        "ASC",
        "ASYMMETRIC",
        "BOTH",
        "CASE",
        "CAST",
        "CHECK",
        "COLLATE",
        "COLUMN",
        "CONSTRAINT",
        "CREATE",
        "CURRENT_CATALOG",
        "CURRENT_DATE",
        "CURRENT_ROLE",
        "CURRENT_TIME",
        "CURRENT_TIMESTAMP",
        "CURRENT_USER",
        "DEFAULT",
        "DEFERRABLE",
        "DESC",
        "DISTINCT",
        "DO",
        "ELSE",
        "END",
        "EXCEPT",
        "FALSE",
        "FETCH",
        "FOR",
        "FOREIGN",
        "FROM",
        "GRANT",
        "GROUP",
        "HAVING",
        "IN",
        "INITIALLY",
        "INTERSECT",
        "INTO",
        "LATERAL",
        "LEADING",
        "LIMIT",
        "LOCALTIME",
        "LOCALTIMESTAMP",
        "NOT",
        "NULL",
        "OFFSET",
        "ON",
        "ONLY",
        "OR",
        "ORDER",
        "PLACING",
        "PRIMARY",
        "REFERENCES",
        "RETURNING",
        "SELECT",
        "SESSION_USER",
        "SOME",
        "SYMMETRIC",
        "TABLE",
        "THEN",
        "TO",
        "TRAILING",
        "TRUE",
        "UNION",
        "UNIQUE",
        "USER",
        "USING",
        "VARIADIC",
        "WHEN",
        "WHERE",
        "WINDOW",
        "WITH",
    }


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _normalize_dataset(dataset: Optional[str]) -> str:
    name = (dataset or DEFAULT_DATASET or "tpch").strip().lower()
    name = DATASET_ALIASES.get(name, name)
    if name not in DATASET_CONFIGS:
        known = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset '{dataset}'. Use one of: {known}")
    return name


def _dataset_config(dataset: Optional[str] = None) -> DatasetConfig:
    return DATASET_CONFIGS[_normalize_dataset(dataset)]


def _dataset_columns_requiring_quotes(dataset: Optional[str] = None) -> set[str]:
    config = _dataset_config(dataset)
    columns: set[str] = set()
    for table_info in config.schema_info.values():
        if not isinstance(table_info, dict):
            continue
        for column in table_info.get("columns") or []:
            column_name = str(column)
            if _identifier_needs_quotes(column_name):
                columns.add(column_name)
        row_id = table_info.get("row_id")
        if row_id and _identifier_needs_quotes(str(row_id)):
            columns.add(str(row_id))
    return columns


def _dataset_numeric_columns(dataset: Optional[str] = None) -> set[str]:
    config = _dataset_config(dataset)
    numeric_types = {"decimal", "double", "float", "integer", "number", "numeric", "real"}
    columns: set[str] = set()
    for table_info in config.schema_info.values():
        if not isinstance(table_info, dict):
            continue
        for column_name, column_type in (table_info.get("column_types") or {}).items():
            if str(column_type or "").strip().lower() in numeric_types:
                columns.add(str(column_name))
        columns_info = table_info.get("columns") or {}
        if isinstance(columns_info, dict):
            for column_name, column_info in columns_info.items():
                if not isinstance(column_info, dict):
                    continue
                if str(column_info.get("type") or "").strip().lower() in numeric_types:
                    columns.add(str(column_name))
    return columns


def _quote_sql_columns_for_dataset(sql_query: str, dataset: Optional[str] = None) -> str:
    columns_to_quote = _dataset_columns_requiring_quotes(dataset)
    if not columns_to_quote:
        return sql_query

    out: List[str] = []
    index = 0
    length = len(sql_query)

    while index < length:
        char = sql_query[index]

        if char == "'":
            start = index
            index += 1
            while index < length:
                if sql_query[index] == "'":
                    index += 1
                    if index < length and sql_query[index] == "'":
                        index += 1
                        continue
                    break
                index += 1
            out.append(sql_query[start:index])
            continue

        if char == '"':
            start = index
            index += 1
            while index < length:
                if sql_query[index] == '"':
                    index += 1
                    if index < length and sql_query[index] == '"':
                        index += 1
                        continue
                    break
                index += 1
            out.append(sql_query[start:index])
            continue

        if char == "-" and index + 1 < length and sql_query[index + 1] == "-":
            end = sql_query.find("\n", index + 2)
            if end == -1:
                out.append(sql_query[index:])
                break
            out.append(sql_query[index:end])
            index = end
            continue

        if char == "/" and index + 1 < length and sql_query[index + 1] == "*":
            end = sql_query.find("*/", index + 2)
            if end == -1:
                out.append(sql_query[index:])
                break
            out.append(sql_query[index : end + 2])
            index = end + 2
            continue

        if char == "$":
            match = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$", sql_query[index:])
            if match:
                tag = match.group(0)
                end = sql_query.find(tag, index + len(tag))
                if end != -1:
                    out.append(sql_query[index : end + len(tag)])
                    index = end + len(tag)
                    continue

        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < length and (
                sql_query[index].isalnum() or sql_query[index] in {"_", "$"}
            ):
                index += 1
            identifier = sql_query[start:index]
            if identifier in columns_to_quote:
                out.append(_quote_identifier(identifier))
            else:
                out.append(identifier)
            continue

        out.append(char)
        index += 1

    return "".join(out)


_SQL_IDENTIFIER_TOKEN = r'(?:[A-Za-z_][A-Za-z0-9_$]*|"(?:""|[^"])+")'
_SQL_COLUMN_REF = rf'(?:{_SQL_IDENTIFIER_TOKEN}\s*\.\s*)?{_SQL_IDENTIFIER_TOKEN}'
_SQL_NUMERIC_LITERAL = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_SQL_COMPARISON_OP = r"=|<>|!=|<=|>=|<|>"
_CSV_TEXT_LEFT_COMPARISON_RE = re.compile(
    rf"(?<![A-Za-z0-9_$.:])(?P<col>{_SQL_COLUMN_REF})(?P<space1>\s*)(?P<op>{_SQL_COMPARISON_OP})(?P<space2>\s*)"
    rf"(?P<num>{_SQL_NUMERIC_LITERAL})(?![A-Za-z0-9_$.])"
)
_CSV_TEXT_RIGHT_COMPARISON_RE = re.compile(
    rf"(?<![A-Za-z0-9_$.])(?P<num>{_SQL_NUMERIC_LITERAL})(?P<space1>\s*)"
    rf"(?P<op>{_SQL_COMPARISON_OP})(?P<space2>\s*)(?P<col>{_SQL_COLUMN_REF})"
)
_CSV_TEXT_IN_RE = re.compile(
    rf"(?<![A-Za-z0-9_$.:])(?P<col>{_SQL_COLUMN_REF})(?P<space>\s+)(?P<not>NOT\s+)?IN\s*\((?P<items>[^()]+)\)",
    flags=re.IGNORECASE,
)
_CSV_TEXT_NUMERIC_LIST_ITEM_RE = re.compile(
    rf"(?<![A-Za-z0-9_$.']){_SQL_NUMERIC_LITERAL}(?![A-Za-z0-9_$.'])"
)


def _sql_column_name(column_ref: str) -> str:
    token = column_ref.split(".")[-1].strip()
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1].replace('""', '"')
    return token


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _cast_sql_column_numeric(column_ref: str) -> str:
    if "::" in column_ref:
        return column_ref
    return f"{column_ref}::numeric"


def _rewrite_csv_text_numeric_comparisons_in_code(sql_code: str, dataset: Optional[str]) -> str:
    numeric_columns = _dataset_numeric_columns(dataset)

    def replace_left(match: re.Match[str]) -> str:
        col = match.group("col")
        num = match.group("num")
        if _sql_column_name(col) in numeric_columns:
            lhs = _cast_sql_column_numeric(col)
            rhs = num
        else:
            lhs = col
            rhs = _sql_string_literal(num)
        return f"{lhs}{match.group('space1')}{match.group('op')}{match.group('space2')}{rhs}"

    def replace_right(match: re.Match[str]) -> str:
        col = match.group("col")
        num = match.group("num")
        if _sql_column_name(col) in numeric_columns:
            lhs = num
            rhs = _cast_sql_column_numeric(col)
        else:
            lhs = _sql_string_literal(num)
            rhs = col
        return f"{lhs}{match.group('space1')}{match.group('op')}{match.group('space2')}{rhs}"

    def replace_in(match: re.Match[str]) -> str:
        col = match.group("col")
        items = match.group("items")
        if _sql_column_name(col) in numeric_columns:
            rewritten_items = items
            rewritten_col = _cast_sql_column_numeric(col)
        else:
            rewritten_items = _CSV_TEXT_NUMERIC_LIST_ITEM_RE.sub(
                lambda item_match: _sql_string_literal(item_match.group(0)),
                items,
            )
            rewritten_col = col
        not_sql = match.group("not") or ""
        return f"{rewritten_col}{match.group('space')}{not_sql}IN ({rewritten_items})"

    sql_code = _CSV_TEXT_LEFT_COMPARISON_RE.sub(replace_left, sql_code)
    sql_code = _CSV_TEXT_RIGHT_COMPARISON_RE.sub(replace_right, sql_code)
    return _CSV_TEXT_IN_RE.sub(replace_in, sql_code)


def _rewrite_csv_text_numeric_comparisons(sql_query: str, dataset: Optional[str] = None) -> str:
    out: List[str] = []
    index = 0
    length = len(sql_query)

    while index < length:
        start = index
        while index < length:
            char = sql_query[index]
            if char == "'":
                break
            if char == "-" and index + 1 < length and sql_query[index + 1] == "-":
                break
            if char == "/" and index + 1 < length and sql_query[index + 1] == "*":
                break
            if char == "$":
                match = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$", sql_query[index:])
                if match:
                    break
            index += 1

        if index > start:
            out.append(_rewrite_csv_text_numeric_comparisons_in_code(sql_query[start:index], dataset))
            continue

        char = sql_query[index]
        if char == "'":
            index += 1
            while index < length:
                if sql_query[index] == "'":
                    index += 1
                    if index < length and sql_query[index] == "'":
                        index += 1
                        continue
                    break
                index += 1
            out.append(sql_query[start:index])
            continue

        if char == "-" and index + 1 < length and sql_query[index + 1] == "-":
            end = sql_query.find("\n", index + 2)
            if end == -1:
                out.append(sql_query[index:])
                break
            out.append(sql_query[index:end])
            index = end
            continue

        if char == "/" and index + 1 < length and sql_query[index + 1] == "*":
            end = sql_query.find("*/", index + 2)
            if end == -1:
                out.append(sql_query[index:])
                break
            out.append(sql_query[index : end + 2])
            index = end + 2
            continue

        match = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$", sql_query[index:])
        if match:
            tag = match.group(0)
            end = sql_query.find(tag, index + len(tag))
            if end == -1:
                out.append(sql_query[index:])
                break
            out.append(sql_query[index : end + len(tag)])
            index = end + len(tag)
            continue

        out.append(char)
        index += 1

    return "".join(out)


def _service_sql_for_dataset(sql_query: str, dataset: Optional[str] = None) -> str:
    quoted_sql = _quote_sql_columns_for_dataset(sql_query, dataset)
    return _rewrite_csv_text_numeric_comparisons(quoted_sql, dataset)


def _dataset_runtime(dataset: Optional[str] = None) -> DatasetRuntime:
    name = _normalize_dataset(dataset)
    runtime = _DATASET_RUNTIMES.get(name)
    if runtime is None:
        runtime = DatasetRuntime(
            csv_cache={},
            csv_loaded=False,
            csv_rid_index={},
            global_rid_index={},
            embeddings=None,
            faiss_manager=None,
        )
        _DATASET_RUNTIMES[name] = runtime
    return runtime


#Fixed prompt, user insert QUESTION and CONTEXT_DATA is retrieved 

PROMPT_TEMPLATE ="""
Answer the QUESTION using the provided CONTEXT_DATA.
Return ONLY valid JSON and nothing else.
The entire output MUST be a JSON array.
Each array element MUST be an object with EXACTLY these keys:
 - result: an object representing one output tuple
 - provenance: a Why[X] provenance expression for that tuple
Provenance rules: Each provenance identifier MUST be a string formatted as "<table_name>_<row_number>"
    (e.g., "standings_35").
   - The provenance field MUST be a list of lists of provenance identifiers.
   - Each inner list contains the identifiers that together produce the result tuple.
 JSON schema:
  [{{"result": {{...}}, "provenance": [["t1", "t2"], ["t3"], ...]}}, ...]
  Constraints:
- Do NOT output SQL.
- Do NOT output explanations, comments, markdown, or code fences.
- Do NOT add extra keys.
- If there are no results, return [].
QUESTION:
{question}
CONTEXT_DATA (rows):
{context_data}
Rows in CONTEXT_DATA are identified by their row identifiers (e.g. "region_5").
Use these identifiers to construct provenance.
"""
# EXPLANATION TEMPLATE
EXPLAIN_PROMPT_TEMPLATE = """
You are an expert in SQL query interpretation and database provenance.
You must generate explanations using domain-specific language derived from the database schema.
The explanation must be understandable by someone who understands the application domain, even if they do not understand SQL or provenance theory.

## Database Schema

The following schema describes the meaning of the database tables, columns, and relationships.

{schema_info}

Use the schema to interpret:
- table names
- column names
- relationships between tables

Always explain results using the domain meaning of the schema, not technical SQL terminology.

## Query

{question}

## Query Result with Provenance

The query result and its provenance are:

{answer_with_provenance}

## Original Rows

The following original rows were referenced by the provenance:

{rows_by_id}

## Provenance Concepts

The provenance explains which database rows contributed to producing the result.

Provenance Rules:
- Identifiers like "customer_12" or "orders_45" refer to specific rows from a table.
- Each inner list is one witness set.
- Rows inside the same witness set are combined together to justify the result.
- Different witness sets represent alternative derivations of the same result.
- A witness set should be interpreted as one sufficient explanation for why the result appears.

Your task is to translate this provenance information into a clear explanation using the schema vocabulary.

## Output Structure

Write a natural-language explanation with exactly these sections:

### Query intent
Explain what the query is asking in domain language.

### Result explanation
Explain what the returned result means in domain language.

### Provenance explanation
Explain why this result appears, based on the provenance.

For each witness set:
- identify the rows involved
- explain what each row represents in the domain
- explain how the rows are connected
- explain why together they justify the result

If multiple witness sets exist, explicitly say that multiple independent combinations of rows support the same result.

## Important Rules
- Keep the explanation concise.
- Always use domain language derived from the schema.
- Do not repeat raw SQL syntax.
- Do not explain provenance using symbols like AND, OR, join, or witness set unless briefly needed for clarity.
- Focus on how the records contribute to the result.
- Be concrete and mention relevant values from the rows.
- If the provenance is incomplete or cannot be fully justified from the provided rows, say so explicitly.
- Remember that if the query was asking for a join between two or more tables, the provenance MUST include rows from all the tables involved in the join, and the explanation MUST mention how these rows are connected together to produce the result.
"""
# Retry prompt (only if the first output was not a valid JSON array)
RETRY_SUFFIX = """
REMINDER:
Return ONLY valid JSON and nothing else.
The entire output MUST be a JSON array.
Return the COMPLETE array for the target table again.
Do not return only the corrected row.
No code fences. No commentary. No extra keys.
"""

# =========================
# OpenAI-compatible request schema
# =========================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    dataset: Optional[str] = None
    model_config = ConfigDict(extra="allow")  # allow extra fields, for compatibility with OpenAI's API which can use more fields

class ProvenanceExplainRequest(BaseModel):
    question: str
    answer_json: str
    model: Optional[str] = None
    temperature: Optional[float] = 0.0
    dataset: Optional[str] = None

class UiPlanRequest(BaseModel):
    question: str
    sql: Optional[str] = None
    dataset: Optional[str] = None

class UiLeafPipelineOptions(BaseModel):
    pushdown: bool = False
    iterative: bool = False

class UiRunRequest(BaseModel):
    question: str
    sql: str
    plan: Dict[str, Any]
    leaf_pipeline_choices: Dict[str, str] = Field(default_factory=dict)
    leaf_pipeline_options: Dict[str, UiLeafPipelineOptions] = Field(default_factory=dict)
    temperature: Optional[float] = 0.0
    dataset: Optional[str] = None
# =========================
# Debug state
# =========================
_LAST_DEBUG: Dict[str, Any] = {}
_LAST_EXPLAIN_DEBUG: Dict[str, Any] = {}
# =========================
# Helpers
# =========================

def _get_embeddings(dataset: Optional[str] = None) -> EmbeddingStrategies:
    config = _dataset_config(dataset)
    runtime = _dataset_runtime(config.name)
    if runtime.embeddings is None:
        runtime.embeddings = EmbeddingStrategies(
            model_name=config.emb_model,
            strategy=config.emb_strategy,
            device=EMB_DEVICE,
            batch_size=BATCH_SIZE,
        )
    return runtime.embeddings

def _now() -> int:
    return int(time.time())

def _log_event(event: Dict[str, Any]) -> None:
    event["ts"] = _now()
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def _set_last_debug(**kwargs) -> None:
    global _LAST_DEBUG
    _LAST_DEBUG = {
        **kwargs,
        "updated_at": _now(),
    }

def _set_last_explain_debug(**kwargs) -> None:
    global _LAST_EXPLAIN_DEBUG
    _LAST_EXPLAIN_DEBUG = {
        **kwargs,
        "updated_at": _now(),
    }
def _extract_user_question(messages: List[ChatMessage]) -> str:
    user_msgs = [m.content for m in messages if m.role == "user"]
    return (user_msgs[-1] if user_msgs else "").strip()

def _ollama_generate(model: str, prompt: str, temperature: float) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }
    try:
        r = requests.post(url, json=payload, timeout=OLLAMA_REQUEST_TIMEOUT)
        if not r.ok:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")
        data = r.json()
        return data.get("response", "")
    except Exception as e:
        raise RuntimeError(f"Failed calling Ollama model '{model}' at {url}: {e}")

def _openai_compatible_generate(model: str, prompt: str, temperature: float) -> str:
    if not LLM_API_BASE:
        raise RuntimeError("LLM_API_BASE is required when PLANNER_LLM_PROVIDER=openai")

    url = f"{LLM_API_BASE}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    model_candidates = [model, *LLM_API_MODEL_ALIASES]
    if "/" in model:
        model_candidates.append(model.rsplit("/", 1)[-1])
    model_candidates = list(dict.fromkeys(model_candidates))

    last_error = ""
    for model_candidate in model_candidates:
        payload: Dict[str, Any] = {
            "model": model_candidate,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False,
        }
        r = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=LLM_REQUEST_TIMEOUT,
            verify=LLM_SSL_VERIFY,
        )
        if not r.ok:
            last_error = f"LLM API error {r.status_code} for model '{model_candidate}': {r.text}"
            continue
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM API returned no choices: {data}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            content = choices[0].get("text")
        return content or ""

    raise RuntimeError(
        f"Failed calling LLM model '{model}' at {url}. Tried {model_candidates}. Last error: {last_error}"
    )

def _generate_model(provider: str, model: str, prompt: str, temperature: float) -> str:
    provider = provider.strip().lower()
    if provider in {"openai", "openai-compatible"}:
        try:
            return _openai_compatible_generate(model, prompt, temperature)
        except Exception as primary_error:
            if not PLANNER_LLM_FALLBACK_ENABLED:
                raise
            try:
                output = _ollama_generate(PLANNER_LLM_FALLBACK_MODEL, prompt, temperature)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Primary LLM failed ({primary_error}); Ollama fallback "
                    f"'{PLANNER_LLM_FALLBACK_MODEL}' also failed ({fallback_error})"
                ) from fallback_error
            return output
    if provider == "ollama":
        return _ollama_generate(model, prompt, temperature)
    raise RuntimeError(f"Unsupported model provider: {provider}")

def _is_valid_json_array(text: str) -> Tuple[bool, Optional[str]]:
    """
    Return True if text is a valid JSON array, False otherwise. If False, also return an error message.
    """
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return True, None
        return False, "Root JSON is not an array"
    except Exception as e:
        return False, str(e)

def _extract_json_array_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return the first JSON array embedded in text.
    Models sometimes prepend a short sentence before the requested JSON; the
    evaluator should still parse the array when it is otherwise valid.
    """
    decoder = json.JSONDecoder()
    first_error: Optional[str] = None
    for start, char in enumerate(text):
        if char != "[":
            continue
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            if first_error is None:
                first_error = str(exc)
            continue
        if isinstance(obj, list):
            return text[start : start + end], None

    return None, first_error or "No JSON array found in model output"

def _call_model_with_retry(
    ollama_model: str,
    base_prompt: str,
    temperature: float,
    provider: str = "ollama",
    max_tries: int = 2,
    validator: Optional[Callable[[str], Tuple[bool, Optional[str]]]] = None,
    scorer: Optional[Callable[[str], int]] = None,
    retry_suffix: str = RETRY_SUFFIX,
) -> str:
    """
    Call to the model with the given prompt. If the output is not a valid JSON array, retry up to max_tries times by appending the RETRY_SUFFIX to the prompt.
    """
    prompt = base_prompt
    print("\n========== PROMPT START ==========\n", flush=True)
    print(prompt, flush=True)
    print("\n========== PROMPT END ==========\n", flush=True)
    last = ""
    best = ""
    best_score = -1
    validate = validator or _is_valid_json_array
    for attempt in range(1, max_tries + 1):
        out = _generate_model(provider, ollama_model, prompt, temperature)
        last = out
        ok, err = validate(out)
        score = scorer(out) if scorer is not None else (1 if ok else 0)
        if score > best_score:
            best = out
            best_score = score
        _log_event({
            "type": "attempt",
            "model_provider": provider,
            "ollama_model": ollama_model,
            "attempt": attempt,
            "ok_json_array": ok,
            "error": err,
            "score": score,
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "output": out,
            "output_chars": len(out),
        })
        if ok:
            return out
        prompt = (
            base_prompt
            + "\n\n"
            + f"Previous output failed validation: {err}\n"
            + retry_suffix
        )

    # If every attempt fails, keep the attempt with the most individually valid rows instead of returning a shorter retry.
    return best or last

def _validate_leaf_json_array(
    text: str,
    expected_rows_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[bool, Optional[str]]:
    valid, error, _ = _parse_leaf_json_array_partial(text, expected_rows_by_id)
    return valid, error

def _parse_leaf_json_array_partial(
    text: str,
    expected_rows_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[bool, Optional[str], Optional[List[Dict[str, Any]]]]:
    """
    Validate a leaf JSON array, but keep row items that are individually valid.

    The boolean/error pair is still strict: any malformed or mismatched item
    makes the whole leaf invalid. The parsed output is partial so downstream
    scoring can count rows that the model did copy correctly.
    """
    json_text, extract_error = _extract_json_array_text(text)
    if json_text is None:
        return False, extract_error, None

    obj = json.loads(json_text)

    if not isinstance(obj, list):
        return False, "Root JSON is not an array", None

    valid_items: List[Dict[str, Any]] = []
    accepted_row_ids = set()
    errors: List[str] = []
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            errors.append(f"Item {i} is not an object")
            continue

        if set(item.keys()) != {"row_id", "values"}:
            errors.append(f"Item {i} must have exactly row_id and values")
            continue

        row_id = item["row_id"]
        values = item["values"]

        if not isinstance(row_id, str):
            errors.append(f"Item {i}.row_id must be a string")
            continue

        if row_id in accepted_row_ids:
            errors.append(f"Item {i}.row_id duplicates an earlier valid output row")
            continue

        if expected_rows_by_id is not None and row_id not in expected_rows_by_id:
            errors.append(f"Item {i}.row_id is not present in CONTEXT_DATA")
            continue

        if not isinstance(values, dict):
            errors.append(f"Item {i}.values must be an object")
            continue

        if expected_rows_by_id is not None and values != expected_rows_by_id[row_id]:
            errors.append(f"Item {i}.values does not exactly match CONTEXT_DATA row '{row_id}'")
            continue

        value_rid = values.get("__rid__")
        if value_rid is not None and value_rid != row_id:
            errors.append(f"Item {i}.values.__rid__ does not match row_id")
            continue

        accepted_row_ids.add(row_id)
        valid_items.append(item)

    if errors:
        shown_errors = errors[:10]
        error_text = "; ".join(shown_errors)
        if len(errors) > len(shown_errors):
            error_text += f"; ... and {len(errors) - len(shown_errors)} more"
        return False, error_text, valid_items

    return True, None, valid_items

def _detect_csv_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",|").delimiter
    except csv.Error:
        header = sample.splitlines()[0] if sample else ""
        return "," if header.count(",") > header.count("|") else "|"


def _load_csvs_once(dataset: Optional[str] = None) -> None:
    config = _dataset_config(dataset)
    runtime = _dataset_runtime(config.name)
    if runtime.csv_loaded:
        return

    base = Path(config.csv_dir)
    if not base.exists():
        _log_event({
            "type": "retrieval_error",
            "dataset": config.name,
            "error": f"CSV_DIR not found: {config.csv_dir}",
        })
        runtime.csv_loaded = True
        return

    for p in sorted(base.glob("*.csv")):
        table = p.stem
        id_col = f"{table}_rownum"
        rows: List[Dict[str, Any]] = []
        rid_to_idx: Dict[str, int] = {}
        delimiter = _detect_csv_delimiter(p)
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None or id_col not in reader.fieldnames:
                raise RuntimeError(f"Missing required id column '{id_col}' in {p}")
            for i, r in enumerate(reader):
                r2 = dict(r)
                if None in r2:
                    # extra data not present in the header
                    r2.pop(None, None)
                rid = str(r2.get(id_col, "")).strip()
                if not rid:
                    continue
                r2["__rid__"] = rid
                # len(rows) is robust if any rows is not saved (rid not present), 
                rid_to_idx[rid] = len(rows)
                rows.append(r2)
        runtime.csv_cache[table] = rows
        runtime.csv_rid_index[table] = rid_to_idx

    runtime.csv_loaded = True
    _log_event({
        "type": "retrieval_loaded",
        "dataset": config.name,
        "tables": list(runtime.csv_cache.keys()),
        "csv_dir": config.csv_dir,
    })


def _build_global_rid_index(dataset: Optional[str] = None) -> None:
    _load_csvs_once(dataset)
    runtime = _dataset_runtime(dataset)
    if runtime.global_rid_index:
        return

    for table, rid_map in runtime.csv_rid_index.items():
        for rid, idx in rid_map.items():
            runtime.global_rid_index[rid] = (table, idx)

def _resolve_row_by_rid(rid: str, dataset: Optional[str] = None) -> Optional[Dict[str, Any]]:
    _build_global_rid_index(dataset)
    runtime = _dataset_runtime(dataset)
    hit = runtime.global_rid_index.get(rid)
    if not hit:
        return None

    table, idx = hit
    row = runtime.csv_cache[table][idx]

    return {
        "table": table,
        "rid": rid,
        "row": row,
    }


def _values_equal_for_context(left: Any, right: Any) -> bool:
    return str(left if left is not None else "").strip() == str(right if right is not None else "").strip()


def _resolve_row_by_column_values(
    table: str,
    columns: List[str],
    values: List[Any],
    dataset: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not table or not columns or len(columns) != len(values):
        return None

    _load_csvs_once(dataset)
    runtime = _dataset_runtime(dataset)
    for row in runtime.csv_cache.get(table, []):
        if all(_values_equal_for_context(row.get(col), value) for col, value in zip(columns, values)):
            rid = str(row.get("__rid__") or row.get(f"{table}_rownum") or "").strip()
            if not rid:
                return None
            return {
                "table": table,
                "rid": rid,
                "row": row,
            }

    return None


def _resolve_linked_context_row(
    linked: Dict[str, Any],
    dataset: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    rid = str(linked.get("rid") or linked.get("row_id") or linked.get("rownum_value") or "").strip()
    if rid:
        resolved = _resolve_row_by_rid(rid, dataset)
        if resolved is not None:
            return resolved

    return _resolve_row_by_column_values(
        table=str(linked.get("to_table") or ""),
        columns=[str(col) for col in (linked.get("to_columns") or [])],
        values=list(linked.get("from_values") or linked.get("to_values") or []),
        dataset=dataset,
    )


def _add_correlated_rows_from_metadata(
    ctx: Dict[str, Any],
    metadata: Dict[str, Any],
    dataset: Optional[str] = None,
    remaining: Optional[int] = None,
) -> Tuple[int, List[Dict[str, str]]]:
    if not INCLUDE_CORRELATED_CONTEXT_ROWS:
        return 0, []
    if remaining is not None and remaining <= 0:
        return 0, []

    linked_rows = metadata.get("linked_rows") or []
    if not isinstance(linked_rows, list):
        return 0, []

    added = 0
    additions: List[Dict[str, str]] = []
    for linked in linked_rows:
        if remaining is not None and added >= remaining:
            break
        if not isinstance(linked, dict):
            continue

        resolved = _resolve_linked_context_row(linked, dataset)
        if resolved is None:
            continue

        table = resolved["table"]
        rid = resolved["rid"]
        if table not in ctx and len(ctx) >= MAX_TABLES:
            continue
        if rid in ctx[table]:
            continue

        ctx[table][rid] = resolved["row"]
        added += 1
        additions.append({
            "table": table,
            "rid": rid,
            "source_table": str(metadata.get("table") or ""),
            "source_rid": str(metadata.get("rid") or metadata.get("rownum_value") or ""),
        })

    return added, additions


# =========================
# Retrieval and indexing
# ========================

def _context_to_prompt_text(ctx: Dict[str, Any]) -> str:
    lines = []
    for table, rows in ctx.items():
        for rid, row in rows.items():
            parts = [f"{k}={v}" for k, v in row.items()]
            lines.append(f"{rid} | table={table} | " + " | ".join(parts))
    return "\n".join(lines)


def _get_or_build_faiss(dataset: Optional[str] = None):
    config = _dataset_config(dataset)
    runtime = _dataset_runtime(config.name)
    if runtime.faiss_manager is None:
        runtime.faiss_manager = FaissIndexManager(
            index_folder=config.faiss_index_folder,
            embeddings_factory=lambda: _get_embeddings(config.name),
            csv_cache=runtime.csv_cache,
            load_csvs=lambda: _load_csvs_once(config.name),
            index_set=config.index_set,
            batch_size=BATCH_SIZE,
            emb_model=config.emb_model,
            emb_strategy=config.emb_strategy,
            log_event=_log_event,
        )

    return runtime.faiss_manager.get_or_build()


def _rag_source_identifier(table: str, row_id: str, chunk_id: str) -> str:
    def local_part(value: str) -> str:
        prefix = f"{table}_"
        return value[len(prefix):] if value.startswith(prefix) else value

    return f"rag_{table}_{local_part(row_id)}_{local_part(chunk_id)}"


def _append_correlated_annotation_evidence(
    scored_evidence: List[Dict[str, Any]],
    correlated_additions: List[Dict[str, str]],
) -> None:
    """Include relationally expanded context rows without inventing FAISS scores."""
    scored_row_ids = {
        str(evidence.get("row_id") or "")
        for evidence in scored_evidence
    }
    for addition in correlated_additions:
        row_id = str(addition.get("rid") or "")
        table = str(addition.get("table") or "")
        if not row_id or not table or row_id in scored_row_ids:
            continue
        scored_evidence.append({
            "table": table,
            "row_id": row_id,
            "source_id": f"correlated_{row_id}",
            "source_type": "correlated",
            "retrieval_method": "relational_correlation",
            "source_table": addition.get("source_table") or "",
            "source_row_id": addition.get("source_rid") or "",
        })
        scored_row_ids.add(row_id)


def retrieve_context_data_iterative(
    question: str,
    dataset: Optional[str] = None,
    target_tables: Optional[List[str]] = None,
    include_correlated_rows: bool = True,
    annotation_sink: Optional[List[Dict[str, Any]]] = None,
    annotation_scope: Optional[Dict[str, str]] = None,
    annotation_pipeline: str = "rag",
) -> Dict[str, Any]:
    """
    Iterative retrieval: start with k=RETRIEVER_K, then increase k by RETRIEVER_K
    in each iteration, until no new rows are retrieved or MAX_ITERATIVE_RETRIEVALS is reached.
    """
    config = _dataset_config(dataset)
    runtime = _dataset_runtime(config.name)
    _load_csvs_once(config.name)
    print("\n[ITERATIVE RETRIEVAL] START", flush=True)
    print(f"[ITERATIVE RETRIEVAL] dataset={config.name}", flush=True)
    print(f"[ITERATIVE RETRIEVAL] question={question}", flush=True)
    target_table_set = {
        str(table).strip()
        for table in (target_tables or [])
        if str(table).strip()
    }
    if target_table_set:
        print(f"[ITERATIVE RETRIEVAL] target_tables={sorted(target_table_set)}", flush=True)

    vs = _get_or_build_faiss(config.name)

    ctx: Dict[str, Any] = defaultdict(dict)
    seen_docs = set()
    scored_evidence: List[Dict[str, Any]] = []
    all_correlated_additions: List[Dict[str, str]] = []
    correlated_rows_added = 0
    k = RETRIEVER_K

    for iteration in range(1, MAX_ITERATIVE_RETRIEVALS + 1):
        if annotation_sink is not None:
            scored_docs = vs.similarity_search_with_relevance_scores(question, k=k)
        else:
            scored_docs = [(document, None) for document in vs.similarity_search(question, k=k)]
        new_count = 0
        new_rids = []
        correlated_additions: List[Dict[str, str]] = []

        print(f"\n[ITERATION {iteration}] k={k} | docs_returned={len(scored_docs)}", flush=True)

        for rank, (d, relevance_score) in enumerate(scored_docs, start=1):
            meta = d.metadata or {}
            table = meta.get("table")
            rid = meta.get("rid")

            if not table or not rid:
                continue

            if target_table_set and table not in target_table_set:
                continue

            if rid in seen_docs:
                continue

            idx0 = runtime.csv_rid_index.get(table, {}).get(rid)
            if idx0 is None:
                continue

            if table not in ctx and len(ctx) >= MAX_TABLES:
                continue

            if rid in ctx[table]:
                seen_docs.add(rid)
                continue

            row = runtime.csv_cache[table][idx0]
            ctx[table][rid] = row
            seen_docs.add(rid)
            new_count += 1
            new_rids.append((rank, table, rid))
            if relevance_score is not None:
                chunk_id = str(meta.get("chunk_id") or rid)
                scored_evidence.append({
                    "table": table,
                    "row_id": rid,
                    "chunk_id": chunk_id,
                    "source_id": _rag_source_identifier(table, rid, chunk_id),
                    "source_type": "rag",
                    "rank": rank,
                    "score": float(relevance_score),
                })

            if include_correlated_rows:
                remaining_correlated = max(MAX_CORRELATED_CONTEXT_ROWS - correlated_rows_added, 0)
                added, additions = _add_correlated_rows_from_metadata(
                    ctx,
                    meta,
                    dataset=config.name,
                    remaining=remaining_correlated,
                )
                correlated_rows_added += added
                correlated_additions.extend(additions)
                all_correlated_additions.extend(additions)

        print(f"[ITERATION {iteration}] new_rows_added={new_count}", flush=True)
        if correlated_additions:
            print(
                f"[ITERATION {iteration}] correlated_rows_added={len(correlated_additions)}",
                flush=True,
            )

        for rank, table, rid in new_rids:
            print(f"  + rank={rank:>3} | table={table:<12} | rid={rid}", flush=True)
            print(
                json.dumps(runtime.csv_cache[table][runtime.csv_rid_index[table][rid]], ensure_ascii=False),
                flush=True,
            )

        print(
            f"[ITERATION {iteration}] tables={list(ctx.keys())} | "
            f"total_rows={sum(len(rows) for rows in ctx.values())}",
            flush=True
        )

        ctx_snapshot = {table: dict(rows) for table, rows in ctx.items()}
        new_rows = {
            rid: {
                "rank": rank,
                "table": table,
                "row": runtime.csv_cache[table][runtime.csv_rid_index[table][rid]],
            }
            for rank, table, rid in new_rids
        }

        _log_event({
            "type": "iterative_retrieval",
            "dataset": config.name,
            "iteration": iteration,
            "retriever_k": k,
            "new_rows_added": new_count,
            "new_rids": [
                {"rank": rank, "table": table, "rid": rid}
                for rank, table, rid in new_rids
            ],
            "tables": list(ctx.keys()),
            "new_rows": new_rows,
            "correlated_rows_added": len(correlated_additions),
            "correlated_rows": correlated_additions,
            "context_data": ctx_snapshot,
            "context_rows_total": sum(len(rows) for rows in ctx_snapshot.values()),
            "question": question,
            "target_tables": sorted(target_table_set),
        })

        if new_count == 0:
            print(f"[ITERATION {iteration}] stopping: no new rows added", flush=True)
            break

        k += RETRIEVER_K

    preview = {table: list(rows.keys())[:3] for table, rows in ctx.items()}
    final_rids = {table: list(rows.keys()) for table, rows in ctx.items()}

    print("\n[ITERATIVE RETRIEVAL] FINAL SUMMARY", flush=True)
    for table, rids in final_rids.items():
        print(f"  table={table} | rows={len(rids)}", flush=True)
        for rid in rids:
            print(f"    - {rid}", flush=True)
            print(json.dumps(ctx[table][rid], ensure_ascii=False), flush=True)

    final_context_data = {table: dict(rows) for table, rows in ctx.items()}
    _log_event({
        "type": "iterative_retrieval_final",
        "dataset": config.name,
        "tables": list(ctx.keys()),
        "rows_preview": preview,
        "final_rids": final_rids,
        "correlated_rows_added": correlated_rows_added,
        "context_data": final_context_data,
        "context_rows_total": sum(len(rows) for rows in final_context_data.values()),
        "question": question,
        "target_tables": sorted(target_table_set),
    })

    if annotation_sink is not None:
        _append_correlated_annotation_evidence(
            scored_evidence,
            all_correlated_additions,
        )
        annotation_sink.append({
            "type": "rag_similarity",
            "pipeline": _canonical_pipeline_id(annotation_pipeline),
            "stage": "retrieval",
            "scope": annotation_scope or {"type": "retrieval"},
            "dataset": config.name,
            "embedding_model": config.emb_model,
            "embedding_strategy": config.emb_strategy,
            "retrieval_query": question,
            "metric": "faiss_relevance",
            "interpretation": "higher_is_more_similar",
            "evidence": scored_evidence,
        })

    return dict(ctx)
def _retrieve_context_data(
    question: str,
    dataset: Optional[str] = None,
    annotation_sink: Optional[List[Dict[str, Any]]] = None,
    annotation_pipeline: str = "rag",
) -> Dict[str, Any]:
    config = _dataset_config(dataset)
    runtime = _dataset_runtime(config.name)
    _load_csvs_once(config.name)
    vs = _get_or_build_faiss(config.name)

    if annotation_sink is not None:
        scored_docs = vs.similarity_search_with_relevance_scores(question, k=RETRIEVER_K)
    else:
        scored_docs = [(document, None) for document in vs.similarity_search(question, k=RETRIEVER_K)]

    ctx: Dict[str, Any] = defaultdict(dict)
    scored_evidence: List[Dict[str, Any]] = []
    used = 0
    correlated_rows_added = 0
    correlated_additions: List[Dict[str, str]] = []

    for rank, (d, relevance_score) in enumerate(scored_docs, start=1):
        meta = d.metadata or {}
        table = meta.get("table")
        rid = meta.get("rid")
        if not table or not rid:
            continue

        idx0 = runtime.csv_rid_index.get(table, {}).get(rid)
        if idx0 is None:
            continue
        row = runtime.csv_cache[table][idx0]
        if table not in ctx and len(ctx) >= MAX_TABLES:
            continue

        if rid in ctx[table]:
            continue

        ctx[table][rid] = row
        used += 1
        if relevance_score is not None:
            chunk_id = str(meta.get("chunk_id") or rid)
            scored_evidence.append({
                "table": table,
                "row_id": rid,
                "chunk_id": chunk_id,
                "source_id": _rag_source_identifier(table, rid, chunk_id),
                "source_type": "rag",
                "rank": rank,
                "score": float(relevance_score),
            })

        remaining_correlated = max(MAX_CORRELATED_CONTEXT_ROWS - correlated_rows_added, 0)
        added, additions = _add_correlated_rows_from_metadata(
            ctx,
            meta,
            dataset=config.name,
            remaining=remaining_correlated,
        )
        correlated_rows_added += added
        correlated_additions.extend(additions)

        if used >= MAX_CONTEXT_ROWS:
            break

    ctx = dict(ctx)

    preview = {table: list(rows.keys())[:3] for table, rows in ctx.items()}

    _log_event({
        "type": "retrieval",
        "dataset": config.name,
        "tables": list(ctx.keys()),
        "rows_preview": preview,
        "context_data": ctx,
        "n_rows_used": used,
        "correlated_rows_added": correlated_rows_added,
        "correlated_rows": correlated_additions,
        "question": question,
    })

    if annotation_sink is not None:
        _append_correlated_annotation_evidence(
            scored_evidence,
            correlated_additions,
        )
        annotation_sink.append({
            "type": "rag_similarity",
            "pipeline": _canonical_pipeline_id(annotation_pipeline),
            "stage": "retrieval",
            "scope": {"type": "query"},
            "dataset": config.name,
            "embedding_model": config.emb_model,
            "embedding_strategy": config.emb_strategy,
            "retrieval_query": question,
            "metric": "faiss_relevance",
            "interpretation": "higher_is_more_similar",
            "evidence": scored_evidence,
        })

    return ctx
# =========================
# Planner-only helpers
# =========================

def _dedupe_strings(values: List[Any]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _build_leaf_retrieval_query(
    task: Dict[str, Any],
    include_pushdown: bool = False,
) -> str:
    """
    Build a retrieval query for one leaf task.
    """
    """
    # if the embedding used is sentence-transformers/all-mpnet-base-v2:
    table = str(task.get("table_name") or task.get("table") or "").strip()
    columns = []
    for key in ("select_columns", "join_keys", "group_by_columns", "aggregate_columns", "columns"):
        for col in task.get(key) or []:
            if col not in columns:
                columns.append(col)

    if table:
        query = f"table: {table}"
        if columns:
            query += " | columns: " + ", ".join(columns)
        return query
    """
    # For the default pipeline, keep the broad table-level retrieval baseline.
    # The pushdown pipeline includes local predicates so entity/value tokens reach FAISS.
    table = str(task.get("table_name") or task.get("table") or "").strip()
    if table and not include_pushdown:
        return f"Retrieve relevant rows for table '{table}'"

    predicates = _dedupe_strings(task.get("local_predicates") or [])
    columns = _dedupe_strings(
        [
            *(task.get("select_columns") or []),
            *(task.get("join_keys") or []),
            *(task.get("group_by_columns") or []),
            *(task.get("aggregate_columns") or []),
            *(task.get("columns") or []),
        ]
    )

    parts: List[str] = []
    if table:
        parts.append(f"Retrieve rows from table '{table}'")
    else:
        parts.append("Retrieve relevant rows for this task")

    if predicates:
        parts.append("where " + " and ".join(predicates))
    if columns:
        parts.append("needed columns: " + ", ".join(columns))

    return ". ".join(parts)

def _leaf_rows_by_id(ctx: Dict[str, Any], table_name: str) -> Dict[str, Dict[str, Any]]:
    return dict(ctx.get(table_name) or {})

def _run_leaf_task(
    task: Dict[str, Any],
    ollama_model: str,
    model_provider: str,
    temperature: float,
    ctx: Dict[str, Any],
    retrieval_query: str,
) -> Dict[str, Any]:
    table_name = str(task.get("table_name") or task.get("table") or "").strip()

    expected_rows_by_id = _leaf_rows_by_id(ctx, table_name)

    prompt = build_leaf_prompt(task, ctx, mode="first")
    out_text = _call_model_with_retry(
        ollama_model,
        prompt,
        temperature,
        provider=model_provider,
        max_tries=2,
        validator=lambda text: _validate_leaf_json_array(text, expected_rows_by_id),
        scorer=lambda text: len(_parse_leaf_json_array_partial(text, expected_rows_by_id)[2] or []),
    )

    valid_leaf_output, validation_error, parsed_output = _parse_leaf_json_array_partial(
        out_text,
        expected_rows_by_id,
    )
    parse_error = None if valid_leaf_output else validation_error

    return {
        "table_name": table_name,
        "task": task,
        "retrieval_query": retrieval_query,
        "context_data": ctx,
        "prompt": prompt,
        "output_text": out_text,
        "parsed_output": parsed_output,
        "parse_error": parse_error,
        "valid_leaf_output": valid_leaf_output,
    }

def _run_iterative_join_leaf_task(
    task: Dict[str, Any],
    ollama_model: str,
    model_provider: str,
    temperature: float,
    ctx: Dict[str, Any],
    retrieval_query: str,
    inherited_bindings: Dict[str, List[str]],
    source_row_ids: List[str],
) -> Dict[str, Any]:
    table_name = str(task.get("table_name") or task.get("table") or "").strip()
    expected_rows_by_id = _leaf_rows_by_id(ctx, table_name)
    prompt = build_iterative_join_leaf_prompt(
        task=task,
        ctx=ctx,
        inherited_bindings=inherited_bindings,
        source_row_ids=source_row_ids,
    )
    out_text = _call_model_with_retry(
        ollama_model,
        prompt,
        temperature,
        provider=model_provider,
        max_tries=2,
        validator=lambda text: _validate_leaf_json_array(text, expected_rows_by_id),
        scorer=lambda text: len(_parse_leaf_json_array_partial(text, expected_rows_by_id)[2] or []),
    )
    valid_leaf_output, validation_error, parsed_output = _parse_leaf_json_array_partial(
        out_text,
        expected_rows_by_id,
    )
    parse_error = None if valid_leaf_output else validation_error

    return {
        "table_name": table_name,
        "task": task,
        "retrieval_query": retrieval_query,
        "context_data": ctx,
        "prompt": prompt,
        "output_text": out_text,
        "parsed_output": parsed_output,
        "parse_error": parse_error,
        "valid_leaf_output": valid_leaf_output,
    }
def _run_planner_first(
    sql_query: str,
    ollama_model: str,
    model_provider: str,
    temperature: float,
    dataset: Optional[str] = None,
    pipeline_id: str = PLANNER_ONLY_MODEL_ID,
) -> Dict[str, Any]:
    selected_dataset = _dataset_config(dataset).name
    plan = build_query_plan(sql_query)

    if plan is None:
        raise RuntimeError("build_query_plan returned None")

    if not hasattr(plan, "to_dict") or not hasattr(plan, "leaf_tasks"):
        raise RuntimeError("Invalid plan object returned by build_query_plan")

    plan_dict = plan.to_dict()

    leaf_outputs = []
    for task in plan.leaf_tasks:
        task_dict = asdict(task)
        retrieval_query = _build_leaf_retrieval_query(task_dict)
        print(f"\n--- Running leaf task for table '{task.table_name}' with retrieval query: {retrieval_query}\n", flush=True)

        print("[planner only] calling retrieve_context_data_iterative()", flush=True)
        annotations: List[Dict[str, Any]] = []
        leaf_ctx = retrieve_context_data_iterative(
            retrieval_query,
            dataset=selected_dataset,
            annotation_sink=annotations,
            annotation_scope={"type": "leaf", "table": task.table_name},
            annotation_pipeline=pipeline_id,
        )
        print("[planner only] retrieve_context_data_iterative() returned", flush=True)

        leaf_output = _run_leaf_task(
            task=task_dict,
            ollama_model=ollama_model,
            model_provider=model_provider,
            temperature=temperature,
            ctx=leaf_ctx,
            retrieval_query=retrieval_query,
        )
        leaf_output["annotations"] = annotations
        leaf_outputs.append(leaf_output)

    return {
        "dataset": selected_dataset,
        "sql": sql_query,
        "plan": plan_dict,
        "leaf_outputs": leaf_outputs,
        "annotations": _collect_leaf_annotations(leaf_outputs),
    }

def _manual_leaf_output(
    task: Dict[str, Any],
    ctx: Dict[str, Any],
    retrieval_query: str,
) -> Dict[str, Any]:
    table_name = str(task.get("table_name") or task.get("table") or "").strip()
    rows_by_id = _leaf_rows_by_id(ctx, table_name)
    parsed_output = [
        {
            "row_id": row_id,
            "values": row,
        }
        for row_id, row in rows_by_id.items()
    ]

    return {
        "table_name": table_name,
        "task": task,
        "retrieval_query": retrieval_query,
        "context_data": ctx,
        "prompt": "MANUAL REVIEW MODE",
        "output_text": json.dumps(parsed_output, ensure_ascii=False, indent=2),
        "parsed_output": parsed_output,
        "parse_error": None,
        "valid_leaf_output": True,
    }

def _raw_model_leaf_output(
    task: Dict[str, Any],
    pipeline: str,
    model_id: str,
    model_provider: str,
    temperature: float,
    retrieval_query: str,
    raw_output: str,
) -> Dict[str, Any]:
    table_name = str(task.get("table_name") or task.get("table") or "").strip()
    return {
        "table_name": table_name,
        "task": task,
        "retrieval_query": retrieval_query,
        "context_data": {},
        "prompt": f"{pipeline.upper()} MODE",
        "output_text": raw_output,
        "parsed_output": [],
        "parse_error": "Pipeline returned free-form model output, not leaf rows.",
        "valid_leaf_output": False,
        "pipeline": pipeline,
        "model_id": model_id,
        "model_provider": model_provider,
    }

def _sql_table_leaf_output(
    task: Dict[str, Any],
    dataset: str,
) -> Dict[str, Any]:
    table_name = str(task.get("table_name") or task.get("table") or "").strip()
    _load_csvs_once(dataset)
    runtime = _dataset_runtime(dataset)
    rows = runtime.csv_cache.get(table_name)
    if rows is None:
        raise ValueError(f"SQL table leaf references unknown table: {table_name}")
    parsed_output = [
        {"row_id": str(row.get("__rid__") or f"{table_name}_{index}"), "values": dict(row)}
        for index, row in enumerate(rows, start=1)
    ]
    return {
        "table_name": table_name,
        "task": task,
        "retrieval_query": "",
        "context_data": {},
        "prompt": "SQL TABLE MODE",
        "output_text": "",
        "parsed_output": parsed_output,
        "parse_error": None,
        "valid_leaf_output": True,
    }

def _llm_internal_leaf_output(
    task: Dict[str, Any],
    dataset: str,
    temperature: float,
) -> Dict[str, Any]:
    table_name = str(task.get("table_name") or task.get("table") or "").strip()
    leaf_question = _leaf_question_sql(task)
    answer, raw_output, confidence = _run_llm_internal_query(
        leaf_question,
        dataset,
        temperature,
    )
    parsed_output: List[Dict[str, Any]] = []
    for index, item in enumerate(answer, start=1):
        provenance = item.get("provenance") or []
        row_id = f"{table_name}_internal_{index}"
        if provenance and isinstance(provenance[0], list) and provenance[0]:
            row_id = str(provenance[0][0])
        parsed_output.append({"row_id": row_id, "values": item["result"]})
    return {
        "table_name": table_name,
        "task": task,
        "retrieval_query": "",
        "context_data": {},
        "prompt": "LLM INTERNAL MODE",
        "output_text": raw_output,
        "parsed_output": parsed_output,
        "parse_error": None,
        "valid_leaf_output": True,
        "annotations": [{
            **confidence,
            "scope": {"type": "leaf", "table": table_name},
        }],
    }


def _pipeline_uses_pushdown_retrieval(pipeline: str) -> bool:
    return _canonical_pipeline_id(pipeline) == PLANNER_ONLY_PUSHDOWN_MODEL_ID


def _run_ui_leaf_pipeline(
    task: Dict[str, Any],
    pipeline: str,
    sql_query: str,
    temperature: float,
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    pipeline = _canonical_pipeline_id(pipeline)
    selected_dataset = _dataset_config(dataset).name
    retrieval_query = _build_leaf_retrieval_query(
        task,
        include_pushdown=_pipeline_uses_pushdown_retrieval(pipeline),
    )

    if pipeline == LLM_INTERNAL_PIPELINE_ID:
        return _llm_internal_leaf_output(task, selected_dataset, temperature)
    if pipeline == SQL_TABLE_PIPELINE_ID:
        return _sql_table_leaf_output(task, selected_dataset)

    retrieval_pipelines = {
        "manual",
        RAG_PIPELINE_ID,
        PLANNER_ONLY_MODEL_ID,
        PLANNER_ONLY_PUSHDOWN_MODEL_ID,
        PLANNER_ONLY_EXPLANATION_MODEL_ID,
    }
    if pipeline not in retrieval_pipelines:
        raise ValueError(f"Unsupported UI pipeline: {pipeline}")

    annotations: List[Dict[str, Any]] = []
    ctx = retrieve_context_data_iterative(
        retrieval_query,
        dataset=dataset,
        annotation_sink=annotations,
        annotation_scope={"type": "leaf", "table": str(task.get("table_name") or "")},
        annotation_pipeline=pipeline,
    )
    if pipeline == "manual":
        leaf_output = _manual_leaf_output(task, ctx, retrieval_query)
    else:
        model_id = pipeline
        model_provider = PLANNER_LLM_PROVIDER
        leaf_output = _run_leaf_task(
            task=task,
            ollama_model=MODEL_ROUTING[model_id],
            model_provider=model_provider,
            temperature=temperature,
            ctx=ctx,
            retrieval_query=retrieval_query,
        )
    leaf_output["annotations"] = annotations
    return leaf_output

def _context_row_count(ctx: Dict[str, Any]) -> int:
    total = 0
    for rows in ctx.values():
        if isinstance(rows, dict):
            total += len(rows)
        elif isinstance(rows, list):
            total += len(rows)
    return total

def _context_preview(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Return all retrieved context for display in the UI."""
    return {
        table_name: dict(rows) if isinstance(rows, dict) else list(rows)
        for table_name, rows in ctx.items()
        if isinstance(rows, (dict, list))
    }


def _collect_leaf_annotations(leaf_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    annotations: List[Dict[str, Any]] = []
    for leaf_output in leaf_outputs:
        leaf_annotations = leaf_output.get("annotations")
        if isinstance(leaf_annotations, list):
            annotations.extend(
                annotation for annotation in leaf_annotations if isinstance(annotation, dict)
            )
    return annotations

def _run_ui_leaf_pipeline_with_context(
    task: Dict[str, Any],
    pipeline: str,
    sql_query: str,
    temperature: float,
    retrieval_query: str,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    pipeline = _canonical_pipeline_id(pipeline)

    if pipeline == "manual":
        return _manual_leaf_output(task, ctx, retrieval_query)

    if pipeline == RAG_PIPELINE_ID:
        return _run_leaf_task(
            task=task,
            ollama_model=MODEL_ROUTING[RAG_PIPELINE_ID],
            model_provider=PLANNER_LLM_PROVIDER,
            temperature=temperature,
            ctx=ctx,
            retrieval_query=retrieval_query,
        )

    if pipeline == PLANNER_ONLY_EXPLANATION_MODEL_ID:
        return _run_leaf_task(
            task=task,
            ollama_model=MODEL_ROUTING[PLANNER_ONLY_EXPLANATION_MODEL_ID],
            model_provider=PLANNER_LLM_PROVIDER,
            temperature=temperature,
            ctx=ctx,
            retrieval_query=retrieval_query,
        )

    if pipeline not in {PLANNER_ONLY_MODEL_ID, PLANNER_ONLY_PUSHDOWN_MODEL_ID}:
        raise ValueError(f"Unsupported UI pipeline: {pipeline}")

    return _run_leaf_task(
        task=task,
        ollama_model=MODEL_ROUTING[pipeline],
        model_provider=PLANNER_LLM_PROVIDER,
        temperature=temperature,
        ctx=ctx,
        retrieval_query=retrieval_query,
    )

def _ui_uses_iterative_join_pipeline(
    leaf_tasks: List[Dict[str, Any]],
    leaf_pipeline_choices: Dict[str, str],
) -> bool:
    for task in leaf_tasks:
        if not isinstance(task, dict):
            continue
        table_name = str(task.get("table_name") or task.get("table") or "").strip()
        pipeline = _canonical_pipeline_id(leaf_pipeline_choices.get(table_name))
        if pipeline == ITERATIVE_PIPELINE_MODEL_ID:
            return True
    return False

def _run_ui_iterative_join_pipeline(
    sql_query: str,
    plan: Dict[str, Any],
    temperature: float,
    dataset: str,
    pushdown: bool = False,
) -> Dict[str, Any]:
    def run_leaf(
        task: Dict[str, Any],
        retrieval_query: str,
        inherited_bindings: Dict[str, List[str]],
        source_row_ids: List[str],
    ) -> Dict[str, Any]:
        table_name = str(task.get("table_name") or task.get("table") or "").strip()
        annotations: List[Dict[str, Any]] = []
        ctx = retrieve_context_data_iterative(
            retrieval_query,
            dataset=dataset,
            annotation_sink=annotations,
            annotation_scope={"type": "leaf", "table": table_name},
            annotation_pipeline=ITERATIVE_PIPELINE_MODEL_ID,
        )
        leaf_output = _run_iterative_join_leaf_task(
            task=task,
            ollama_model=MODEL_ROUTING[ITERATIVE_PIPELINE_MODEL_ID],
            model_provider=PLANNER_LLM_PROVIDER,
            temperature=temperature,
            ctx=ctx,
            retrieval_query=retrieval_query,
            inherited_bindings=inherited_bindings,
            source_row_ids=source_row_ids,
        )
        leaf_output["annotations"] = annotations
        return leaf_output

    planner_result = run_iterative_join_pipeline(
        sql_query=sql_query,
        plan=plan,
        dataset=dataset,
        pipeline_id=ITERATIVE_PIPELINE_MODEL_ID,
        run_leaf=run_leaf,
        build_base_retrieval_query=lambda task, iterative_pushdown: _build_leaf_retrieval_query(
            task,
            include_pushdown=pushdown and iterative_pushdown,
        ),
        log_event=_log_event,
    )
    planner_result["annotations"] = _collect_leaf_annotations(
        planner_result.get("leaf_outputs") or []
    )
    return planner_result

def _ui_rows_from_leaf_outputs(
    leaf_outputs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    answer: List[Dict[str, Any]] = []
    rows_by_id: Dict[str, Any] = {}
    errors: List[str] = []

    for leaf in leaf_outputs:
        table_name = leaf.get("table_name", "")
        parse_error = leaf.get("parse_error")
        if parse_error:
            errors.append(f"{table_name}: {parse_error}")

        parsed_output = leaf.get("parsed_output") or []
        if not isinstance(parsed_output, list):
            continue

        for item in parsed_output:
            if not isinstance(item, dict):
                continue
            row_id = item.get("row_id")
            values = item.get("values")
            if not isinstance(row_id, str) or not isinstance(values, dict):
                continue

            rows_by_id[row_id] = {
                "table": table_name,
                "row": values,
                "pipeline": _canonical_pipeline_id(leaf.get("pipeline")),
            }
            answer.append({
                "result": {
                    "table": table_name,
                    "row_id": row_id,
                    **values,
                },
                "provenance": [[row_id]],
            })

    return answer, rows_by_id, errors

def _leaf_question_sql(task: Dict[str, Any]) -> str:
    existing = str(task.get("question_sql") or task.get("sql") or "").strip()
    if existing:
        return existing

    table_name = str(task.get("table_name") or task.get("table") or "").strip()
    if not table_name:
        return "SELECT *;"

    predicates = [
        str(predicate).strip()
        for predicate in task.get("local_predicates") or []
        if str(predicate).strip()
    ]

    sql = f"SELECT * FROM {table_name}"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    return sql + ";"

def _run_ui_ap_explanation(
    sql_query: str,
    plan: Dict[str, Any],
    leaf_outputs: List[Dict[str, Any]],
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    service_sql_query = _service_sql_for_dataset(sql_query, dataset)
    planner_result = {
        "sql": sql_query,
        "plan": plan,
        "leaf_outputs": leaf_outputs,
    }

    with _EXPLANATION_PIPELINE_LOCK:
        explanation_client = ExplanationClient(
            base_url=EXPLANATION_URL,
            post_endpoint=EXPLANATION_ENDPOINT,
            timeout=EXPLANATION_REQUEST_TIMEOUT,
        )

        pipeline_result = run_planner_first_explanation_pipeline(
            sql_query=sql_query,
            planner_result=planner_result,
            bucket_dir=EXPLANATION_BUCKET_DIR,
            explanation_client=explanation_client,
            delimiter=EXPLANATION_CSV_DELIMITER,
            keep_rownum=EXPLANATION_KEEP_ROWNUM,
            service_sql_query=service_sql_query,
        )

    return {
        "scope": "query",
        "query_sql": sql_query,
        "service_sql": service_sql_query,
        "generated_csv_files": pipeline_result["generated_csv_files"],
        "explanation_output": pipeline_result["explanation_output"],
        "response_text": pipeline_result["response_text"],
    }

def _planner_result_from_ui_run(
    sql_query: str,
    plan: Dict[str, Any],
    leaf_outputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "sql": sql_query,
        "plan": plan,
        "leaf_outputs": leaf_outputs,
    }

def _generate_ui_csv_files(
    sql_query: str,
    plan: Dict[str, Any],
    leaf_outputs: List[Dict[str, Any]],
) -> List[str]:
    clean_bucket(EXPLANATION_BUCKET_DIR)
    return planner_result_to_csv_files(
        planner_result=_planner_result_from_ui_run(sql_query, plan, leaf_outputs),
        output_dir=EXPLANATION_BUCKET_DIR,
        delimiter=EXPLANATION_CSV_DELIMITER,
        keep_rownum=EXPLANATION_KEEP_ROWNUM,
    )

def _copy_dataset_csv_files(
    dataset: Optional[str] = None,
    table_names: Optional[set[str]] = None,
) -> List[str]:
    """Stage the selected database's source tables for deterministic AP execution."""
    config = _dataset_config(dataset)
    source_dir = Path(config.csv_dir)
    if not source_dir.is_dir():
        raise RuntimeError(f"Dataset CSV directory not found: {source_dir}")

    clean_bucket(EXPLANATION_BUCKET_DIR)
    copied: List[str] = []
    for source in sorted(source_dir.glob("*.csv")):
        if table_names is not None and source.stem not in table_names:
            continue
        destination = EXPLANATION_BUCKET_DIR / source.name
        shutil.copy2(source, destination)
        copied.append(source.name)
    if not copied:
        raise RuntimeError(f"No CSV tables found for dataset '{config.name}'")
    return copied

def _apply_ui_leaf_pipeline_options(req: UiRunRequest, leaf_tasks: List[Dict[str, Any]]) -> None:
    choices: Dict[str, str] = {}
    for task in leaf_tasks:
        if not isinstance(task, dict):
            continue
        table_name = str(task.get("table_name") or task.get("table") or "").strip()
        pipeline = _canonical_pipeline_id(req.leaf_pipeline_choices.get(table_name))
        if pipeline not in {
            RAG_PIPELINE_ID,
            PLANNER_ONLY_PUSHDOWN_MODEL_ID,
            PLANNER_ONLY_EXPLANATION_MODEL_ID,
            ITERATIVE_PIPELINE_MODEL_ID,
            LLM_INTERNAL_PIPELINE_ID,
            SQL_TABLE_PIPELINE_ID,
        }:
            raise ValueError(f"Unsupported UI pipeline for {table_name}: {pipeline}")
        options = req.leaf_pipeline_options.get(table_name)
        if pipeline == RAG_PIPELINE_ID and options:
            if options.iterative:
                pipeline = ITERATIVE_PIPELINE_MODEL_ID
            elif options.pushdown:
                pipeline = PLANNER_ONLY_PUSHDOWN_MODEL_ID
        choices[table_name] = pipeline
    req.leaf_pipeline_choices = choices

def _run_llm_internal_query(
    question: str,
    dataset: str,
    temperature: float,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    def validate(text: str) -> Tuple[bool, Optional[str]]:
        try:
            _parse_answer_json(text)
            return True, None
        except ValueError as exc:
            return False, str(exc)

    prompt_template = get_internal_knowledge_prompt_template(dataset)
    prompt = prompt_template.format(question=question)
    raw_output = _call_model_with_retry(
        MODEL_ROUTING[LLM_INTERNAL_PIPELINE_ID],
        prompt,
        temperature,
        provider="ollama",
        max_tries=2,
        validator=validate,
    )
    answer = _parse_answer_json(raw_output)
    confidence = _assess_llm_internal_confidence(
        question=question,
        generated_output=answer,
        temperature=temperature,
    )
    return answer, raw_output, confidence


def _assess_llm_internal_confidence(
    question: str,
    generated_output: List[Dict[str, Any]],
    temperature: float,
) -> Dict[str, Any]:
    def parse_assessment(text: str) -> Dict[str, str]:
        decoder = json.JSONDecoder()
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            level = str(value.get("level") or "").strip().lower()
            reason = str(value.get("reason") or "").strip()
            if level in {"low", "medium", "high"} and reason:
                return {"level": level, "reason": reason}
        raise ValueError("Expected a JSON object with level=low|medium|high and a reason")

    def validate_assessment(text: str) -> Tuple[bool, Optional[str]]:
        try:
            parse_assessment(text)
            return True, None
        except ValueError as exc:
            return False, str(exc)

    assessment_prompt = f"""
You are assessing the confidence of an answer produced only from an LLM's
internal knowledge. Evaluate the generated data itself; you do not have access
to retrieval evidence or the source database.

Assign exactly one level:
- high: the generated data contains stable, well-known facts and is internally
  consistent, specific, and supported by coherent provenance identifiers;
- medium: the answer is plausible and mostly consistent, but some values,
  completeness, or provenance details may be uncertain;
- low: the answer appears guessed, contains instance-specific facts that cannot
  be verified from internal knowledge, is inconsistent, or lacks adequate
  support.

An empty result may be rated high only when abstaining is clearly more reliable
than inventing unavailable instance data. This is a self-assessment, not an
externally verified probability.

Return ONLY this JSON object:
{{"level":"low|medium|high","reason":"one concise sentence"}}

QUESTION:
{question}

GENERATED OUTPUT:
{json.dumps(generated_output, ensure_ascii=False)}
"""
    raw_assessment = _call_model_with_retry(
        MODEL_ROUTING[LLM_INTERNAL_PIPELINE_ID],
        assessment_prompt,
        temperature,
        provider="ollama",
        max_tries=2,
        validator=validate_assessment,
        retry_suffix=(
            "Return ONLY one JSON object with exactly the keys level and reason. "
            "The level must be low, medium, or high."
        ),
    )
    assessment = parse_assessment(raw_assessment)
    return {
        "type": "llm_confidence",
        "pipeline": LLM_INTERNAL_PIPELINE_ID,
        "stage": "generation",
        "level": assessment["level"],
        "reason": assessment["reason"],
        "assessment_method": "llm_self_assessment",
        "model": MODEL_ROUTING[LLM_INTERNAL_PIPELINE_ID],
    }

def _run_ui_ap_explanation_for_csv_files(
    sql_query: str,
    csv_files: List[str],
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    service_sql_query = _service_sql_for_dataset(sql_query, dataset)
    explanation_client = ExplanationClient(
        base_url=EXPLANATION_URL,
        post_endpoint=EXPLANATION_ENDPOINT,
        timeout=EXPLANATION_REQUEST_TIMEOUT,
    )
    explanation_output = explanation_client.run_explanation(
        sql_query=service_sql_query,
        csv_files=csv_files,
        delimiter=EXPLANATION_CSV_DELIMITER,
    )
    generated_files_markdown = "\n".join(f"- `{name}`" for name in csv_files) or "No CSV files generated."
    response_text = (
        "### Full Pipeline Result\n\n"
        "#### Query\n\n"
        f"```sql\n{sql_query}\n```\n\n"
        "#### Generated CSV files\n\n"
        f"{generated_files_markdown}\n\n"
        "#### Explanation Service Output\n\n"
        "```json\n"
        f"{json.dumps(explanation_output, ensure_ascii=False, indent=2)}\n"
        "```\n"
    )
    return {
        "scope": "query",
        "query_sql": sql_query,
        "service_sql": service_sql_query,
        "generated_csv_files": csv_files,
        "explanation_output": explanation_output,
        "response_text": response_text,
    }

def _answer_from_ap_explanation(explanation: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = explanation.get("explanation_output")
    if not isinstance(output, dict):
        return []

    result = output.get("result")
    if not isinstance(result, dict):
        return []

    derivations = result.get("derivations")
    if not isinstance(derivations, list):
        return []

    answer: List[Dict[str, Any]] = []
    for derivation in derivations:
        if not isinstance(derivation, dict):
            continue
        row = derivation.get("answer")
        if not isinstance(row, dict):
            continue
        answer.append({
            "result": row,
            "provenance": derivation.get("provenance"),
        })
    return answer


def _provenance_row_ids(provenance: Any, known_row_ids: set[str]) -> List[str]:
    found: List[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            for row_id in known_row_ids:
                if row_id in found:
                    continue
                if value == row_id or re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(row_id)}(?![A-Za-z0-9_])",
                    value,
                ):
                    found.append(row_id)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(provenance)
    return found


def _retrieval_confidence_band(score: float) -> str:
    if score >= 0.75:
        return "close"
    if score >= 0.50:
        return "relevant"
    if score >= 0.25:
        return "weak"
    return "far"


def _annotate_final_answer(
    answer: List[Dict[str, Any]],
    leaf_outputs: List[Dict[str, Any]],
    rows_by_id: Dict[str, Any],
) -> List[Dict[str, Any]]:
    leaf_annotations = _collect_leaf_annotations(leaf_outputs)
    known_row_ids = set(rows_by_id)
    rag_evidence: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    llm_annotations_by_table: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for annotation in leaf_annotations:
        annotation_type = annotation.get("type")
        if annotation_type == "rag_similarity":
            for evidence in annotation.get("evidence") or []:
                if isinstance(evidence, dict) and isinstance(evidence.get("row_id"), str):
                    rag_evidence[evidence["row_id"]].append({
                        **evidence,
                        "pipeline": annotation.get("pipeline"),
                        "embedding_model": annotation.get("embedding_model"),
                    })
        elif annotation_type == "llm_confidence":
            table_name = str((annotation.get("scope") or {}).get("table") or "")
            llm_annotations_by_table[table_name].append(annotation)

    level_rank = {"low": 0, "medium": 1, "high": 2}
    for result_index, item in enumerate(answer, start=1):
        if not isinstance(item, dict):
            continue
        row_ids = _provenance_row_ids(item.get("provenance"), known_row_ids)
        if not row_ids:
            row_ids = list(known_row_ids)

        source_rows = [rows_by_id[row_id] for row_id in row_ids if row_id in rows_by_id]
        source_pipelines = sorted({
            str(source.get("pipeline") or "")
            for source in source_rows
            if source.get("pipeline")
        })
        confidence_annotations: List[Dict[str, Any]] = []

        contributing_rag = [
            evidence
            for row_id in row_ids
            for evidence in rag_evidence.get(row_id, [])
            if isinstance(evidence.get("score"), (int, float))
        ]
        if contributing_rag:
            minimum_score = min(float(evidence["score"]) for evidence in contributing_rag)
            confidence_annotations.append({
                "type": "final_answer_confidence",
                "source_type": "rag_retrieval",
                "pipelines": sorted({
                    str(evidence.get("pipeline") or RAG_PIPELINE_ID)
                    for evidence in contributing_rag
                }),
                "metric": "faiss_relevance",
                "score": minimum_score,
                "level": _retrieval_confidence_band(minimum_score),
                "aggregation": "minimum_contributing_score",
                "source_row_ids": sorted({
                    str(evidence["row_id"]) for evidence in contributing_rag
                }),
                "source_ids": sorted({
                    str(evidence["source_id"])
                    for evidence in contributing_rag
                    if evidence.get("source_id")
                }),
                "reason": "The final result uses the weakest relevance score among its contributing retrieved rows.",
                "scope": {"type": "result", "index": result_index},
            })

        llm_tables = {
            str(source.get("table") or "")
            for source in source_rows
            if source.get("pipeline") == LLM_INTERNAL_PIPELINE_ID
        }
        contributing_llm = [
            annotation
            for table_name in llm_tables
            for annotation in llm_annotations_by_table.get(table_name, [])
        ]
        if contributing_llm:
            lowest = min(
                contributing_llm,
                key=lambda annotation: level_rank.get(str(annotation.get("level")), -1),
            )
            confidence_annotations.append({
                "type": "final_answer_confidence",
                "source_type": "llm_internal",
                "pipelines": [LLM_INTERNAL_PIPELINE_ID],
                "level": lowest.get("level"),
                "aggregation": "lowest_contributing_level",
                "reason": lowest.get("reason"),
                "assessment_method": "llm_self_assessment",
                "source_tables": sorted(llm_tables),
                "scope": {"type": "result", "index": result_index},
            })

        sql_rows = [
            row_id for row_id in row_ids
            if (rows_by_id.get(row_id) or {}).get("pipeline") == SQL_TABLE_PIPELINE_ID
        ]
        if sql_rows:
            confidence_annotations.append({
                "type": "final_answer_confidence",
                "source_type": "deterministic_sql",
                "pipelines": [SQL_TABLE_PIPELINE_ID],
                "level": "verified_execution",
                "source_row_ids": sql_rows,
                "reason": "These contributing rows were read deterministically from the selected table; this is not a probability of source-data correctness.",
                "scope": {"type": "result", "index": result_index},
            })

        item["source_pipelines"] = source_pipelines
        item["confidence_annotations"] = confidence_annotations

    return answer


# =========================
# Provenance helpers
# =========================
def _parse_answer_json(text: str) -> List[Dict[str, Any]]:
    array_text, extraction_error = _extract_json_array_text(text)
    if array_text is None:
        raise ValueError(f"Invalid JSON: {extraction_error}")
    try:
        obj = json.loads(array_text)
    except Exception as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(obj, list):
        raise ValueError("Model output is not a JSON array")

    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not an object")

        if set(item.keys()) != {"result", "provenance"}:
            raise ValueError(f"Item {i} must have exactly these keys: result, provenance")

        if not isinstance(item["result"], dict):
            raise ValueError(f"Item {i}.result must be an object")

        prov = item["provenance"]
        if not isinstance(prov, list):
            raise ValueError(f"Item {i}.provenance must be a list")

        for j, ws in enumerate(prov):
            if not isinstance(ws, list):
                raise ValueError(f"Item {i}.provenance[{j}] must be a list")
            for k, rid in enumerate(ws):
                if not isinstance(rid, str):
                    raise ValueError(f"Item {i}.provenance[{j}][{k}] must be a string")

    return obj

def _display_value(value: Any) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    if len(text) > GATEWAY_MARKDOWN_MAX_CELL_CHARS:
        text = text[: GATEWAY_MARKDOWN_MAX_CELL_CHARS - 3].rstrip() + "..."
    return text

def _markdown_cell(value: Any) -> str:
    return _display_value(value).replace("|", "\\|")

def _markdown_code(value: str) -> str:
    return "`" + value.replace("`", "\\`") + "`"

def _render_markdown_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No rows."

    columns: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    visible_rows = rows[:GATEWAY_MARKDOWN_MAX_ROWS]
    header = "| " + " | ".join(_markdown_cell(col) for col in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_markdown_cell(row.get(col, "")) for col in columns) + " |"
        for row in visible_rows
    ]

    table = "\n".join([header, separator, *body])
    omitted = len(rows) - len(visible_rows)
    if omitted > 0:
        table += f"\n\nShowing {len(visible_rows)} of {len(rows)} rows."
    return table

def _result_label(result: Dict[str, Any]) -> str:
    if not result:
        return "{}"
    parts = [f"{key}={_display_value(value)}" for key, value in result.items()]
    return ", ".join(parts)

def _provenance_label(provenance: Any) -> str:
    if not isinstance(provenance, list) or not provenance:
        return ""

    witness_labels: List[str] = []
    for witness_set in provenance:
        if isinstance(witness_set, list):
            witness_labels.append(" + ".join(_markdown_code(str(rid)) for rid in witness_set))
        else:
            witness_labels.append(_markdown_code(str(witness_set)))
    return "<br>".join(witness_labels)

def _raw_json_details(raw_payload: Any, summary: str = "Raw JSON") -> str:
    if not GATEWAY_MARKDOWN_INCLUDE_RAW:
        return ""
    raw_text = json.dumps(raw_payload, ensure_ascii=False, indent=2)
    return f"\n\n<details>\n<summary>{summary}</summary>\n\n```json\n{raw_text}\n```\n\n</details>"

def _render_answer_items_markdown(answer_items: List[Dict[str, Any]]) -> str:
    if not answer_items:
        return "### Answer\n\nNo results found.\n\n### Provenance\n\nNo provenance rows."

    result_rows = [
        item.get("result", {})
        for item in answer_items
        if isinstance(item, dict) and isinstance(item.get("result"), dict)
    ]
    provenance_rows = [
        {
            "result": _result_label(item.get("result", {})),
            "supporting_rows": _provenance_label(item.get("provenance")),
        }
        for item in answer_items
        if isinstance(item, dict)
    ]

    markdown = (
        "### Answer\n\n"
        + _render_markdown_table(result_rows)
        + "\n\n### Provenance\n\n"
        + _render_markdown_table(provenance_rows)
    )
    return markdown + _raw_json_details(answer_items)

def _render_planner_first_markdown(planner_result: Dict[str, Any]) -> str:
    plan = planner_result.get("plan") or {}
    leaf_outputs = planner_result.get("leaf_outputs") or []
    leaf_rows: List[Dict[str, Any]] = []

    for index, leaf in enumerate(leaf_outputs):
        parsed = leaf.get("parsed_output")
        if isinstance(parsed, list):
            output_rows = len(parsed)
        else:
            output_rows = 0

        context_data = leaf.get("context_data") or {}
        context_rows = sum(len(rows) for rows in context_data.values() if isinstance(rows, dict))
        leaf_rows.append({
            "leaf": index + 1,
            "table": leaf.get("table_name", ""),
            "retrieval_query": leaf.get("retrieval_query", ""),
            "context_rows": context_rows,
            "output_rows": output_rows,
            "valid": leaf.get("valid_leaf_output", False),
            "parse_error": leaf.get("parse_error") or "",
        })

    joins = plan.get("joins") or []
    post_ops = plan.get("post_ops") or []
    summary_rows = [{
        "query_type": plan.get("query_type", ""),
        "leaf_tasks": len(leaf_outputs),
        "joins": len(joins),
        "post_ops": len(post_ops),
    }]

    markdown = (
        "### Planner Only Result\n\n"
        + _render_markdown_table(summary_rows)
        + "\n\n### Leaf Tasks\n\n"
        + _render_markdown_table(leaf_rows)
    )

    return markdown + _raw_json_details(planner_result, summary="Raw planner output")

def _render_chat_response(raw_text: str, parsed_payload: Optional[Any] = None) -> str:
    if GATEWAY_RESPONSE_FORMAT not in {"markdown", "md", "ui"}:
        return raw_text

    payload = parsed_payload
    if payload is None:
        try:
            payload = json.loads(raw_text)
        except Exception:
            array_text, _ = _extract_json_array_text(raw_text)
            if array_text is not None:
                try:
                    payload = json.loads(array_text)
                except Exception:
                    payload = None

    try:
        if isinstance(payload, list):
            return _render_answer_items_markdown(payload)
        if isinstance(payload, dict) and "leaf_outputs" in payload:
            return _render_planner_first_markdown(payload)
    except Exception as e:
        _log_event({
            "type": "markdown_render_error",
            "error": str(e),
            "raw_text": raw_text,
        })

    return raw_text

def _collect_provenance_rows(
    answer_items: List[Dict[str, Any]],
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    rows_by_id: Dict[str, Any] = {}
    missing_ids: List[str] = []

    for item in answer_items:
        for witness_set in item["provenance"]:
            for rid in witness_set:
                if rid in rows_by_id:
                    continue
                resolved = _resolve_row_by_rid(rid, dataset=dataset)
                if resolved is None:
                    missing_ids.append(rid)
                else:
                    rows_by_id[rid] = resolved

    return {
        "rows_by_id": rows_by_id,
        "missing_ids": sorted(set(missing_ids)),
    }

def _validate_provenance_structure(
    answer_items: List[Dict[str, Any]],
    rows_by_id: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []

    for i, item in enumerate(answer_items):
        prov = item["provenance"]
        seen_ws = set()

        for j, ws in enumerate(prov):
            if len(ws) == 0:
                errors.append(f"result[{i}] witness_set[{j}] is empty")

            if len(ws) != len(set(ws)):
                errors.append(f"result[{i}] witness_set[{j}] contains duplicate row ids")

            for rid in ws:
                if rid not in rows_by_id:
                    errors.append(f"result[{i}] witness_set[{j}] references unknown row id '{rid}'")

            ws_key = tuple(sorted(ws))
            if ws_key in seen_ws:
                errors.append(f"result[{i}] contains duplicate witness set {ws}")
            seen_ws.add(ws_key)

    return errors

def _build_schema_subset_from_rows(
    rows_by_id: Dict[str, Any],
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    schema_info = _dataset_config(dataset).schema_info
    tables = sorted({info["table"] for info in rows_by_id.values()})
    return {t: schema_info[t] for t in tables if t in schema_info}

def _provenance_to_formula(prov: List[List[str]]) -> str:
    parts = []
    for ws in prov:
        if len(ws) == 1:
            parts.append(ws[0])
        else:
            parts.append("(" + " AND ".join(ws) + ")")
    return " OR ".join(parts)

def _explain_provenance_with_model(
    question: str,
    answer_items: List[Dict[str, Any]],
    rows_by_id: Dict[str, Any],
    temperature: float = 0.0,
    explainer_model: Optional[str] = None,
    dataset: Optional[str] = None,
) -> str:
    schema_info = _build_schema_subset_from_rows(rows_by_id, dataset=dataset)
    model_name = explainer_model or EXPLAIN_MODEL

    prompt = EXPLAIN_PROMPT_TEMPLATE.format(
        schema_info=json.dumps(schema_info, ensure_ascii=False, indent=2),
        question=question,
        answer_with_provenance=json.dumps(answer_items, ensure_ascii=False, indent=2),
        rows_by_id=json.dumps(rows_by_id, ensure_ascii=False, indent=2),
    )

    last = ""
    for attempt in range(1, EXPLAIN_MAX_TRIES + 1):
        print("\n========== EXPLAIN PROMPT START ==========\n", flush=True)
        print(prompt, flush=True)
        print("\n========== EXPLAIN PROMPT END ==========\n", flush=True)
        out = _ollama_generate(model_name, prompt, temperature)
        last = out.strip()
        _log_event({
            "type": "explain_attempt",
            "model": model_name,
            "attempt": attempt,
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "output": last,
            "output_chars": len(last),
            "empty": not bool(last),
        })
        if last:
            return last

        _log_event({
            "type": "explain_attempt_empty",
            "model": model_name,
            "attempt": attempt,
        })

    return last

# =========================
# Routes (OpenAI-compatible: OpenWebUI has already an OPENAI-compatible backend, so we can just mimic that) 
# =========================

UI_DIR = Path(__file__).resolve().parent / "ui"

def _ui_file_response(path: Path) -> FileResponse:
    return FileResponse(
        path,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )

@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")

@app.get("/ui", response_class=HTMLResponse)
@app.get("/ui/", response_class=HTMLResponse)
def ui_index() -> FileResponse:
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI index.html not found")
    return _ui_file_response(index_path)

@app.get("/ui/{asset_name}")
def ui_asset(asset_name: str) -> Any:
    if asset_name == "csv":
        return ui_csv_files()

    if asset_name not in {"app.js", "styles.css"}:
        raise HTTPException(status_code=404, detail="UI asset not found")

    asset_path = UI_DIR / asset_name
    if not asset_path.exists():
        raise HTTPException(status_code=404, detail=f"UI asset not found: {asset_name}")
    return _ui_file_response(asset_path)

@app.get("/ui/csv", response_class=HTMLResponse)
def ui_csv_files() -> HTMLResponse:
    rows = []
    if EXPLANATION_BUCKET_DIR.exists():
        for path in sorted(EXPLANATION_BUCKET_DIR.glob("*.csv")):
            try:
                preview = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                preview = f"Could not read {path.name}: {e}"
            rows.append(
                "<section class='csv-card'>"
                f"<h2>{html.escape(path.name)}</h2>"
                f"<pre>{html.escape(preview)}</pre>"
                "</section>"
            )

    body = "\n".join(rows) if rows else "<p class='empty'>No generated CSV files yet.</p>"
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Generated CSV files</title>
    <link rel="stylesheet" href="/ui/styles.css?v=flow-6" />
  </head>
  <body>
    <main class="csv-page">
      <header class="csv-header">
        <div>
          <p class="eyebrow">Generated Data</p>
          <h1>CSV files produced by the selected pipelines</h1>
        </div>
        <a class="button-link" href="/ui">Back to workspace</a>
      </header>
      """
        + body
        + """
    </main>
  </body>
</html>
        """
    )

@app.post("/ui/plan")
def ui_plan(req: UiPlanRequest) -> Dict[str, Any]:
    sql_query = (req.sql or req.question).strip()
    if not sql_query:
        raise HTTPException(status_code=400, detail="Empty SQL query")

    try:
        dataset = _dataset_config(req.dataset).name
        plan = build_query_plan(sql_query)
        if plan is None:
            raise RuntimeError("build_query_plan returned None")

        plan_dict = plan.to_dict()
        for task in plan_dict.get("leaf_tasks", []):
            if isinstance(task, dict) and not task.get("question_sql"):
                task["question_sql"] = _leaf_question_sql(task)

        _log_event({
            "type": "ui_plan",
            "dataset": dataset,
            "sql_query": sql_query,
            "leaf_tasks": len(plan_dict.get("leaf_tasks", [])),
        })
        return {
            "source": "gateway",
            "dataset": dataset,
            "sql": sql_query,
            "plan": plan_dict,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"UI planning failed: {e}")

@app.post("/ui/run")
def ui_run(req: UiRunRequest) -> Dict[str, Any]:
    sql_query = req.sql.strip() or req.question.strip()
    if not sql_query:
        raise HTTPException(status_code=400, detail="Empty SQL query")

    leaf_tasks = req.plan.get("leaf_tasks") or []
    if not isinstance(leaf_tasks, list):
        raise HTTPException(status_code=400, detail="plan.leaf_tasks must be a list")

    temperature = 0.0 if req.temperature is None else float(req.temperature)
    leaf_outputs: List[Dict[str, Any]] = []
    pipeline_choices: List[Dict[str, str]] = []

    try:
        dataset = _dataset_config(req.dataset).name
        _apply_ui_leaf_pipeline_options(req, leaf_tasks)
        if _ui_uses_iterative_join_pipeline(leaf_tasks, req.leaf_pipeline_choices):
            print("\n[UI] Running iterative join-aware query pipeline\n", flush=True)
            planner_result = _run_ui_iterative_join_pipeline(
                sql_query=sql_query,
                plan=req.plan,
                temperature=temperature,
                dataset=dataset,
                pushdown=any(
                    options.iterative and options.pushdown
                    for options in req.leaf_pipeline_options.values()
                ),
            )
            leaf_outputs = planner_result.get("leaf_outputs") or []
            pipeline_choices = [
                {
                    "table": str(leaf.get("table_name") or ""),
                    "pipeline": ITERATIVE_PIPELINE_MODEL_ID,
                }
                for leaf in leaf_outputs
            ]
        else:
            for task in leaf_tasks:
                if not isinstance(task, dict):
                    continue

                table_name = str(task.get("table_name") or task.get("table") or "").strip()
                pipeline = _canonical_pipeline_id(req.leaf_pipeline_choices.get(table_name))
                print(
                    f"\n[UI] Running table '{table_name}' with pipeline '{pipeline}'\n",
                    flush=True,
                )
                leaf_output = _run_ui_leaf_pipeline(
                    task=task,
                    pipeline=pipeline,
                    sql_query=sql_query,
                    temperature=temperature,
                    dataset=dataset,
                )
                leaf_output["pipeline"] = pipeline
                leaf_outputs.append(leaf_output)
                pipeline_choices.append({
                    "table": table_name,
                    "pipeline": pipeline,
                })

        _leaf_answer, rows_by_id, errors = _ui_rows_from_leaf_outputs(leaf_outputs)
        explanations: List[Dict[str, Any]] = []
        answer: List[Dict[str, Any]] = []
        try:
            explanation = _run_ui_ap_explanation(
                sql_query=sql_query,
                plan=req.plan,
                leaf_outputs=leaf_outputs,
                dataset=dataset,
            )
            explanations.append(explanation)
            answer = _answer_from_ap_explanation(explanation)
            answer = _annotate_final_answer(answer, leaf_outputs, rows_by_id)
        except Exception as e:
            errors.append(f"ap-explanation: {e}")
            _log_event({
                "type": "ui_ap_explanation_error",
                "sql_query": sql_query,
                "error": str(e),
            })

        result = {
            "source": "gateway",
            "dataset": dataset,
            "sql": sql_query,
            "plan": req.plan,
            "answer": answer,
            "final_answer_annotations": [
                annotation
                for item in answer
                for annotation in item.get("confidence_annotations") or []
            ],
            "rows_by_id": rows_by_id,
            "leaf_answer": _leaf_answer,
            "leaf_outputs": leaf_outputs,
            "annotations": _collect_leaf_annotations(leaf_outputs),
            "pipeline_choices": pipeline_choices,
            "errors": errors,
            "explanations": explanations,
            "csv_page_url": "/ui/csv",
            "csv_ready": bool(explanations),
        }

        _log_event({
            "type": "ui_run",
            "dataset": dataset,
            "sql_query": sql_query,
            "pipeline_choices": pipeline_choices,
            "answer_rows": len(answer),
            "explanations": len(explanations),
            "errors": errors,
        })
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"UI run failed: {e}")

@app.post("/ui/run/stream")
def ui_run_stream(req: UiRunRequest) -> StreamingResponse:
    sql_query = req.sql.strip() or req.question.strip()
    if not sql_query:
        raise HTTPException(status_code=400, detail="Empty SQL query")

    leaf_tasks = req.plan.get("leaf_tasks") or []
    if not isinstance(leaf_tasks, list):
        raise HTTPException(status_code=400, detail="plan.leaf_tasks must be a list")

    temperature = 0.0 if req.temperature is None else float(req.temperature)
    dataset = _dataset_config(req.dataset).name
    _apply_ui_leaf_pipeline_options(req, leaf_tasks)

    def encode_event(event_type: str, payload: Dict[str, Any]) -> str:
        return json.dumps(
            {
                "type": event_type,
                **payload,
            },
            ensure_ascii=False,
        ) + "\n"

    def event_stream():
        leaf_outputs: List[Dict[str, Any]] = []
        pipeline_choices: List[Dict[str, str]] = []
        explanations: List[Dict[str, Any]] = []
        errors: List[str] = []
        rows_by_id: Dict[str, Any] = {}
        leaf_answer: List[Dict[str, Any]] = []
        answer: List[Dict[str, Any]] = []
        generated_csv_files: List[str] = []

        pending_events: List[str] = []

        def emit(event_type: str, payload: Dict[str, Any]) -> None:
            pending_events.append(encode_event(event_type, payload))

        try:
            yield encode_event("start", {
                "message": "Running selected leaf pipelines.",
                "dataset": dataset,
                "leaf_count": len(leaf_tasks),
            })

            if _ui_uses_iterative_join_pipeline(leaf_tasks, req.leaf_pipeline_choices):
                yield encode_event("leaf_start", {
                    "table": "",
                    "pipeline": ITERATIVE_PIPELINE_MODEL_ID,
                    "index": 1,
                    "message": "Running iterative join-aware query pipeline.",
                })
                planner_result = _run_ui_iterative_join_pipeline(
                    sql_query=sql_query,
                    plan=req.plan,
                    temperature=temperature,
                    dataset=dataset,
                    pushdown=any(
                        options.iterative and options.pushdown
                        for options in req.leaf_pipeline_options.values()
                    ),
                )
                leaf_outputs = planner_result.get("leaf_outputs") or []
                for index, leaf_output in enumerate(leaf_outputs):
                    table_name = str(leaf_output.get("table_name") or "")
                    iterative_meta = leaf_output.get("iterative_join") or {}
                    pipeline_choices.append({
                        "table": table_name,
                        "pipeline": ITERATIVE_PIPELINE_MODEL_ID,
                    })
                    ctx = leaf_output.get("context_data") or {}
                    parsed_rows = leaf_output.get("parsed_output") or []
                    yield encode_event("leaf_context", {
                        "table": table_name,
                        "pipeline": ITERATIVE_PIPELINE_MODEL_ID,
                        "message": f"Iterative context retrieved for {table_name}.",
                        "rows": _context_row_count(ctx),
                        "tables": sorted(ctx.keys()),
                        "retrieval_query": leaf_output.get("retrieval_query", ""),
                        "iterative_step": iterative_meta.get("step", index + 1),
                        "inherited_bindings": iterative_meta.get("inherited_bindings") or {},
                        "source_row_ids": iterative_meta.get("source_row_ids") or [],
                        "source_row_summaries": iterative_meta.get("source_row_summaries") or [],
                        "context_preview": _context_preview(ctx),
                        "annotations": leaf_output.get("annotations") or [],
                    })
                    yield encode_event("leaf_done", {
                        "table": table_name,
                        "pipeline": ITERATIVE_PIPELINE_MODEL_ID,
                        "iterative_step": iterative_meta.get("step", index + 1),
                        "inherited_bindings": iterative_meta.get("inherited_bindings") or {},
                        "source_row_ids": iterative_meta.get("source_row_ids") or [],
                        "source_row_summaries": iterative_meta.get("source_row_summaries") or [],
                        "rows": len(parsed_rows) if isinstance(parsed_rows, list) else 0,
                        "message": f"{table_name} iterative leaf step completed.",
                    })
            else:
                for index, task in enumerate(leaf_tasks):
                    if not isinstance(task, dict):
                        continue

                    table_name = str(task.get("table_name") or task.get("table") or "").strip()
                    pipeline = _canonical_pipeline_id(req.leaf_pipeline_choices.get(table_name))
                    pipeline_choices.append({
                        "table": table_name,
                        "pipeline": pipeline,
                    })
                    yield encode_event("leaf_start", {
                        "table": table_name,
                        "pipeline": pipeline,
                        "index": index + 1,
                        "message": f"Running {table_name} with {pipeline}.",
                    })

                    if pipeline in {LLM_INTERNAL_PIPELINE_ID, SQL_TABLE_PIPELINE_ID}:
                        yield encode_event("leaf_context", {
                            "table": table_name,
                            "pipeline": pipeline,
                            "message": (
                                "No retrieval needed; using internal LLM knowledge."
                                if pipeline == LLM_INTERNAL_PIPELINE_ID
                                else "No retrieval needed; loading the complete source table."
                            ),
                            "rows": 0,
                        })
                        leaf_output = _run_ui_leaf_pipeline(
                            task=task,
                            pipeline=pipeline,
                            sql_query=sql_query,
                            temperature=temperature,
                            dataset=dataset,
                        )
                    else:
                        retrieval_query = _build_leaf_retrieval_query(
                            task,
                            include_pushdown=_pipeline_uses_pushdown_retrieval(pipeline),
                        )
                        annotations: List[Dict[str, Any]] = []
                        ctx = retrieve_context_data_iterative(
                            retrieval_query,
                            dataset=dataset,
                            annotation_sink=annotations,
                            annotation_scope={"type": "leaf", "table": table_name},
                            annotation_pipeline=pipeline,
                        )
                        yield encode_event("leaf_context", {
                            "table": table_name,
                            "pipeline": pipeline,
                            "message": f"Context retrieved for {table_name}.",
                            "rows": _context_row_count(ctx),
                            "tables": sorted(ctx.keys()),
                            "retrieval_query": retrieval_query,
                            "context_preview": _context_preview(ctx),
                            "annotations": annotations or [],
                        })
                        leaf_output = _run_ui_leaf_pipeline_with_context(
                            task=task,
                            pipeline=pipeline,
                            sql_query=sql_query,
                            temperature=temperature,
                            retrieval_query=retrieval_query,
                            ctx=ctx,
                        )
                        if annotations:
                            leaf_output["annotations"] = annotations

                    leaf_output["pipeline"] = pipeline
                    leaf_outputs.append(leaf_output)
                    parsed_rows = leaf_output.get("parsed_output") or []
                    yield encode_event("leaf_done", {
                        "table": table_name,
                        "pipeline": pipeline,
                        "rows": len(parsed_rows) if isinstance(parsed_rows, list) else 0,
                        "message": f"{table_name} leaf pipeline completed.",
                    })

            leaf_answer, rows_by_id, errors = _ui_rows_from_leaf_outputs(leaf_outputs)

            try:
                with _EXPLANATION_PIPELINE_LOCK:
                    generated_csv_files = _generate_ui_csv_files(
                        sql_query=sql_query,
                        plan=req.plan,
                        leaf_outputs=leaf_outputs,
                    )
                    yield encode_event("csv_done", {
                        "files": generated_csv_files,
                        "csv_page_url": "/ui/csv",
                        "message": "CSV files generated from leaf outputs.",
                    })

                    explanation = _run_ui_ap_explanation_for_csv_files(
                        sql_query=sql_query,
                        csv_files=generated_csv_files,
                        dataset=dataset,
                    )
                    explanations.append(explanation)
                    answer = _answer_from_ap_explanation(explanation)
                    answer = _annotate_final_answer(answer, leaf_outputs, rows_by_id)
                    yield encode_event("ap_explanation_done", {
                        "explanation": explanation,
                        "answer": answer,
                        "message": "AP explanation service completed.",
                    })
                    yield encode_event("answer_done", {
                        "answer": answer,
                        "message": "Answer received from AP explanation service.",
                    })
            except Exception as e:
                errors.append(f"ap-explanation: {e}")
                _log_event({
                    "type": "ui_ap_explanation_error",
                    "sql_query": sql_query,
                    "error": str(e),
                })
                yield encode_event("error", {
                    "message": f"AP explanation failed: {e}",
                    "errors": errors,
                })

            result = {
                "source": "gateway",
                "dataset": dataset,
                "sql": sql_query,
                "plan": req.plan,
                "answer": answer,
                "final_answer_annotations": [
                    annotation
                    for item in answer
                    for annotation in item.get("confidence_annotations") or []
                ],
                "rows_by_id": rows_by_id,
                "leaf_answer": leaf_answer,
                "leaf_outputs": leaf_outputs,
                "annotations": _collect_leaf_annotations(leaf_outputs),
                "pipeline_choices": pipeline_choices,
                "errors": errors,
                "explanations": explanations,
                "generated_csv_files": generated_csv_files,
                "csv_page_url": "/ui/csv",
                "csv_ready": bool(generated_csv_files),
            }

            _log_event({
                "type": "ui_run",
                "dataset": dataset,
                "sql_query": sql_query,
                "pipeline_choices": pipeline_choices,
                "answer_rows": len(answer),
                "explanations": len(explanations),
                "errors": errors,
            })
            yield encode_event("complete", {
                "result": result,
                "message": "Pipeline execution complete.",
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield encode_event("fatal_error", {
                "message": f"UI run failed: {e}",
            })

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# API for listing available models
@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {"object": "list", "data": EXPOSED_MODELS}

# API for calling a model to get a chat completion
@app.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    
    request_id = str(uuid.uuid4())[:8]
    global _LAST_EXPLAIN_DEBUG
    question = _extract_user_question(req.messages)
    if not question:
        raise HTTPException(status_code=400, detail="Empty user question")

    # Routing: UI model -> Ollama model
    ui_model = _canonical_model_id(req.model)
    ollama_model = MODEL_ROUTING.get(ui_model)
    if not ollama_model:
        raise HTTPException(status_code=400, detail=f"Unknown model id: {req.model}")

    # if not specified, default to 0.0 (deterministic)
    temperature = 0.0 if req.temperature is None else float(req.temperature)
    dataset = _dataset_config(req.dataset).name

    if ui_model == SQL_TABLE_PIPELINE_ID:
        try:
            plan = build_query_plan(question)
            if plan is None:
                raise RuntimeError("build_query_plan returned None")
            table_names = {
                str(task.table_name)
                for task in plan.leaf_tasks
                if str(task.table_name).strip()
            }
            with _EXPLANATION_PIPELINE_LOCK:
                csv_files = _copy_dataset_csv_files(dataset, table_names=table_names)
                pipeline_result = _run_ui_ap_explanation_for_csv_files(
                    sql_query=question,
                    csv_files=csv_files,
                    dataset=dataset,
                )
            created = _now()
            return {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion",
                "created": created,
                "model": ui_model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": pipeline_result["response_text"],
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SQL table pipeline failed: {exc}")

    if ui_model == PLANNER_ONLY_MODEL_ID:

        try:
            planner_result = _run_planner_first(
                sql_query=question,
                ollama_model=ollama_model,
                model_provider=PLANNER_LLM_PROVIDER,
                temperature=temperature,
                dataset=dataset,
                pipeline_id=PLANNER_ONLY_MODEL_ID,
            )

            out_text = json.dumps(planner_result, ensure_ascii=False, indent=2)
            response_text = _render_chat_response(out_text, planner_result)

            _log_event({
                "type": "planner_first_request",
                "dataset": dataset,
                "ui_model": ui_model,
                "requested_model": req.model,
                "request_id": request_id,
                "ollama_model": ollama_model,
                "model_provider": PLANNER_LLM_PROVIDER,
                "temperature": temperature,
                "stream": req.stream,
                "sql_query": question,
                "messages_count": len(req.messages),
                "raw_messages": [m.model_dump() for m in req.messages],
                "has_auth": bool(authorization),
                "response_format": GATEWAY_RESPONSE_FORMAT,
                "planner_result": planner_result,
            })

            _set_last_debug(
                question=question,
                raw_messages=[m.model_dump() for m in req.messages],
                ui_model=ui_model,
                ollama_model=ollama_model,
                model_provider=PLANNER_LLM_PROVIDER,
                temperature=temperature,
                context_tables=[],
                context_preview={},
                context_data={},
                prompt_full="PLANNER ONLY MODE",
                output_text=out_text,
                response_text=response_text,
                planner_result=planner_result,
            )

            _LAST_EXPLAIN_DEBUG = {}

            created = _now()
            return {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion",
                "created": created,
                "model": ui_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"planner only failed: {e}")
        
    if ui_model == PLANNER_ONLY_EXPLANATION_MODEL_ID:
        
        try:
            with _EXPLANATION_PIPELINE_LOCK:
                planner_result = _run_planner_first(
                    sql_query=question,
                    ollama_model=ollama_model,
                    model_provider=PLANNER_LLM_PROVIDER,
                    temperature=temperature,
                    dataset=dataset,
                    pipeline_id=PLANNER_ONLY_EXPLANATION_MODEL_ID,
                )

                explanation_client = ExplanationClient(
                    base_url=EXPLANATION_URL,
                    post_endpoint=EXPLANATION_ENDPOINT,
                    timeout=EXPLANATION_REQUEST_TIMEOUT,
                )

                pipeline_result = run_planner_first_explanation_pipeline(
                    sql_query=question,
                    planner_result=planner_result,
                    bucket_dir=EXPLANATION_BUCKET_DIR,
                    explanation_client=explanation_client,
                    delimiter=EXPLANATION_CSV_DELIMITER,
                    keep_rownum=EXPLANATION_KEEP_ROWNUM,
                    service_sql_query=_service_sql_for_dataset(question, dataset),
                )

                response_text = pipeline_result["response_text"]

                _log_event({
                    "type": "planner_first_explanation_request",
                    "dataset": dataset,
                    "ui_model": ui_model,
                    "requested_model": req.model,
                    "request_id": request_id,
                    "ollama_model": ollama_model,
                    "model_provider": PLANNER_LLM_PROVIDER,
                    "temperature": temperature,
                    "sql_query": question,
                    "generated_csv_files": pipeline_result["generated_csv_files"],
                    "explanation_output": pipeline_result["explanation_output"],
                    "planner_result": planner_result,
                })

                created = _now()

                return {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion",
                    "created": created,
                    "model": ui_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"planner only explanation pipeline failed: {e}",
            )

    if ui_model == ITERATIVE_PIPELINE_MODEL_ID:
        try:
            plan = build_query_plan(question)
            if plan is None:
                raise RuntimeError("build_query_plan returned None")
            planner_result = _run_ui_iterative_join_pipeline(
                sql_query=question,
                plan=plan.to_dict(),
                temperature=temperature,
                dataset=dataset,
            )

            with _EXPLANATION_PIPELINE_LOCK:
                explanation_client = ExplanationClient(
                    base_url=EXPLANATION_URL,
                    post_endpoint=EXPLANATION_ENDPOINT,
                    timeout=EXPLANATION_REQUEST_TIMEOUT,
                )
                pipeline_result = run_planner_first_explanation_pipeline(
                    sql_query=question,
                    planner_result=planner_result,
                    bucket_dir=EXPLANATION_BUCKET_DIR,
                    explanation_client=explanation_client,
                    delimiter=EXPLANATION_CSV_DELIMITER,
                    keep_rownum=EXPLANATION_KEEP_ROWNUM,
                    service_sql_query=_service_sql_for_dataset(question, dataset),
                )

            response_text = pipeline_result["response_text"]
            _log_event({
                "type": "iterative_join_explanation_request",
                "dataset": dataset,
                "ui_model": ui_model,
                "requested_model": req.model,
                "request_id": request_id,
                "ollama_model": ollama_model,
                "model_provider": PLANNER_LLM_PROVIDER,
                "temperature": temperature,
                "sql_query": question,
                "generated_csv_files": pipeline_result["generated_csv_files"],
                "explanation_output": pipeline_result["explanation_output"],
                "planner_result": planner_result,
            })

            created = _now()
            return {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion",
                "created": created,
                "model": ui_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"iterative join-aware pipeline failed: {e}",
            )

    if ui_model == LLM_INTERNAL_PIPELINE_ID:
        answer, out_text, confidence = _run_llm_internal_query(
            question,
            dataset,
            temperature,
        )
        confidence["scope"] = {"type": "query"}
        annotations = [confidence]
        response_text = _render_chat_response(out_text, answer)
        _log_event({
            "type": "internal_knowledge_request",
            "dataset": dataset,
            "ui_model": req.model,
            "request_id": request_id,
            "ollama_model": ollama_model,
            "temperature": temperature,
            "stream": req.stream,
            "question": question,
            "output": out_text,
            "response_chars": len(out_text),
            "annotations": annotations,
            "messages_count": len(req.messages),
            "raw_messages": [m.model_dump() for m in req.messages],
            "has_auth": bool(authorization),
            "response_format": GATEWAY_RESPONSE_FORMAT,
        })
        return {
            "id": f"chatcmpl-{_now()}",
            "object": "chat.completion",
            "created": _now(),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "annotations": annotations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


    #schema_info = _get_relevant_schema_info(question)
    annotations: List[Dict[str, Any]] = []
    ctx = _retrieve_context_data(
        question,
        dataset=dataset,
        annotation_sink=annotations,
        annotation_pipeline="rag",
    )

    base_prompt = PROMPT_TEMPLATE.format(
        question=question,
        #schema_info=json.dumps(schema_info, ensure_ascii=False, indent=2),
        context_data=json.dumps(_context_to_prompt_text(ctx), ensure_ascii=False),
    )

    _log_event({
        "type": "request",
        "dataset": dataset,
        "ui_model": req.model,
        "request_id": request_id,
        "ollama_model": ollama_model,
        "temperature": temperature,
        "stream": req.stream,
        "question": question,
        "prompt": base_prompt,
        "context_tables": list(ctx.keys()),
        "context_data": ctx,
        "annotations": annotations,
        "messages_count": len(req.messages),
        "raw_messages": [m.model_dump() for m in req.messages],
        "has_auth": bool(authorization),
        "prompt_chars": len(base_prompt),
    })

    # Call + retry JSON validity
    out_text = _call_model_with_retry(ollama_model, base_prompt, temperature, max_tries=2)
    _log_event({
        "type": "response",
        "ui_model": req.model,
        "ollama_model": ollama_model,
        "request_id": request_id,
        "output": out_text,
        "response_format": GATEWAY_RESPONSE_FORMAT,
        "response_chars": len(out_text),
    })
    response_text = _render_chat_response(out_text)

    _set_last_debug(
        question=question,
        raw_messages=[m.model_dump() for m in req.messages],
        ui_model=req.model,
        ollama_model=ollama_model,
        temperature=temperature,
        context_tables=list(ctx.keys()),
        context_preview={k: list(v.keys())[:5] for k, v in ctx.items()},
        context_data=ctx,
        #schema_info=schema_info,
        prompt_full=base_prompt,
        output_text=out_text,
        response_text=response_text,
    )

    _LAST_EXPLAIN_DEBUG = {}
    created = _now()
    return {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "annotations": annotations,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

@app.post("/v1/provenance/explain")
def explain_provenance(
    req: ProvenanceExplainRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    try:
        dataset = _dataset_config(req.dataset).name
        _load_csvs_once(dataset)

        if not req.question.strip():
            raise HTTPException(status_code=400, detail="Empty question")

        try:
            answer_items = _parse_answer_json(req.answer_json)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid answer_json: {e}")

        collected = _collect_provenance_rows(answer_items, dataset=dataset)
        rows_by_id = collected["rows_by_id"]
        missing_ids = collected["missing_ids"]

        validation_errors = _validate_provenance_structure(answer_items, rows_by_id)

        model_name = req.model or EXPLAIN_MODEL
        temperature = 0.0 if req.temperature is None else float(req.temperature)

        explanation_text: Optional[str] = None

        if not missing_ids:
            try:
                explanation_text = _explain_provenance_with_model(
                    question=req.question,
                    answer_items=answer_items,
                    rows_by_id=rows_by_id,
                    temperature=temperature,
                    explainer_model=model_name,
                    dataset=dataset,
                )
            except Exception as e:
                validation_errors.append(f"Explanation generation failed: {e}")
        else:
            validation_errors.append(
                "Some provenance ids could not be resolved to original rows; explanation skipped."
            )

        formulae = [
            {
                "result": item["result"],
                "formula": _provenance_to_formula(item["provenance"])
            }
            for item in answer_items
        ]

        _log_event({
            "type": "provenance_explain",
            "dataset": dataset,
            "has_auth": bool(authorization),
            "model": model_name,
            "temperature": temperature,
            "question": req.question,
            "answer_items": len(answer_items),
            "rows_resolved": len(rows_by_id),
            "missing_ids": missing_ids,
            "validation_errors": validation_errors,
        })
        _set_last_explain_debug(
            question=req.question,
            answer=answer_items,
            formulae=formulae,
            rows_by_id=rows_by_id,
            missing_ids=missing_ids,
            validation_errors=validation_errors,
            explanation_text=explanation_text,
            model=model_name,
            temperature=temperature,
            dataset=dataset,
        )

        return {
            "dataset": dataset,
            "question": req.question,
            "answer": answer_items,
            "formulae": formulae,
            "rows_by_id": rows_by_id,
            "missing_ids": missing_ids,
            "validation_errors": validation_errors,
            "explanation_text": explanation_text,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/debug/ui", response_class=HTMLResponse)
def debug_ui() -> str:
    chat_data = _LAST_DEBUG or {}
    explain_data = _LAST_EXPLAIN_DEBUG or {}

    # chat section
    question_raw = chat_data.get("question", "")
    output_text_raw = chat_data.get("output_text", "")

    question = html.escape(question_raw)
    raw_messages = html.escape(json.dumps(chat_data.get("raw_messages", []), indent=2, ensure_ascii=False))
    context_tables = html.escape(json.dumps(chat_data.get("context_tables", []), indent=2, ensure_ascii=False))
    context_preview = html.escape(json.dumps(chat_data.get("context_preview", {}), indent=2, ensure_ascii=False))
    context_data = html.escape(json.dumps(chat_data.get("context_data", {}), indent=2, ensure_ascii=False))
    prompt_full = html.escape(chat_data.get("prompt_full", ""))
    output_text = html.escape(output_text_raw)

    model_routing = html.escape(json.dumps({
        "ui_model": chat_data.get("ui_model", ""),
        "ollama_model": chat_data.get("ollama_model", "")
    }, indent=2, ensure_ascii=False))

    # explain section
    explain_question = html.escape(explain_data.get("question", ""))
    explain_answer = html.escape(json.dumps(explain_data.get("answer", []), indent=2, ensure_ascii=False))
    explain_rows_by_id = html.escape(json.dumps(explain_data.get("rows_by_id", {}), indent=2, ensure_ascii=False))
    explain_missing_ids = html.escape(json.dumps(explain_data.get("missing_ids", []), indent=2, ensure_ascii=False))
    explain_validation_errors = html.escape(json.dumps(explain_data.get("validation_errors", []), indent=2, ensure_ascii=False))
    explain_text_raw = explain_data.get("explanation_text", "") or ""
    explain_text = html.escape(explain_text_raw)

    explain_metadata = html.escape(json.dumps({
        "model": explain_data.get("model", ""),
        "temperature": explain_data.get("temperature", ""),
    }, indent=2, ensure_ascii=False))

    # values for JS: use json.dumps so strings are valid JavaScript string literals
    explain_question_raw = explain_data.get("question", "")

    js_question = json.dumps(question_raw, ensure_ascii=False)
    js_answer = json.dumps(output_text_raw, ensure_ascii=False)
    js_explain_question = json.dumps(explain_question_raw, ensure_ascii=False)
    js_has_explain = "true" if bool(explain_text_raw.strip()) else "false"
    return f"""
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>Gateway Debug</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background: #f7f7f9;
                color: #222;
            }}
            .card {{
                background: white;
                border-radius: 10px;
                padding: 16px;
                margin-bottom: 16px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            }}
            pre {{
                white-space: pre-wrap;
                word-break: break-word;
                background: #f1f3f5;
                padding: 12px;
                border-radius: 8px;
                overflow-x: auto;
            }}
            button {{
                padding: 8px 14px;
                border: none;
                border-radius: 6px;
                background: #3b82f6;
                color: white;
                cursor: pointer;
                margin-bottom: 16px;
            }}
            button:hover {{
                background: #2563eb;
            }}
            h2 {{
                margin-top: 32px;
            }}
            .muted {{
                color: #666;
                font-style: italic;
            }}
        </style>
        <script>
        async function runExplain() {{
            try {{
                const question = {js_question};
                const answer = {js_answer};
                const explainQuestion = {js_explain_question};
                const hasExplain = {js_has_explain};

                const needsExplain =
                    !!question &&
                    !!answer &&
                    (!hasExplain || explainQuestion !== question);

                if (!needsExplain) {{
                    return;
                }}

                console.log("Calling /v1/provenance/explain ...");

                const resp = await fetch("/v1/provenance/explain", {{
                    method: "POST",
                    headers: {{
                        "Content-Type": "application/json"
                    }},
                    body: JSON.stringify({{
                        question: question,
                        answer_json: answer
                    }})
                }});

                if (!resp.ok) {{
                    const txt = await resp.text();
                    console.error("Explain call failed:", resp.status, txt);
                    return;
                }}

                location.reload();
            }} catch (e) {{
                console.error("Explain call failed", e);
            }}
        }}

        window.addEventListener("load", () => {{
            runExplain();
        }});
        </script>
    </head>
    <body>
        <h1>Gateway Debug</h1>

        <button onclick="location.reload()">Refresh</button>

        <h2>Chat Completion Debug</h2>

        <div class="card">
            <h3>Question</h3>
            <pre>{question}</pre>
        </div>

        <div class="card">
            <h3>Model Routing</h3>
            <pre>{model_routing}</pre>
        </div>

        <div class="card">
            <h3>Raw Messages</h3>
            <pre>{raw_messages}</pre>
        </div>

        <div class="card">
            <h3>Context Tables</h3>
            <pre>{context_tables}</pre>
        </div>

        <div class="card">
            <h3>Context Preview</h3>
            <pre>{context_preview}</pre>
        </div>

        <div class="card">
            <h3>Retrieved Context</h3>
            <pre>{context_data}</pre>
        </div>

        <div class="card">
            <h3>Extended Prompt</h3>
            <pre>{prompt_full}</pre>
        </div>

        <div class="card">
            <h3>Model Output</h3>
            <pre>{output_text}</pre>
        </div>

        <h2>Provenance Explain Debug</h2>
        <div class="card">
            <h3>Model Explanation</h3>
            <pre>{explain_metadata}</pre>
        </div>
        <div class="card">
            <h3>Explain Question</h3>
            <pre>{explain_question}</pre>
        </div>

        <div class="card">
            <h3>Explain Answer</h3>
            <pre>{explain_answer}</pre>
        </div>


        <div class="card">
            <h3>Resolved Rows</h3>
            <pre>{explain_rows_by_id}</pre>
        </div>

        <div class="card">
            <h3>Missing IDs</h3>
            <pre>{explain_missing_ids}</pre>
        </div>

        <div class="card">
            <h3>Validation Errors</h3>
            <pre>{explain_validation_errors}</pre>
        </div>

        <div class="card">
            <h3>Explanation Text</h3>
            <pre id="explain-empty">{explain_text}</pre>
            {"<div class='muted'>Generating explanation...</div>" if output_text_raw.strip() and not explain_text_raw.strip() else ""}
        </div>
    </body>
    </html>
    """
