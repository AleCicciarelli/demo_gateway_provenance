#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import asdict
import html
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import csv
from pathlib import Path
from collections import defaultdict
import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from fastapi.responses import HTMLResponse
from prompt import build_leaf_prompt
from prompt_internal_knowledge import PROMPT_INTERNAL_KNOWLEDGE_TEMPLATE
from tpch_schema_info import SCHEMA_INFO
from planner import  build_query_plan
import uuid
import torch
app = FastAPI()

# =========================
# Config
# =========================
LOG_PATH = os.getenv("GATEWAY_LOG_PATH", "/app/logs/provsql_gateway_logs.jsonl")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300"))
CSV_DIR = os.getenv("CSV_DIR", "/app/tpch_no_provsql")
MAX_CONTEXT_ROWS = int(os.getenv("MAX_CONTEXT_ROWS", "37"))
MAX_TABLES = int(os.getenv("MAX_TABLES", "5"))

OLLAMA_MODEL_BASE = os.getenv("OLLAMA_MODEL_BASE", "llama3:8b")
OLLAMA_MODEL_FT_NL = os.getenv("OLLAMA_MODEL_FT_NL", "llama3-8b-dpo2-sft1-nl:latest")
OLLAMA_MODEL_FT_SQL = os.getenv("OLLAMA_MODEL_FT_SQL", "llama3-8b-dpo1-sft2-sql:latest")

FAISS_INDEX_FOLDER = os.getenv("FAISS_INDEX_FOLDER", "/app/faiss_index_tpch")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMB_DEVICE = os.getenv("EMB_DEVICE", "auto")
INDEX_TABLES = os.getenv("INDEX_TABLES", "")
INDEX_SET = set(t.strip() for t in INDEX_TABLES.split(",") if t.strip())
BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "500"))

EXPLAIN_MODEL = os.getenv("OLLAMA_MODEL_EXPLAIN", "deepseek-r1:70b")
EXPLAIN_MAX_TRIES = int(os.getenv("EXPLAIN_MAX_TRIES", "2"))

RETRIEVER_K = int(os.getenv("RETRIEVER_K", "12"))
MAX_ITERATIVE_RETRIEVALS = int(os.getenv("MAX_ITERATIVE_RETRIEVALS", "4"))
# Mapping "UI model id" -> "Ollama model name".
MODEL_ROUTING: Dict[str, str] = {
    "base-llama3-8b": os.getenv("OLLAMA_MODEL_BASE", "llama3:8b"),
    "best-ft-llama3-8b-nl": os.getenv("OLLAMA_MODEL_FT_NL", "llama3-8b-dpo2-sft1-nl:latest"),
    "best-ft-llama3-8b-sql": os.getenv("OLLAMA_MODEL_FT_SQL", "llama3-8b-dpo1-sft2-sql:latest"),
    "planner-first": os.getenv("OLLAMA_MODEL_PLANNER_FIRST", "llama3:70b"),
    "internal-knowledge": os.getenv("OLLAMA_MODEL_INTERNAL_KNOWLEDGE", "llama3:8b"),
}

# Exposing only the UI model ids in the /v1/models endpoint.
EXPOSED_MODELS = [{"id": mid, "object": "model"} for mid in MODEL_ROUTING.keys()]


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
    model_config = ConfigDict(extra="allow")  # allow extra fields, for compatibility with OpenAI's API which can use more fields

class ProvenanceExplainRequest(BaseModel):
    question: str
    answer_json: str
    model: Optional[str] = None
    temperature: Optional[float] = 0.0
# =========================
# Debug state
# =========================
_LAST_DEBUG: Dict[str, Any] = {}
_LAST_EXPLAIN_DEBUG: Dict[str, Any] = {}
# =========================
# Helpers
# =========================

class STEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        if EMB_DEVICE == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = EMB_DEVICE

        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        print(f"[EMB] SentenceTransformer using device: {device}", flush=True)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()
    
_EMBEDDINGS: Optional[STEmbeddings] = None

def _get_embeddings() -> STEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = STEmbeddings(EMB_MODEL)
    return _EMBEDDINGS  

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
    max_tries: int = 2,
    validator: Optional[Callable[[str], Tuple[bool, Optional[str]]]] = None,
    scorer: Optional[Callable[[str], int]] = None,
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
        out = _ollama_generate(ollama_model, prompt, temperature)
        last = out
        ok, err = validate(out)
        score = scorer(out) if scorer is not None else (1 if ok else 0)
        if score > best_score:
            best = out
            best_score = score
        _log_event({
            "type": "attempt",
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
            + RETRY_SUFFIX
        )

    # If every attempt fails, keep the attempt with the most individually valid
    # rows instead of blindly returning a shorter correction-only retry.
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

# Cache in memoria: table -> list[dict]
_CSV_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_CSV_LOADED = False
_CSV_RID_INDEX: Dict[str, Dict[str, int]] = {}
_GLOBAL_RID_INDEX: Dict[str, Tuple[str, int]] = {}
def _load_csvs_once() -> None:
    global _CSV_LOADED
    if _CSV_LOADED:
        return

    base = Path(CSV_DIR)
    if not base.exists():
        _log_event({"type": "retrieval_error", "error": f"CSV_DIR not found: {CSV_DIR}"})
        _CSV_LOADED = True
        return

    for p in sorted(base.glob("*.csv")):
        table = p.stem
        id_col = f"{table}_rownum"
        rows: List[Dict[str, Any]] = []
        rid_to_idx: Dict[str, int] = {}
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f, delimiter="|")
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
        _CSV_CACHE[table] = rows
        _CSV_RID_INDEX[table] = rid_to_idx

    _CSV_LOADED = True
    _log_event({"type": "retrieval_loaded", "tables": list(_CSV_CACHE.keys()), "csv_dir": CSV_DIR})


def _build_global_rid_index() -> None:
    _load_csvs_once()
    global _GLOBAL_RID_INDEX
    if _GLOBAL_RID_INDEX:
        return

    for table, rid_map in _CSV_RID_INDEX.items():
        for rid, idx in rid_map.items():
            _GLOBAL_RID_INDEX[rid] = (table, idx)

def _resolve_row_by_rid(rid: str) -> Optional[Dict[str, Any]]:
    _build_global_rid_index()
    hit = _GLOBAL_RID_INDEX.get(rid)
    if not hit:
        return None

    table, idx = hit
    row = _CSV_CACHE[table][idx]

    return {
        "table": table,
        "rid": rid,
        "row": row,
    }


# =========================
# Retrieval and indexing
# ========================
_VECTOR_STORE: Optional[FAISS] = None

def _row_to_text(row: Dict[str, Any], table: Optional[str] = None) -> str:
    parts = []

    if table:
        parts.append(f"table: {table}")

    for k, v in row.items():
        if k == "__rid__" or k.endswith("_rownum"):
            continue
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        parts.append(f"{k}: {s}")

    return " | ".join(parts)
def _context_to_prompt_text(ctx: Dict[str, Any]) -> str:
    lines = []
    for table, rows in ctx.items():
        for rid, row in rows.items():
            parts = [f"{k}={v}" for k, v in row.items()]
            lines.append(f"{rid} | table={table} | " + " | ".join(parts))
    return "\n".join(lines)


def _get_or_build_faiss() -> FAISS:
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    embedding_model = _get_embeddings()
    index_path = Path(FAISS_INDEX_FOLDER)

    # Try loading an existing FAISS index only if directory exists and is not empty
    if index_path.exists() and index_path.is_dir() and any(index_path.iterdir()):
        try:
            _VECTOR_STORE = FAISS.load_local(
                FAISS_INDEX_FOLDER,
                embedding_model,
                allow_dangerous_deserialization=True,
            )
            _log_event({
                "type": "faiss_loaded",
                "path": FAISS_INDEX_FOLDER,
                "emb_model": EMB_MODEL
            })
            return _VECTOR_STORE
        except Exception as e:
            _log_event({
                "type": "faiss_load_failed",
                "path": FAISS_INDEX_FOLDER,
                "emb_model": EMB_MODEL,
                "error": str(e)
            })

    _load_csvs_once()

    vector_store: Optional[FAISS] = None
    batch_docs: List[Document] = []
    total_docs = 0

    for table, rows in _CSV_CACHE.items():
        if INDEX_SET and table not in INDEX_SET:
            continue

        for r in rows:
            rid = r.get("__rid__")
            if not rid:
                continue

            text = _row_to_text(r, table)
            if not text:
                continue

            batch_docs.append(
                Document(
                    page_content=text,
                    metadata={"table": table, "rid": rid}
                )
            )

            if len(batch_docs) >= BATCH_SIZE:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch_docs, embedding=embedding_model)
                else:
                    vector_store.add_documents(batch_docs)

                total_docs += len(batch_docs)
                batch_docs = []
                print(f"Indexed {total_docs} documents", flush=True)

    if batch_docs:
        if vector_store is None:
            vector_store = FAISS.from_documents(batch_docs, embedding=embedding_model)
        else:
            vector_store.add_documents(batch_docs)
        total_docs += len(batch_docs)

    if vector_store is None:
        raise RuntimeError("No documents indexed. Check CSV_DIR / INDEX_TABLES / file contents.")

    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(FAISS_INDEX_FOLDER)

    _log_event({
        "type": "faiss_built",
        "path": FAISS_INDEX_FOLDER,
        "n_docs": total_docs,
        "emb_model": EMB_MODEL
    })

    _VECTOR_STORE = vector_store
    return _VECTOR_STORE
def retrieve_context_data_iterative(question: str) -> Dict[str, Any]:
    """
    Iterative retrieval: start with k=RETRIEVER_K, then increase k by RETRIEVER_K
    in each iteration, until no new rows are retrieved or MAX_ITERATIVE_RETRIEVALS is reached.
    """
    _load_csvs_once()
    print("\n[ITERATIVE RETRIEVAL] START", flush=True)
    print(f"[ITERATIVE RETRIEVAL] question={question}", flush=True)

    vs = _get_or_build_faiss()

    ctx: Dict[str, Any] = defaultdict(dict)
    seen_docs = set()
    k = RETRIEVER_K

    for iteration in range(1, MAX_ITERATIVE_RETRIEVALS + 1):
        docs = vs.similarity_search(question, k=k)
        new_count = 0
        new_rids = []

        print(f"\n[ITERATION {iteration}] k={k} | docs_returned={len(docs)}", flush=True)

        for rank, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            table = meta.get("table")
            rid = meta.get("rid")

            if not table or not rid:
                continue

            if rid in seen_docs:
                continue

            idx0 = _CSV_RID_INDEX.get(table, {}).get(rid)
            if idx0 is None:
                continue

            if table not in ctx and len(ctx) >= MAX_TABLES:
                continue

            row = _CSV_CACHE[table][idx0]
            ctx[table][rid] = row
            seen_docs.add(rid)
            new_count += 1
            new_rids.append((rank, table, rid))

        print(f"[ITERATION {iteration}] new_rows_added={new_count}", flush=True)

        for rank, table, rid in new_rids:
            print(f"  + rank={rank:>3} | table={table:<12} | rid={rid}", flush=True)
            print(
                json.dumps(_CSV_CACHE[table][_CSV_RID_INDEX[table][rid]], ensure_ascii=False),
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
                "row": _CSV_CACHE[table][_CSV_RID_INDEX[table][rid]],
            }
            for rank, table, rid in new_rids
        }

        _log_event({
            "type": "iterative_retrieval",
            "iteration": iteration,
            "retriever_k": k,
            "new_rows_added": new_count,
            "new_rids": [
                {"rank": rank, "table": table, "rid": rid}
                for rank, table, rid in new_rids
            ],
            "tables": list(ctx.keys()),
            "new_rows": new_rows,
            "context_data": ctx_snapshot,
            "context_rows_total": sum(len(rows) for rows in ctx_snapshot.values()),
            "question": question,
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
        "tables": list(ctx.keys()),
        "rows_preview": preview,
        "final_rids": final_rids,
        "context_data": final_context_data,
        "context_rows_total": sum(len(rows) for rows in final_context_data.values()),
        "question": question,
    })

    return dict(ctx)
def _retrieve_context_data(question: str) -> Dict[str, Any]:
    _load_csvs_once()
    vs = _get_or_build_faiss()

    docs = vs.similarity_search(question, k=RETRIEVER_K)

    ctx: Dict[str, Any] = defaultdict(dict)
    used = 0

    for d in docs:
        meta = d.metadata or {}
        table = meta.get("table")
        rid = meta.get("rid")
        if not table or not rid:
            continue

        idx0 = _CSV_RID_INDEX.get(table, {}).get(rid)
        if idx0 is None:
            continue
        row = _CSV_CACHE[table][idx0]
        if table not in ctx and len(ctx) >= MAX_TABLES:
            continue

        if rid in ctx[table]:
            continue

        ctx[table][rid] = row
        used += 1
        if used >= MAX_CONTEXT_ROWS:
            break

    ctx = dict(ctx)

    preview = {table: list(rows.keys())[:3] for table, rows in ctx.items()}

    _log_event({
        "type": "retrieval",
        "tables": list(ctx.keys()),
        "rows_preview": preview,
        "context_data": ctx,
        "n_rows_used": used,
        "question": question,
    })

    return ctx
# =========================
# Planner-first helpers
# =========================

def _build_leaf_retrieval_query(task: Dict[str, Any]) -> str:
    """
    Build a retrieval query for one leaf task.
    The model remains leaf-only: predicates and joins are not pushed into extraction.
    """
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

    return "Retrieve relevant rows for this task"

def _leaf_rows_by_id(ctx: Dict[str, Any], table_name: str) -> Dict[str, Dict[str, Any]]:
    return dict(ctx.get(table_name) or {})

def _run_leaf_task(
    task: Dict[str, Any],
    ollama_model: str,
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
    temperature: float,
) -> Dict[str, Any]:
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

        print("[PLANNER-FIRST] calling retrieve_context_data_iterative()", flush=True)
        leaf_ctx = retrieve_context_data_iterative(retrieval_query)
        print("[PLANNER-FIRST] retrieve_context_data_iterative() returned", flush=True)

        leaf_outputs.append(
            _run_leaf_task(
                task=task_dict,
                ollama_model=ollama_model,
                temperature=temperature,
                ctx=leaf_ctx,
                retrieval_query=retrieval_query,
            )
        )

    return {
        "sql": sql_query,
        "plan": plan_dict,
        "leaf_outputs": leaf_outputs,
    }


# =========================
# Provenance helpers
# =========================
def _parse_answer_json(text: str) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(text)
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

def _collect_provenance_rows(answer_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows_by_id: Dict[str, Any] = {}
    missing_ids: List[str] = []

    for item in answer_items:
        for witness_set in item["provenance"]:
            for rid in witness_set:
                if rid in rows_by_id:
                    continue
                resolved = _resolve_row_by_rid(rid)
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

def _build_schema_subset_from_rows(rows_by_id: Dict[str, Any]) -> Dict[str, Any]:
    tables = sorted({info["table"] for info in rows_by_id.values()})
    return {t: SCHEMA_INFO[t] for t in tables if t in SCHEMA_INFO}

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
) -> str:
    schema_info = _build_schema_subset_from_rows(rows_by_id)
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
    ollama_model = MODEL_ROUTING.get(req.model)
    if not ollama_model:
        raise HTTPException(status_code=400, detail=f"Unknown model id: {req.model}")

    # if not specified, default to 0.0 (deterministic)
    temperature = 0.0 if req.temperature is None else float(req.temperature)
    

    if req.model == "planner-first":

        try:
            planner_result = _run_planner_first(
                sql_query=question,
                ollama_model=ollama_model,
                temperature=temperature,
            )

            out_text = json.dumps(planner_result, ensure_ascii=False, indent=2)

            _log_event({
                "type": "planner_first_request",
                "ui_model": req.model,
                "request_id": request_id,
                "ollama_model": ollama_model,
                "temperature": temperature,
                "stream": req.stream,
                "sql_query": question,
                "messages_count": len(req.messages),
                "raw_messages": [m.model_dump() for m in req.messages],
                "has_auth": bool(authorization),
                "planner_result": planner_result,
            })

            _set_last_debug(
                question=question,
                raw_messages=[m.model_dump() for m in req.messages],
                ui_model=req.model,
                ollama_model=ollama_model,
                temperature=temperature,
                context_tables=[],
                context_preview={},
                context_data={},
                prompt_full="PLANNER-FIRST MODE",
                output_text=out_text,
                planner_result=planner_result,
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
                        "message": {"role": "assistant", "content": out_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Planner-first failed: {e}")
    if req.model == "internal-knowledge":
        base_prompt = PROMPT_INTERNAL_KNOWLEDGE_TEMPLATE.format(question=question)
        out_text = _call_model_with_retry(ollama_model, base_prompt, temperature, max_tries=2)
        _log_event({
            "type": "internal_knowledge_request",
            "ui_model": req.model,
            "request_id": request_id,
            "ollama_model": ollama_model,
            "temperature": temperature,
            "stream": req.stream,
            "question": question,
            "prompt": base_prompt,
            "output": out_text,
            "prompt_chars": len(base_prompt),
            "response_chars": len(out_text),
            "messages_count": len(req.messages),
            "raw_messages": [m.model_dump() for m in req.messages],
            "has_auth": bool(authorization),
        })
        return {
            "id": f"chatcmpl-{_now()}",
            "object": "chat.completion",
            "created": _now(),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": out_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


    #schema_info = _get_relevant_schema_info(question)
    ctx = _retrieve_context_data(question)

    base_prompt = PROMPT_TEMPLATE.format(
        question=question,
        #schema_info=json.dumps(schema_info, ensure_ascii=False, indent=2),
        context_data=json.dumps(_context_to_prompt_text(ctx), ensure_ascii=False),
    )

    _log_event({
        "type": "request",
        "ui_model": req.model,
        "request_id": request_id,
        "ollama_model": ollama_model,
        "temperature": temperature,
        "stream": req.stream,
        "question": question,
        "prompt": base_prompt,
        "context_tables": list(ctx.keys()),
        "context_data": ctx,
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
        "response_chars": len(out_text),
    })

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
                "message": {"role": "assistant", "content": out_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

@app.post("/v1/provenance/explain")
def explain_provenance(
    req: ProvenanceExplainRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    try:
        _load_csvs_once()

        if not req.question.strip():
            raise HTTPException(status_code=400, detail="Empty question")

        try:
            answer_items = _parse_answer_json(req.answer_json)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid answer_json: {e}")

        collected = _collect_provenance_rows(answer_items)
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
        )

        return {
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
