# Row FAISS Index with BGE-M3

This document explains how the row-level FAISS index over the TPC-H CSV files in
`tpch_no_provsql/` is built. Each indexed document represents one source row plus useful
metadata such as table name, primary key, rownum/rid, column values, and selected
foreign-key links.

## Main Build Command

The usual entry point is:

```bash
bash build_row_ind.sh
```

That wrapper:

1. activates `.venv`;
2. creates temporary Hugging Face / Torch cache directories under
   `${TMPDIR:-/tmp}/faiss_${OAR_JOB_ID:-manual}`;
3. builds `documents.jsonl` from the CSV files and schema profile;
4. builds the FAISS vectors from `documents.jsonl` with BGE-M3 on CUDA;
5. uses `--no-resume` for the FAISS step so stale checkpoint metadata is not
   reused.
Outputs:

- `faiss_index_tpch_rows_bge_m3/documents.jsonl`: row documents used as input to
  embedding.
- `faiss_index_tpch_rows_bge_m3/index.faiss`: FAISS vector index.
- `faiss_index_tpch_rows_bge_m3/index.pkl`: LangChain document store metadata.
- `faiss_index_tpch_rows_bge_m3.checkpoint/`: resumable checkpoint containing
  `index.faiss`, `index.pkl`, and `manifest.json`.

The current `documents.jsonl` contains `866602` row documents.

## How Documents Are Made

`build_row_index.py` does the document preparation:

- reads every `*.csv` from `tpch_no_provsql/` using `|` as separator;
- reads schema information from `tpch_no_provsql/schema_profile_tpch.json`; 
  - this file is the output of the script `schema_extractor.py` that loads CSV files, examinates columns and values to extract the primary keys (unique/non-null values) and the foreign keys (values and name similarity).
- detects primary-key candidates and rownum columns (provenance column);
- writes one `Document` per CSV row;
- optionally stops after writing documents when `--documents-only` is passed.

`build_row_faiss_index.py` does the embedding/indexing step:

- reads the completed `documents.jsonl`;
- embeds documents in batches with `BAAI/bge-m3`;
- saves periodic checkpoints;
- saves the final LangChain FAISS index.

Embeddings are created with:

```python
HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
```

## Metadata Stored per Row

Each JSONL record has two fields:

- `page_content`: the text embedded by BGE-M3 and stored in FAISS.
- `metadata`: structured information saved in `index.pkl` with the vector.

The metadata is added in `build_row_index.py` when each LangChain `Document` is
created. It lets the gateway recover the exact CSV row after FAISS returns a
nearest document.

`linked_rows` is created only from schema `foreign_key_candidates` whose
`name_similarity` is greater than `0.0`. Value overlap alone is not enough,
because unrelated numeric columns can have the same value domain.

The current schema profile has `10` foreign-key candidates and `0` candidates
with `name_similarity <= 0.0`. For example, the previous false-positive links
`part.p_size -> customer.c_custkey` and `part.p_size -> supplier.s_suppkey` are
not kept.

Example metadata for the `part_5` row after rebuilding with the filtered
schema:

```json
{
  "doc_type": "row",
  "table": "part",
  "row_id": "part:part_rownum=part_5",
  "rownum_column": "part_rownum",
  "rownum_value": "part_5",
  "primary_key": {
    "p_partkey": 5
  },
  "values": {
    "p_partkey": 5,
    "p_name": "forest brown coral puff cream",
    "p_mfgr": "Manufacturer#3           ",
    "p_brand": "Brand#32  ",
    "p_type": "STANDARD POLISHED TIN",
    "p_size": 15,
    "p_container": "SM PKG    ",
    "p_retailprice": 905.0,
    "p_comment": " wake carefully "
  },
  "linked_rows": []
}
```

For the same row, BGE-M3 embeds this `page_content` text:

```text
Row from table part.
This row represents one record from part.

Primary key:
part.p_partkey = 5.

Column values:
part.p_partkey = 5.
part.p_name = forest brown coral puff cream.
part.p_mfgr = Manufacturer#3.
part.p_brand = Brand#32.
part.p_type = STANDARD POLISHED TIN.
part.p_size = 15.
part.p_container = SM PKG.
part.p_retailprice = 905.0.
part.p_comment = wake carefully.
```

So the non-provenance row values are indexed twice in different forms:

- as text in `page_content`, which BGE-M3 embeds for FAISS similarity search;
- as structured JSON in `metadata.values`, which preserves the original fields
  after a row is retrieved.

Since `part.p_size` has no positive name similarity with a target key, it is not
kept as a linked row.

Example with a valid linked row, from `supplier_15`:

```json
{
  "doc_type": "row",
  "table": "supplier",
  "row_id": "supplier:supplier_rownum=supplier_15",
  "rownum_column": "supplier_rownum",
  "rownum_value": "supplier_15",
  "primary_key": {
    "s_suppkey": 15
  },
  "values": {
    "s_suppkey": 15,
    "s_name": "Supplier#000000015       ",
    "s_address": "olXVbNBfVzRqgokr1T,Ie",
    "s_nationkey": 8,
    "s_phone": "18-453-357-6394",
    "s_acctbal": 308.56,
    "s_comment": " across the furiously regular platelets wake even deposits. quickly express she"
  },
  "linked_rows": [
    {
      "relation": "foreign_key",
      "from_table": "supplier",
      "from_columns": ["s_nationkey"],
      "from_values": ["8"],
      "to_table": "nation",
      "to_columns": ["n_nationkey"],
      "linked_values": {
        "nation.n_nationkey": 8,
        "nation.n_name": "INDIA                    ",
        "nation.n_regionkey": 2
      }
    }
  ]
}
```

This link is kept because the schema profile contains a positive-similarity
foreign-key candidate:

```json
{
  "from_table": "supplier",
  "from_columns": ["s_nationkey"],
  "to_table": "nation",
  "to_columns": ["n_nationkey"],
  "coverage": 1.0,
  "name_similarity": 0.4,
  "score": 0.85
}
```

BGE-M3 embeds the supplier row together with a short text representation of the
linked nation row:

```text
Row from table supplier.
This row represents one record from supplier.

Primary key:
supplier.s_suppkey = 15.

Column values:
supplier.s_suppkey = 15.
supplier.s_name = Supplier#000000015.
supplier.s_address = olXVbNBfVzRqgokr1T,Ie.
supplier.s_nationkey = 8.
supplier.s_phone = 18-453-357-6394.
supplier.s_acctbal = 308.56.
supplier.s_comment = across the furiously regular platelets wake even deposits. quickly express she.

Foreign key relations:
supplier.s_nationkey = 8 references nation.n_nationkey = 8.
Linked nation row:
nation.n_nationkey = 8.
nation.n_name = INDIA.
nation.n_regionkey = 2.
```

## Example from `evaluation/questions.json`

`query11`:

```json
{
  "query_id": "query11",
  "question_nl": "What is the retail price of the part named 'forest brown coral puff cream'?",
  "question_sql": "SELECT p.p_retailprice FROM part p WHERE p.p_name = 'forest brown coral puff cream';"
}
```


Expected top document:

```text
table: part
row_id: part:part_rownum=part_5
primary_key: {'p_partkey': 5}
part.p_name = forest brown coral puff cream.
part.p_retailprice = 905.0.
```

SQL reference answer:

```sql
SELECT p.p_retailprice
FROM part p
WHERE p.p_name = 'forest brown coral puff cream';
```

Expected result:

```text
905.0
```

## Querying over FAISS index
TO INSERT
