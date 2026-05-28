# Planner-First Logic and Evaluation

This document explains the planner-first path in the demo gateway and the evaluation workflow used to measure it.

The planner-first mode is exposed as the OpenAI-compatible model id `planner-first`. Unlike the regular gateway path, it does not ask the model to answer the full SQL query directly. It first decomposes the SQL into table-level leaf extraction tasks, retrieves candidate rows for each leaf, asks the model to copy rows from the retrieved context, and returns the full intermediate trace.

At the current stage, planner-first is an extraction pipeline. The gateway builds and returns the plan and leaf outputs, but it does not yet execute the deterministic post-processing operators such as joins, filters, grouping, aggregation, projection, ordering, or limit.

## Main Files

- `planner.py`: parses SQL and builds a structured query plan.
- `gateway.py`: exposes the `planner-first` model route, performs retrieval, builds prompts, calls Ollama, validates leaf output, and logs debug data.
- `prompt.py`: builds the leaf extraction prompt.
- `run_planner_first_eval.py`: runs planner-first over the evaluation dataset and writes JSONL predictions.
- `evaluate_planner_first_outputs.py`: compares predictions against ProvSQL-generated ground truth and writes metrics.
- `run_oar_planner_eval.sh`: convenience wrapper for running the evaluation on an OAR cluster with a local Ollama server.
- `evaluation/generate_ground_truth_provsql.sh`: generates query and leaf ground truth from a ProvSQL-enabled PostgreSQL container.

Related notes:

- `docs/planner_first_results.md`: current planner-first result discussion.
- `docs/internal_knowledge_logic_and_results.md`: internal-knowledge baseline logic and result discussion.

## Runtime Flow

### 1. User Calls `planner-first`

Send a request to the normal chat completions endpoint:

```bash
curl -X POST http://localhost:9005/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "planner-first",
    "messages": [
      {
        "role": "user",
        "content": "SELECT n.n_name FROM nation n"
      }
    ]
  }'
```

The visible model id `planner-first` is routed through `MODEL_ROUTING` in `gateway.py` to the actual Ollama model named by `OLLAMA_MODEL_PLANNER_FIRST`. If that environment variable is not set, the default is `llama3:8b`.

### 2. SQL Is Parsed Into a Plan

`gateway._run_planner_first()` calls:

```python
plan = build_query_plan(sql_query)
```

`planner.build_query_plan()` uses `sqlglot` to parse the SQL AST and extract:

- base tables and aliases
- join clauses and join keys
- selected columns
- local `WHERE` predicates
- `GROUP BY` columns
- aggregate functions
- `ORDER BY` items
- `LIMIT`

It returns a `QueryPlan` containing:

- `query_type`: currently `SELECT`
- `sql`: the original SQL
- `leaf_tasks`: one table-level extraction task per required table
- `joins`: extracted join specifications
- `post_ops`: operators that should be applied after leaf extraction

### 3. Leaf Extraction Tasks Are Built

Each `LeafExtractionTask` describes what must be extracted from one table:

```json
{
  "table_name": "nation",
  "alias": "n",
  "scan_op": "FilterLeafScan",
  "columns": ["n_name", "n_regionkey"],
  "local_predicates": ["n_regionkey = 1"],
  "join_keys": [],
  "select_columns": ["n_name"],
  "group_by_columns": [],
  "aggregate_columns": []
}
```

The current leaf prompt intentionally does not apply the predicates. Even when the planner marks a leaf as `FilterLeafScan`, the model is asked only to copy rows from the target table that appear in retrieved context. Filtering and relational semantics are reserved for deterministic post-processing.

### 4. A Retrieval Query Is Created for Each Leaf

For every leaf task, `gateway._build_leaf_retrieval_query()` creates a compact semantic query:

```text
table: nation | columns: n_name, n_regionkey
```

The query includes the target table and relevant columns from select columns, join keys, group-by columns, aggregate columns, or generic task columns. Predicates are not pushed into retrieval.

### 5. Context Rows Are Retrieved Iteratively

`gateway.retrieve_context_data_iterative()` retrieves rows from the FAISS index. It starts with `RETRIEVER_K` nearest documents and increases by `RETRIEVER_K` on each iteration until:

- no new rows are found, or
- `MAX_ITERATIVE_RETRIEVALS` is reached.

Relevant environment variables:

- `CSV_DIR`: directory containing the TPC-H CSV files.
- `FAISS_INDEX_FOLDER`: where the FAISS index is loaded from or saved to.
- `EMB_MODEL`: SentenceTransformer embedding model.
- `EMB_DEVICE`: `auto`, `cpu`, or `cuda`.
- `INDEX_TABLES`: optional comma-separated table allowlist for index building.
- `INDEX_BATCH_SIZE`: document indexing batch size.
- `RETRIEVER_K`: initial retrieval size and per-iteration increment.
- `MAX_ITERATIVE_RETRIEVALS`: maximum retrieval iterations.
- `MAX_TABLES`: maximum number of tables included in retrieved context.

If the FAISS index folder is missing or empty, the first request builds the index from the CSV files. This is why the first planner-first request can be much slower.

### 6. The Leaf Prompt Asks for Row Copying Only

`prompt.build_leaf_prompt()` creates a strict JSON extraction prompt. The model must:

- read only `CONTEXT_DATA["<target_table>"]`
- ignore every other table
- output one item per row in the target table context
- copy the complete row object exactly
- return `[]` if the target table is absent
- avoid SQL, markdown, commentary, projection, filtering, joins, aggregation, sorting, and final result computation

Expected item schema:

```json
{
  "row_id": "nation_2",
  "values": {
    "nation_rownum": "nation_2",
    "n_name": "ARGENTINA",
    "n_nationkey": "1",
    "__rid__": "nation_2"
  }
}
```

### 7. Model Output Is Validated and Partially Parsed

`gateway._call_model_with_retry()` calls Ollama and validates the response. If the response is not a valid leaf JSON array, the gateway retries once with a stricter reminder.

Validation is strict at the leaf level:

- root output must be a JSON array
- each item must contain exactly `row_id` and `values`
- `row_id` must be a string
- `row_id` must exist in the retrieved `CONTEXT_DATA` for the target table
- duplicate valid row ids are rejected
- `values` must exactly equal the source row from context
- `values.__rid__`, when present, must match `row_id`

Even when the complete leaf output is invalid, `_parse_leaf_json_array_partial()` keeps individually valid rows. This lets evaluation measure partial correctness instead of treating every malformed response as totally empty.

### 8. Planner-First Response Shape

The API response content is a JSON string containing:

```json
{
  "sql": "SELECT ...",
  "plan": {
    "query_type": "SELECT",
    "sql": "SELECT ...",
    "leaf_tasks": [],
    "joins": [],
    "post_ops": []
  },
  "leaf_outputs": [
    {
      "table_name": "nation",
      "task": {},
      "retrieval_query": "table: nation | columns: n_name",
      "context_data": {},
      "prompt": "...",
      "output_text": "...",
      "parsed_output": [],
      "parse_error": null,
      "valid_leaf_output": true
    }
  ]
}
```

This shape is designed for debugging and evaluation. It includes the prompt and context, so prediction files can be inspected without replaying the run.

## Current Semantic Boundary

Planner-first currently separates the task into two layers:

1. Model layer: copy candidate source rows for each table from retrieved context.
2. Deterministic layer: intended future layer for joins, filters, grouping, aggregation, projection, ordering, limit, and final provenance construction.

The deterministic layer is represented in `plan.post_ops`, but not executed by the gateway yet.

This matters for interpretation:

- A good leaf output means the model copied the right source rows for that leaf task.
- A valid leaf output does not mean the final SQL answer was computed.
- Evaluation currently focuses on leaf extraction quality, not final query result correctness.

## Evaluation Dataset

The main input dataset is:

```text
evaluation/leaf_node_questions.json
```

Each entry contains:

- `query_id`
- natural-language full query
- SQL full query
- `leaf_tasks`, each with:
  - `table_name`
  - leaf natural-language question
  - leaf SQL, usually `SELECT * FROM <table>;`

The evaluation runner can run either:

- `root`: full SQL queries
- `leaf`: individual leaf SQL tasks
- `both`: root and leaf records

The metrics script currently evaluates only records where `run_mode == "leaf"`.

## Ground Truth Generation

Ground truth is generated with:

```bash
cd evaluation
bash generate_ground_truth_provsql.sh
```

The script expects a Docker container named `provsql-tpch` with PostgreSQL, TPC-H data, and ProvSQL available. It rewrites each SQL query to add:

```sql
provsql.sr_why(provsql.provenance(), 'provmap') AS why_prov
```

It writes:

- `evaluation/ground_truth_queries.json`: full-query ground truth
- `evaluation/ground_truth_leaf_tasks.json`: leaf-task ground truth

For leaf tasks, repeated leaf SQL is cached so each unique leaf SQL is executed once.

## Running Planner-First Evaluation Locally

First ensure Ollama is running and has the model you want to evaluate:

```bash
ollama serve
```

Then run:

```bash
python3 run_planner_first_eval.py \
  --input evaluation/leaf_node_questions.json \
  --output evaluation/planner_first_outputs.jsonl \
  --mode leaf \
  --ollama-model llama3:70b \
  --temperature 0.0
```

Useful options:

- `--mode root|leaf|both`: choose which records to run.
- `--query-id query01`: run only one query id. Can be repeated.
- `--limit N`: run only the first N selected records.
- `--resume` / `--no-resume`: skip or rerun completed successful records.
- `--overwrite`: truncate the output file before running.
- `--verbose-gateway`: print gateway retrieval and prompt logs to the console.
- `--gateway-log-output PATH`: store gateway stdout somewhere specific.
- `--ollama-model MODEL`: use an actual Ollama model name instead of `gateway.MODEL_ROUTING["planner-first"]`.

## Running Internal-Knowledge On The Same Leaf Tasks

The internal-knowledge runner can use the same `root|leaf|both` split as planner-first. To compare against planner-first leaf extraction, run it with `--mode leaf` on `evaluation/leaf_node_questions.json`:

```bash
python3 run_internal_knowledge_eval.py \
  --input evaluation/leaf_node_questions.json \
  --output evaluation/internal_knowledge_leaf_outputs.jsonl \
  --mode leaf \
  --ollama-model llama3:70b \
  --temperature 0.0
```

Then score those leaf records against the leaf-task ground truth:

```bash
python3 evaluate_internal_knowledge_outputs.py \
  --predictions evaluation/internal_knowledge_leaf_outputs.jsonl \
  --ground-truth evaluation/ground_truth_leaf_tasks.json \
  --mode leaf \
  --output evaluation/internal_knowledge_leaf_metrics.json \
  --csv-output evaluation/internal_knowledge_leaf_metrics.csv \
  --plots-dir evaluation/internal_knowledge_leaf_plots
```

For OAR, set the same mode and matching ground truth:

```bash
EVAL_INPUT=evaluation/leaf_node_questions.json \
EVAL_MODE=leaf \
GROUND_TRUTH=evaluation/ground_truth_leaf_tasks.json \
METRICS_MODE=leaf \
EVAL_OUTPUT=evaluation/internal_knowledge_leaf_outputs_${OAR_JOB_ID}.jsonl \
oarsub -l /gpu=1,walltime=4:0:0 './run_oar_internal_knowledge_eval.sh --ollama-model llama3:70b'
```

By default, local evaluation sets:

- `CSV_DIR` to `tpch_no_provsql`
- `FAISS_INDEX_FOLDER` to `faiss_index_tpch`
- `GATEWAY_LOG_PATH` to `logs/provsql_gateway_logs.jsonl`

## Running on OAR

`run_oar_planner_eval.sh` starts an Ollama server inside the job, waits for readiness, activates `.venv` if present, and then runs `run_planner_first_eval.py`.

Example:

```bash
oarsub -l /gpu=1,walltime=4:0:0 './run_oar_planner_eval.sh'
```

Common overrides:

```bash
EVAL_MODE=leaf \
EVAL_OUTPUT=evaluation/planner_first_outputs_8b.jsonl \
OLLAMA_PORT=11434 \
OLLAMA_BIN=/absolute/path/to/ollama \
oarsub -l /gpu=1,walltime=4:0:0 './run_oar_planner_eval.sh --ollama-model llama3:8b'
```

The wrapper writes Ollama logs to:

```text
evaluation/logs/ollama_<OAR_JOB_ID>.log
```

It exports GPU-friendly defaults:

- `EMB_DEVICE=cuda`
- `OLLAMA_REQUEST_TIMEOUT=1800`

## Prediction JSONL Format

Each line in a planner-first evaluation output file is one record:

```json
{
  "record_id": "query01:leaf:0:supplier",
  "query_id": "query01",
  "run_mode": "leaf",
  "question_nl": "...",
  "question_sql": "...",
  "leaf_index": 0,
  "leaf_table_name": "supplier",
  "leaf_question_nl": "List the suppliers.",
  "leaf_question_sql": "SELECT * FROM supplier;",
  "sql_to_run": "SELECT * FROM supplier;",
  "ollama_model": "llama3:70b",
  "temperature": 0.0,
  "started_at": "...",
  "finished_at": "...",
  "elapsed_seconds": 12.345,
  "ok": true,
  "error": null,
  "summary": {
    "leaf_task_count": 1,
    "valid_leaf_task_count": 1,
    "leaf_outputs": []
  },
  "planner_first_result": {}
}
```

The `record_id` is used for resume behavior. Successful records already present in the output JSONL are skipped when `--resume` is enabled.

## Computing Metrics

Run:

```bash
python3 evaluate_planner_first_outputs.py \
  --predictions evaluation/planner_first_outputs.jsonl \
  --ground-truth evaluation/ground_truth_leaf_tasks.json \
  --output evaluation/planner_first_metrics.json \
  --csv-output evaluation/planner_first_metrics.csv \
  --plots-dir evaluation/planner_first_plots
```

To only print metrics:

```bash
python3 evaluate_planner_first_outputs.py \
  --predictions evaluation/planner_first_outputs.jsonl \
  --ground-truth evaluation/ground_truth_leaf_tasks.json \
  --no-write
```

To skip plots:

```bash
python3 evaluate_planner_first_outputs.py \
  --predictions evaluation/planner_first_outputs.jsonl \
  --ground-truth evaluation/ground_truth_leaf_tasks.json \
  --no-plots
```

The evaluator writes:

- JSON report with summary and per-leaf details
- optional CSV details
- optional PNG plots:
  - `summary_metrics.png`
  - `row_confusion_counts.png`
  - `row_f1_by_leaf.png`
  - `content_row_f1_by_leaf.png`
  - `mean_row_f1_by_query.png`

## Metrics Explained

The evaluator matches prediction records to ground-truth leaf tasks by:

- `query_id`
- generated leaf task id
- table name
- normalized leaf SQL

It computes row-id set metrics:

- `tp`: expected row ids that were predicted
- `fp`: predicted row ids absent from ground truth
- `fn`: expected row ids missing from predictions
- `precision`: `tp / (tp + fp)`
- `recall`: `tp / (tp + fn)`
- `row_f1`: harmonic mean of precision and recall
- `row_accuracy`: `tp / (tp + fp + fn)`
- `exact_match`: predicted row-id set equals expected row-id set

It also computes content-level checks:

- `answer_exact_match`: no hallucinated rows, no missing rows, and no value mismatches
- `content_row_accuracy`
- `content_row_f1`
- `hallucinated_row_count`
- `missing_answer_row_count`
- `value_mismatch_count`
- `hallucination_free_rate`

Content checks compare complete row values, not only row ids. This catches cases where the model used a real row id but changed, omitted, or added row content.

## Interpreting Common Results

High precision with very low recall usually means the model copied only a small subset of valid rows and did not invent rows. This is common when the retrieved context contains too few target rows or the model truncates long JSON outputs.

Low `valid_leaf_output_rate` means many leaf outputs did not satisfy the strict schema, even if some rows were partially recoverable.

High `hallucination_free_rate` means the accepted rows are grounded in retrieved context. It does not imply completeness.

Large `missing_answer_row_count` usually means the leaf SQL ground truth is broad, often `SELECT * FROM <table>;`, while retrieval only surfaced a small number of rows.

## Logs and Debugging

Gateway events are written to:

```text
logs/provsql_gateway_logs.jsonl
```

Important event types include:

- `faiss_loaded`
- `faiss_built`
- `iterative_retrieval`
- `iterative_retrieval_final`
- `attempt`
- `planner_first_request`

Evaluation runs also capture gateway stdout by default next to the prediction file:

```text
evaluation/planner_first_outputs.jsonl.gateway_stdout.log
```

Use `--verbose-gateway` when you want to see prompts and retrieval output live.

## Known Limitations

- Only `SELECT` queries are supported by the planner.
- The planner-first endpoint does not yet execute `post_ops`.
- Leaf prompts currently ignore local predicates, so leaf outputs are table-context copies rather than filtered table scans.
- Retrieval is approximate and bounded by `RETRIEVER_K`, `MAX_ITERATIVE_RETRIEVALS`, and context limits.
- Large leaf tasks can exceed practical model output length.
- The evaluation script evaluates leaf records only.
