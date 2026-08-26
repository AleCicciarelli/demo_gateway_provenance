# Iterative Join-Aware Pipeline

This document explains the `iterative-join-aware` pipeline. The goal of this
mode is to keep the table-level leaf abstraction, while making leaf execution
stateful across joins.

The regular planner-only flow runs each leaf independently:

```text
races leaf   -> retrieval/model extraction
results leaf -> retrieval/model extraction
drivers leaf -> retrieval/model extraction
```

That is weak for join queries because the `results` retrieval step does not know
which `raceId` was found in `races`, and the `drivers` retrieval step does not
know which `driverId` was found in `results`.

The iterative join-aware flow still runs one leaf at a time, but it carries
bindings forward:

```text
races leaf
  -> extract raceId values
  -> use them in results retrieval
results leaf
  -> extract driverId values
  -> use them in drivers retrieval
drivers leaf
  -> produce final supporting rows
```

The pipeline is selected as a query-level behavior, but it emits the same
`leaf_outputs` shape as the older planner pipelines. This keeps CSV generation
and AP explanation compatible.

## Main Files

- `iterative_join_pipeline.py`: query-level controller, join graph, state, and
  dynamic retrieval-query construction.
- `gateway.py`: wires the controller into the UI routes, streaming route, model
  routing, retrieval callback, leaf model callback, CSV generation, and AP
  explanation.
- `ui/app.js`: exposes the `Iterative join-aware` option in the leaf pipeline
  dropdown.
- `json_to_csv.py`: unchanged; it writes CSV files from the normal
  `leaf_outputs` contract.
- `explanation_pipeline.py`: unchanged; it sends generated CSV files to the AP
  explanation service.

## When It Runs

The UI still shows pipeline choices per leaf. If any leaf is assigned:

```text
iterative-join-aware
```

then the backend treats this as a whole-query execution mode and runs the full
plan through the iterative controller. The controller marks every produced leaf
output with the same pipeline id.

The OpenAI-compatible route also accepts the model id:

```text
iterative-join-aware
```

In that path, the gateway builds a plan, runs the iterative controller, writes
CSV files, and calls the AP explanation service.

## Runtime Flow

### 1. SQL Is Planned Normally

The gateway still calls:

```python
plan = build_query_plan(sql_query)
```

The iterative pipeline uses the existing planner output:

- `leaf_tasks`: one task per table.
- `joins`: join conditions and join columns.
- `post_ops`: projection, grouping, aggregation, ordering, and limit metadata.

No new SQL parser is introduced for this pipeline.

### 2. Join Edges Are Built

`iterative_join_pipeline._join_edges_from_plan()` reads `plan["joins"]`.

For this SQL:

```sql
SELECT d.surname
FROM races r
JOIN results res ON r.raceId = res.raceId
JOIN drivers d ON res.driverId = d.driverId
WHERE r.year = 2021
  AND r.name = 'Monaco Grand Prix'
  AND res.positionOrder = 1;
```

the controller builds edges equivalent to:

```text
races.raceId  <-> results.raceId
results.driverId <-> drivers.driverId
```

Each edge stores:

```python
JoinEdge(
    left_table="races",
    left_column="raceId",
    right_table="results",
    right_column="raceId",
    on_sql="r.raceId = res.raceId",
)
```

### 3. Query State Is Initialized

The controller keeps an `IterativeJoinState`:

```python
selected_rows_by_table
bindings
binding_sources
completed_tables
```

The most important structure is `bindings`. It records values learned from
previous leaf outputs that should guide future retrieval.

Example:

```python
bindings["results"]["raceId"] = {"1056"}
bindings["drivers"]["driverId"] = {"830"}
```

`binding_sources` remembers which row produced each value:

```python
binding_sources["results"]["raceId"]["1056"] = {"races_1042"}
```

This source information is included in later retrieval queries as linked-row
context.

### 4. The Next Leaf Is Chosen

`_choose_next_leaf()` scores remaining leaves using:

1. whether the table already has inherited bindings;
2. how many local predicates the leaf has;
3. how many join keys the leaf has.

For the Monaco query, the first leaf is usually `races`, because it has the
strongest local predicates:

```text
year = 2021
name = 'Monaco Grand Prix'
```

After `races` produces a `raceId` binding, `results` becomes the best next leaf.
After `results` produces a `driverId` binding, `drivers` becomes the next leaf.

### 5. A Dynamic Retrieval Query Is Built

The controller builds a compact retrieval description containing only the
target table, needed columns, local filters, and join bindings. Source row ids
and source-row summaries remain available as provenance, but are not embedded
into the semantic retrieval query.

For the first leaf:

```text
Table: races. Columns: raceId, year, name.
Filters: year = 2021 and name = 'Monaco Grand Prix'
```

After a `races` row is selected, the `results` leaf receives a more specific
retrieval query:

```text
Table: results. Columns: raceId, driverId, positionOrder.
Filters: positionOrder = 1. Join filter: raceId = 1056
```

After a `results` row is selected, the `drivers` leaf receives:

```text
Table: drivers. Columns: surname, driverId. Join filter: driverId = 830
```

Every leaf uses FAISS semantic retrieval over the shared multi-table index.
Later leaves include compact inherited join bindings in the retrieval text,
while source-row provenance remains outside the embedded query. Retrieved rows
from other tables remain in context as relational evidence; the leaf model uses
the target table and bindings as prompt constraints and emits only target-table
rows.

### 6. Leaf Execution Uses Existing Gateway Logic

The controller does not know how to call FAISS or the model directly. Instead,
`gateway._run_ui_iterative_join_pipeline()` passes a callback:

```python
def run_leaf(task, retrieval_query):
    ctx = retrieve_context_data_iterative(retrieval_query, dataset=dataset)
    return _run_leaf_task(...)
```

This means each iterative leaf still uses the existing:

- FAISS row retrieval;
- mixed-table row context;
- correlated-row context expansion;
- guided iterative leaf prompt from `prompt.py`;
- model call and retry logic;
- strict JSON validation and partial parsing.

Source-row values from previous leaves are kept in iterative state and UI
events for traceability. They are not included in FAISS text. Retrieval context
may contain several tables and relationally correlated rows, while the prompt
requires output rows to come only from the target table.

### 7. Row Selection Happens in the Leaf Prompt

The iterative controller does not filter `parsed_output` after the leaf model
returns it. Instead, the iterative leaf prompt asks the model to inspect the full
retrieved context and select target-table rows using:

- local predicates from the leaf task;
- inherited join bindings from earlier leaves;
- source row ids that produced those bindings;
- other retrieved tables as supporting evidence.

This is intentional. The pipeline uses retrieval plus model-side selection to
collect candidate evidence rows, but it does not silently remove rows after the
model has returned them.

If a leaf returns multiple candidate rows, all of those rows remain in
`parsed_output`, and all eligible join-key values are propagated forward.

The AP explanation service remains responsible for evaluating the actual SQL
over the generated CSV files.

### 8. Bindings Are Propagated

After each leaf finishes, `_record_selected_rows()` stores its returned rows.
Then `_propagate_bindings()` follows the join graph.

If the selected `races` row contains:

```text
raceId = 1056
```

and the join graph contains:

```text
races.raceId <-> results.raceId
```

then the controller records:

```python
bindings["results"]["raceId"].add("1056")
```

The same mechanism propagates:

```text
results.driverId = 830
```

into:

```python
bindings["drivers"]["driverId"].add("830")
```

### 9. Normal Leaf Outputs Are Returned

The controller returns:

```json
{
  "dataset": "relf1",
  "sql": "SELECT ...",
  "plan": {},
  "leaf_outputs": [],
  "iterative_join": {
    "join_edges": [],
    "bindings": {},
    "completed_tables": []
  }
}
```

Each leaf output keeps the existing structure:

```json
{
  "table_name": "results",
  "task": {},
  "retrieval_query": "... raceId = 1056 ...",
  "context_data": {},
  "prompt": "...",
  "output_text": "...",
  "parsed_output": [],
  "parse_error": null,
  "valid_leaf_output": true,
  "pipeline": "iterative-join-aware",
  "iterative_join": {
    "step": 2,
    "inherited_bindings": {
      "raceId": ["1056"]
    },
    "source_row_ids": ["races_1042"]
  }
}
```

Because the `leaf_outputs` contract is preserved, downstream CSV generation does
not need special handling.

### 10. CSVs and AP Explanation Are Reused

After the iterative pipeline returns leaf outputs, the gateway uses the same
functions as the existing UI run:

```python
_generate_ui_csv_files(...)
_run_ui_ap_explanation_for_csv_files(...)
```

`json_to_csv.planner_result_to_csv_files()` writes one CSV per table from the
rows in `parsed_output`.

The AP explanation service then executes the SQL over these generated CSVs and
computes the final answer and provenance.

## Runtime and Failure Handling

The OpenAI-compatible provider uses separate connection and response budgets:

```text
LLM_CONNECT_TIMEOUT=5
LLM_READ_TIMEOUT=120
```

After a provider failure, a circuit breaker sends subsequent model calls
directly to the configured Ollama fallback for
`LLM_CIRCUIT_BREAKER_SECONDS` (300 seconds by default). This avoids repeating a
long request to a VPN-only endpoint for every leaf.

Retrieval requests the complete candidate budget in one FAISS call. The former
progressive `k`, `2k`, ... loop embedded and searched the same query repeatedly,
even though its final result already contained the earlier neighbors. Configure
the one-call budget with `RETRIEVER_CANDIDATE_K`.

For iterative leaves, the model returns the selected row IDs together with the
complete source-row values. The gateway validates both the IDs and exact row
contents against target-table context before binding propagation and CSV
generation.

Full prompt and model-output logging is disabled by default. Set
`VERBOSE_MODEL_LOGS=true` only for detailed debugging. Progress events include
retrieval and model durations for UI diagnosis.

## Division of Responsibility

The iterative pipeline is responsible for evidence discovery:

```text
Use the query plan and previously found rows to retrieve better candidate rows
for later leaves.
```

The AP explanation service is responsible for SQL semantics:

```text
Execute the SQL over the generated CSVs and compute answer provenance.
```

This means the iterative controller does not implement final query semantics
such as:

- joins;
- projection;
- grouping;
- aggregation;
- ordering;
- limit;
- SQL duplicate behavior.

It only uses join keys to improve retrieval for subsequent leaves.

## Example Trace

For:

```sql
SELECT d.surname
FROM races r
JOIN results res ON r.raceId = res.raceId
JOIN drivers d ON res.driverId = d.driverId
WHERE r.year = 2021
  AND r.name = 'Monaco Grand Prix'
  AND res.positionOrder = 1;
```

an ideal trace is:

```text
Step 1: races
  retrieval includes:
    year = 2021
    name = 'Monaco Grand Prix'
  selected row:
    races_1042, raceId = 1056
  propagated binding:
    results.raceId = 1056

Step 2: results
  retrieval includes:
    positionOrder = 1
    raceId = 1056
    linked to races_1042
  selected row:
    results_25042, driverId = 830
  propagated binding:
    drivers.driverId = 830

Step 3: drivers
  retrieval includes:
    driverId = 830
    linked to results_25042
  selected row:
    drivers_830, surname = Verstappen
```

The generated CSVs should contain at least these rows:

```text
races.csv:
  races_1042

results.csv:
  results_25042

drivers.csv:
  drivers_830
```

The AP service then runs the original SQL over those CSVs.

## Logging

The controller emits log events through the gateway logger:

```text
iterative_join_leaf_start
iterative_join_leaf_done
```

The start event includes:

- dataset;
- SQL query;
- step number;
- table;
- dynamic retrieval query;
- inherited bindings;
- source row ids.

The done event includes:

- dataset;
- SQL query;
- step number;
- table;
- number of parsed rows;
- current bindings.

These events make it possible to inspect whether each leaf received the expected
join-guided retrieval query.

## UI Feedback

The UI uses the streaming route `/ui/run/stream` for pipeline progress. When the
iterative pipeline is selected, the backend emits progress events for the
coordinated query run and then for each produced leaf step.

For each iterative leaf, the UI progress card shows:

- table name;
- pipeline id;
- iterative step number;
- number of parsed rows returned by the leaf;
- dynamic retrieval query;
- inherited join bindings;
- source row ids that produced those bindings;
- retrieved context preview.

This lets the user inspect how the controller moved through the join graph. For
example, the `results` step can show that retrieval was guided by:

```text
join bindings: raceId = 1056
linked to source rows: races_1042
```

The feedback is meant for debugging and transparency. It does not imply that the
gateway has executed the final SQL. Final SQL semantics are still handled by the
AP explanation service after CSV generation.

## Current Limitations

- Join edge extraction assumes join columns appear in pairs in `on_columns`.
- The controller propagates bindings from every row returned by the leaf model.
- There is no post-filtering inside the controller; row selection is prompt-side.
- Broad leaves can therefore produce many bindings and widen later retrieval.
- Complex join predicates beyond simple equality may need richer join-edge
  extraction.
- The leaf prompt still asks the model to copy rows from retrieved context, not
  to evaluate SQL.
- Final query correctness depends on AP explanation receiving enough candidate
  rows in the generated CSV files.

## Why This Is Useful

The main benefit is improved retrieval for join queries. Instead of retrieving
each table from only the original leaf task, later retrieval steps include IDs
and row identifiers discovered earlier.

This is especially useful for queries where one selective table determines the
small set of rows needed from a much larger joined table.

The design keeps the system modular:

```text
planner.py
  -> decides table tasks and joins

iterative_join_pipeline.py
  -> coordinates leaf order and join-guided retrieval

gateway.py
  -> performs retrieval and model calls

json_to_csv.py
  -> writes evidence CSVs

AP explanation service
  -> executes SQL and computes provenance
```
