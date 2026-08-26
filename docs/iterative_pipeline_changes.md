# Iterative RAG Pipeline Changes

This document summarizes the final changes made to the iterative RAG pipeline.
It covers model-status reporting, retrieval-query design, mixed-table context,
join bindings, model output validation, and runtime performance.

## 1. Clear Model Status

The UI previously displayed the configured model and provider as though the
model were already processing the request:

```text
Model: openai/kimi-k2.6
Provider: openai
```

That information did not prove that the gateway had connected to Kimi or
received a response.

The UI now distinguishes three states:

- **Requested model/provider** while a request is being attempted.
- **Fallback model/provider** when the gateway switches to Ollama.
- **Active model/provider** only after a response has been received.

While waiting, the heartbeat reports:

```text
Attempting openai/kimi-k2.6 for races;
no response received yet (30s elapsed).
```

When fallback begins, it reports:

```text
No response from openai/kimi-k2.6;
falling back to Ollama llama3:8b.
```

## 2. Compact Retrieval Queries

The previous retrieval query mixed retrieval criteria with verbose provenance:

```text
Retrieve rows from circuits...
linked to races_1038, races_1057...
source values: full race rows...
target table: circuits...
```

The new query contains only information intended to influence retrieval:

```text
Table: circuits.
Columns: name, circuitId.
Join filter: circuitId in (3, 77)
```

A first-leaf query with local predicates looks like:

```text
Table: races.
Columns: raceId, circuitId, year, name.
Filters: year = 2021 and ...
```

Source row IDs and complete source values remain available in iterative state
and UI progress details. They are not included in the text embedded for FAISS.

This separates two responsibilities:

```text
Retrieval query: what evidence should be retrieved?
Provenance state: which earlier rows produced the join bindings?
```

## 3. RAG Remains Active for Every Leaf

Every `rag-iterative` leaf continues to use RAG, including later leaves such as
`circuits`:

```text
compact retrieval query
    ↓
FAISS semantic retrieval
    ↓
mixed-table context
    ↓
LLM row extraction
```

There is no direct CSV lookup and no deterministic replacement for RAG.

## 4. Mixed-Table Retrieval Is Preserved

FAISS searches the shared multi-table index. Retrieved rows are not discarded
because they belong to a table other than the current leaf's target table.

Relationally correlated-row expansion also remains enabled. The model may use
rows from any retrieved table as evidence, but its prompt specifies a target:

```text
TARGET_TABLE: circuits
```

The model must emit only rows from that target table:

```text
FAISS → broad relational evidence
LLM   → target-table selection
```

## 5. Join-Binding Creation and Propagation

The SQL planner creates a join edge such as:

```text
races.circuitId ↔ circuits.circuitId
```

After the model selects race rows, the controller extracts the join-key values:

```text
races_1038.circuitId = 3
races_1057.circuitId = 77
```

It creates the binding:

```json
{
  "circuitId": ["3", "77"]
}
```

The controller also records which source row produced each binding. The
bindings are then included in:

- the compact retrieval query for the next leaf;
- the next leaf's model prompt;
- the UI explanation;
- iterative provenance state.

## 6. Full-Row Model Responses

The iterative model returns complete selected rows:

```json
[
  {
    "row_id": "circuits_3",
    "values": {
      "circuitId": "3",
      "name": "Bahrain International Circuit",
      "__rid__": "circuits_3"
    }
  }
]
```

The gateway validates that:

- the row ID exists in the retrieved target-table context;
- each item contains exactly `row_id` and `values`;
- `values` is the complete row object;
- the values exactly match the retrieved source row;
- `__rid__`, when present, matches `row_id`;
- duplicate row IDs are rejected.

A temporary row-ID-only output design was considered and then reverted. The
final implementation retains complete rows.

## 7. Faster Kimi Failure Detection

The previous configuration allowed the Kimi request to wait for 1,800 seconds.
The gateway now uses separate connection and response timeouts:

```yaml
LLM_CONNECT_TIMEOUT: "5"
LLM_READ_TIMEOUT: "120"
```

The connection timeout limits how long the gateway waits to establish access
to the private endpoint. The read timeout limits how long it waits for a model
response after connecting.

## 8. Kimi Circuit Breaker

After a Kimi request fails, the gateway opens a circuit breaker:

```yaml
LLM_CIRCUIT_BREAKER_SECONDS: "300"
```

For the next five minutes, subsequent model calls skip the unavailable Kimi
endpoint and proceed directly to the configured Ollama fallback. This prevents
every leaf from repeating the same slow VPN-dependent connection attempt.

A successful Kimi response closes the circuit again.

## 9. Ollama Timeout

Ollama calls now have an explicit timeout:

```yaml
OLLAMA_REQUEST_TIMEOUT: "700"
```

The local timeout remains deliberately longer than the remote-provider timeout.
CPU inference for full-row JSON responses can exceed three minutes; a shorter
timeout cancels a healthy generation and causes Ollama to log
`srv stop: cancel task`.

GPU access was not enabled automatically because doing so could prevent the
stack from starting on a host without a compatible NVIDIA container runtime.

## 10. One FAISS Search per Leaf

The previous iterative retrieval behavior performed progressively larger
searches using the same query:

```text
top 5 search
top 10 search
```

The second search already contained the first search's neighbors and could
embed the same query again.

The pipeline now performs one search with the complete candidate budget:

```yaml
RETRIEVER_CANDIDATE_K: "10"
```

Mixed-table processing and correlated-row expansion are applied to the result
of that single search.

## 11. Reduced Logging Overhead

Full prompts and model outputs were previously printed to container output and
stored in JSONL attempt logs. This could generate substantial Docker logging
and disk I/O.

Verbose logging is now disabled by default:

```yaml
VERBOSE_MODEL_LOGS: "false"
```

Normal attempt logs retain compact diagnostic information:

- model and provider;
- attempt number;
- validation result and error;
- prompt and output lengths;
- request duration.

Set `VERBOSE_MODEL_LOGS=true` when complete prompts and model responses are
needed for debugging.

## 12. Stage Timings

Progress events now include durations for:

- retrieval;
- individual model requests;
- completed model extraction.

The UI displays values such as:

```text
Duration: 1.24s
```

This helps distinguish retrieval latency, remote Kimi latency, Ollama fallback
latency, and model-validation retries.

## 13. Final Runtime Flow

The final pipeline is:

```text
SQL plan
    ↓
Compact RAG query
    ↓
One shared-index FAISS search
    ↓
Mixed-table and correlated context
    ↓
LLM returns complete target-table rows
    ↓
Gateway validates complete rows
    ↓
Controller propagates join bindings
    ↓
Repeat retrieval and extraction for the next leaf
    ↓
Execute final SQL and provenance processing
```

## 14. Modified Files

- `gateway.py`
  - split model timeouts;
  - Kimi circuit breaker and Ollama fallback events;
  - one-call FAISS retrieval;
  - compact model logging;
  - retrieval and model timings.

- `iterative_join_pipeline.py`
  - compact retrieval-query construction;
  - separation of provenance details from embedded query text.

- `ui/app.js`
  - requested, fallback, and active model states;
  - duration display.

- `docker-compose.yml`
  - connection, read, Ollama, and circuit-breaker timeouts;
  - retrieval candidate budget;
  - verbose-logging configuration.

- `docs/iterative_join_aware_pipeline.md`
  - updated retrieval and runtime architecture.
