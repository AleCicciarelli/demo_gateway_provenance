# UI - Gateway - Explanation Pipeline Architecture

This note describes the request path used when the UI selects the `planner-only-explanation` model.

## High-Level Flow

```text
UI
 |
 | POST /v1/chat/completions
 | model = planner-only-explanation
 v
Gateway FastAPI app
 |
 | 1. Build planner-only result from the SQL query
 | 2. Convert planner leaf outputs to CSV files
 | 3. Write CSV files into the shared bucket
 v
explanation_pipeline.py
 |
 | Build AP pipeline and connect the components
 v
explanation_client.py
 |
 | POST /api/v1/aps/explanation
 | GET  /api/v1/aps/explanation/{task_id}
 v
ap-explanation service
 |
 | Uses CSV files from shared bucket
 | Uses postgres-provsql for provenance SQL execution
 | 
 v
Gateway returns markdown response to UI
```

## Components

### UI

The UI talks to the gateway through the OpenAI-compatible chat API:

```text
POST http://localhost:9006/v1/chat/completions
```

To trigger the full explanation path, the request uses:

```json
{
  "model": "planner-only-explanation",
  "messages": [
    {
      "role": "user",
      "content": "SELECT ..."
    }
  ]
}
```

### Gateway

File: `gateway.py`

The gateway is the orchestration layer. For `planner-only-explanation`, it:

1. Extracts the SQL query from the user message.
2. Runs the planner-only pipeline.
3. Creates an `ExplanationClient`.
4. Calls `run_planner_first_explanation_pipeline(...)`.
5. Returns the final markdown result as the assistant message content.

The gateway exposes port `9006` on the host and runs on port `9000` inside Docker.

### Explanation Pipeline

File: `explanation_pipeline.py`

The pipeline prepares the data for the AP explanation service:

1. Cleans the shared bucket directory.
2. Converts planner leaf outputs into one CSV file per table.
3. Calls the AP explanation API through `ExplanationClient`.
4. Renders a markdown response containing:
   - the SQL query
   - generated CSV file names
   - the explanation service output

CSV conversion is implemented in `json_to_csv.py`.

### Explanation Client

File: `explanation_client.py`

The client wraps the AP explanation HTTP API:

1. Builds the AP CSV payload using `create_ap_template.py`.
2. Sends the job to the configured AP endpoint.
3. If the service returns a `task_id`, polls until the task succeeds or fails.

Default polling endpoint:

```text
GET {EXPLANATION_URL}/api/v1/aps/explanation/{task_id}
```

### AP Explanation Service

Docker service: `explanation_app`

Image:

```text
ghcr.io/datagems-eosc/datagems-eosc/ap-explanation:v0.11.0
```

The service receives the AP payload, reads the generated CSV files from the shared bucket, executes the provenance explanation workflow, and returns the explanation result.

It depends on:

- `postgres-provsql`
- `redis`

## Shared Storage

The gateway and AP explanation service share CSV files through a local Docker volume bind:

```text
Host path:        ./bucket
Gateway path:     /shared_bucket
AP service path:  /mnt/s3
```

The AP payload refers to generated CSVs with `s3:/<file_name>` content URLs. In this Docker setup, those files are backed by the shared `./bucket` directory.

## Docker Services

Relevant services from `docker-compose.yml`:

```text
gateway
  FastAPI gateway
  Host port: 9005
  Internal port: 9000
  Uses EXPLANATION_URL=http://explanation_app:5000
  Writes generated CSVs to /shared_bucket

explanation_app
  AP explanation service
  Host port: 5002
  Internal port: 5000
  Reads generated CSVs from /mnt/s3

postgres-provsql
  PostgreSQL with ProvSQL support
  Used by explanation_app

redis
  Celery broker/result backend for explanation_app

ollama
  Local LLM runtime used by the gateway planner-only pipeline
```

## Important Environment Variables

Gateway explanation configuration:

```text
EXPLANATION_URL=http://explanation_app:5000
EXPLANATION_ENDPOINT=/api/v1/aps/explanation
EXPLANATION_BUCKET_DIR=/shared_bucket
EXPLANATION_CSV_DELIMITER=,
EXPLANATION_KEEP_ROWNUM=true
EXPLANATION_REQUEST_TIMEOUT=500
```

AP explanation service configuration:

```text
POSTGRES_HOST=postgres-provsql
POSTGRES_DB=mathe
POSTGRES_USER=provdemo
POSTGRES_PASSWORD=provdemo
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
```

## Use Case Scenario

```text
1. UI sends SQL query to gateway with model planner-only-explanation.
2. Gateway runs the planner-only flow and obtains planner_result.
3. explanation_pipeline cleans /shared_bucket.
4. json_to_csv writes table CSVs into /shared_bucket.
5. explanation_client builds an AP payload referencing those CSVs.
6. explanation_client POSTs the payload to explanation_app.
7. explanation_app starts an async explanation task.
8. explanation_client polls until the task is completed.
9. explanation_pipeline renders the result as markdown.
10. Gateway returns that markdown to the UI.
```
