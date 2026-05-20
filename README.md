# Demo Provenance LLM Gateway

This project provides a **local demo environment** for the Provenance LLM Gateway using:

- **FastAPI Gateway** running in Docker
- **Ollama** running in Docker
- **Fine-tuned models (.gguf)** stored with **Git LFS**
- **Local CSV data**
- **FAISS vector index** built automatically on first query


---
# Architecture

The demo runs two services:

- **Gateway**
  - FastAPI application
  - RAG pipeline
  - FAISS indexing
  - Planner first pipeline

- **Ollama**
  - Hosts the base LLM models
  - Hosts the fine-tuned `.gguf` models

Both services are started using **Docker Compose**.

---

## Planner-First Documentation

The detailed documentation for the planner-first pipeline and evaluation workflow is here:

- [Planner-first logic and evaluation](docs/planner_first.md)

It covers SQL planning, leaf extraction, iterative retrieval, output validation, ProvSQL ground-truth generation, local/OAR evaluation runs, metrics, logs, and current limitations.

---

## 1. Prerequisites

Install the following tools:

- **Docker Desktop**
- **Git**
- **Git LFS** (git lfs install)

## 2. Clone the Repository

Clone the demo repository:

```bash
git clone https://github.com/AleCicciarelli/demo_gateway_provenance.git
cd demo_gateway_provenance
```

Pull the **Git LFS model files (.gguf)**:

```bash
git lfs pull
```

These files contain the fine-tuned models used by Ollama.

---

## 3. Start the Services

Start both the **gateway** and **Ollama** services:

```bash
docker compose up -d
```

---

## 4. Verify the Services Are Running

Check that both services are up:

```bash
docker compose ps
```

You should see containers similar to:

```text
NAME            STATUS
demo_gateway    Up
ollama          Up
```

---

## 5. Check Available Models

Query the models API:

```bash
curl http://localhost:9005/v1/models
```

Expected response:

```json
{"object":"list","data":[{"id":"base-llama3-8b","object":"model"},{"id":"best-ft-llama3-8b-nl","object":"model"},{"id":"best-ft-llama3-8b-sql","object":"model"},{"id":"planner-first","object":"model"}]}
```

You should see the following models:

- `base-llama3-8b`
- `best-ft-llama3-8b-nl`
- `best-ft-llama3-8b-sql`
- `planner-first`

---

## 6. Example Request

Example request using the **planner-first** model:

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

---

## 7. First Request Behavior (FAISS Index Creation)

The **first request will take longer** because the system builds the **FAISS index** from the CSV data.

You can monitor the progress in the logs:

```bash
docker compose logs -f gateway
```

If everything is going well, you should see logs like these:

```text
[startup] Waiting for Ollama API at http://ollama:11434...
[startup] Ollama is ready
[startup] Launching gateway...
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
INFO:     192.168.144.1:60558 - "GET /v1/models HTTP/1.1" 200 OK
Alias map: {'nation': 'nation', 'n': 'nation'}

--- Running leaf task for table 'nation' with retrieval query: Retrieve rows from table nation
Indexed 500 documents
Indexed 1000 documents
Indexed 1500 documents
Indexed 2000 documents
...
```

This means the gateway is starting correctly and the indexing process is running.
Once the indexing is complete, the **RAG pipeline** starts.
Subsequent requests will be faster because the FAISS index has already been created.

---

# 8. Notes

- The gateway communicates with Ollama through the Docker network at:

```text
http://ollama:11434
```

- The FAISS index is created automatically on the first request.
- Large models may be slow without GPU acceleration.
- To test more quickly, you can switch to smaller models if needed.
