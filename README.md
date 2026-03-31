# Demo Provenance LLM Gateway

This project provides a local demo environment for the gateway, using:
- a FastAPI gateway running in Docker
- Ollama running locally on the host machine
- fine-tuned models recreated locally in Ollama
- local CSV data and FAISS index

## 1. Prerequisites
Install:
- Docker Desktop
- Ollama

Verify Ollama is running:

```bash
curl http://localhost:11434/api/tags
```
## 2. Create the fine-tuned models in Ollama 
Example fine tuned model (natural language dataset):
```bash
cd models/ft_nl
ollama create llama3-8b-dpo2-sft1-nl -f Modelfile
```

Example fine tuned model (SQL dataset):
```bash
cd ../ft_sql
ollama create llama3-8b-dpo1-sft2-sql -f Modelfile
```
Verify:
```bash
ollama list
```

## 3. Configure environment
There is an example env file, copy this into an .env fileadjusting names if needed.

```bash
cp .env.example .env
```

## 4. Start the gateway
```bash
docker compose up
```
The gateway will be available at:
http://localhost:9000

## 5. Check the available models 
```bash
curl http://localhost:9000/v1/models
```

This should show the following models:

- base-llama3-8b
- best-ft-llama3-8b-nl
- best-ft-llama3-8b-sql
- planner-first

## 6. Example planner first request using curl:
The question should be written in SQL (for the moment):

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
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

For the debug part, open in browser:
http://localhost:9000/debug/ui

## 8. Notes
- The gateway container connects to Ollama running on the host through:
http://host.docker.internal:11434

- Large models may be slow without using GPU(you can change the models in the gateway.py file to test everything with smaller models).
