#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_NL="${OLLAMA_MODEL_NAME_NL:-llama3-8b-dpo2-sft1-nl:latest}"
MODEL_DIR_NL="${OLLAMA_MODEL_DIR_NL:-/app/models/ft_nl}"
MODEL_NAME_SQL="${OLLAMA_MODEL_NAME_SQL:-llama3-8b-dpo1-sft2-sql:latest}"
MODEL_DIR_SQL="${OLLAMA_MODEL_DIR_SQL:-/app/models/ft_sql}"
OLLAMA_HOST_URL="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"

echo "[startup] Starting Ollama..."
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

cleanup() {
  echo "[startup] Shutting down..."
  if kill -0 "$OLLAMA_PID" 2>/dev/null; then
    kill "$OLLAMA_PID" || true
  fi
}
trap cleanup EXIT

echo "[startup] Waiting for Ollama API..."
for i in $(seq 1 60); do
  if curl -fsS "${OLLAMA_HOST_URL}/api/tags" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "${OLLAMA_HOST_URL}/api/tags" >/dev/null 2>&1; then
  echo "[startup] Ollama API did not become ready"
  cat /tmp/ollama.log || true
  exit 1
fi

echo "[startup] Checking whether model '${MODEL_NAME_NL}' already exists..."
if curl -fsS "${OLLAMA_HOST_URL}/api/tags" | grep -q "\"name\":\"${MODEL_NAME_NL}\""; then
  echo "[startup] Model already present: ${MODEL_NAME_NL}"
else
  echo "[startup] Creating model: ${MODEL_NAME_NL}"
  cd "${MODEL_DIR_NL}"
  ollama create "${MODEL_NAME_NL}" -f Modelfile
fi

echo "[startup] Checking whether model '${MODEL_NAME_SQL}' already exists..."
if curl -fsS "${OLLAMA_HOST_URL}/api/tags" | grep -q "\"name\":\"${MODEL_NAME_SQL}\""; then
  echo "[startup] Model already present: ${MODEL_NAME_SQL}"
else
  echo "[startup] Creating model: ${MODEL_NAME_SQL}"
  cd "${MODEL_DIR_SQL}"
  ollama create "${MODEL_NAME_SQL}" -f Modelfile
fi

echo "[startup] Launching gateway..."
exec uvicorn gateway:app --host 0.0.0.0 --port 9000 --access-log