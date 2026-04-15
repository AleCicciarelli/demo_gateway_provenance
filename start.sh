#!/usr/bin/env bash
set -euo pipefail

OLLAMA_HOST_URL="${OLLAMA_HOST:-http://ollama:11434}"

echo "[startup] Waiting for Ollama API at ${OLLAMA_HOST_URL}..."
for i in $(seq 1 60); do
  if curl -fsS "${OLLAMA_HOST_URL}/api/tags" >/dev/null 2>&1; then
    echo "[startup] Ollama is ready"
    break
  fi
  sleep 1
done

if ! curl -fsS "${OLLAMA_HOST_URL}/api/tags" >/dev/null 2>&1; then
  echo "[startup] Ollama API did not become ready"
  exit 1
fi

echo "[startup] Launching gateway..."
exec uvicorn gateway:app --host 0.0.0.0 --port 9000 --access-log