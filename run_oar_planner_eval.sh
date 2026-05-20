#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

mkdir -p evaluation/logs logs

JOB_ID="${OAR_JOB_ID:-local}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_BIND_HOST="${OLLAMA_BIND_HOST:-127.0.0.1:${OLLAMA_PORT}}"
OLLAMA_API_URL="http://127.0.0.1:${OLLAMA_PORT}"
OLLAMA_LOG="evaluation/logs/ollama_${JOB_ID}.log"
OLLAMA_BIN="${OLLAMA_BIN:-}"

export CSV_DIR="${CSV_DIR:-$REPO_ROOT/tpch_no_provsql}"
export FAISS_INDEX_FOLDER="${FAISS_INDEX_FOLDER:-$REPO_ROOT/faiss_index_tpch}"
export GATEWAY_LOG_PATH="${GATEWAY_LOG_PATH:-$REPO_ROOT/logs/provsql_gateway_logs.jsonl}"
export EMB_DEVICE="${EMB_DEVICE:-cuda}"
export OLLAMA_REQUEST_TIMEOUT="${OLLAMA_REQUEST_TIMEOUT:-1800}"

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    kill "$OLLAMA_PID"
    wait "$OLLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[oar-eval] repo: $REPO_ROOT"
echo "[oar-eval] job id: $JOB_ID"
if [[ -z "$OLLAMA_BIN" ]]; then
  OLLAMA_BIN="$(command -v ollama || true)"
fi

if [[ -z "$OLLAMA_BIN" ]]; then
  echo "[oar-eval] ERROR: ollama command not found."
  echo "[oar-eval] Load the Ollama module/package before submitting, or submit with:"
  echo "[oar-eval]   OLLAMA_BIN=/absolute/path/to/ollama oarsub -l /gpu=1,walltime=4:0:0 './run_oar_planner_eval.sh'"
  echo "[oar-eval] On many clusters you can inspect available modules with: module avail ollama"
  exit 127
fi

echo "[oar-eval] ollama binary: $OLLAMA_BIN"
echo "[oar-eval] starting ollama on $OLLAMA_BIND_HOST"

OLLAMA_HOST="$OLLAMA_BIND_HOST" "$OLLAMA_BIN" serve >"$OLLAMA_LOG" 2>&1 &
OLLAMA_PID="$!"

echo "[oar-eval] ollama pid: $OLLAMA_PID"
echo "[oar-eval] ollama log: $OLLAMA_LOG"

for attempt in $(seq 1 60); do
  if curl -fsS "$OLLAMA_API_URL/api/tags" >/dev/null 2>&1; then
    echo "[oar-eval] ollama is ready"
    break
  fi

  if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "[oar-eval] ollama exited before becoming ready"
    tail -n 80 "$OLLAMA_LOG" || true
    exit 1
  fi

  if [[ "$attempt" == "60" ]]; then
    echo "[oar-eval] timed out waiting for ollama"
    tail -n 80 "$OLLAMA_LOG" || true
    exit 1
  fi

  sleep 2
done

export OLLAMA_HOST="$OLLAMA_API_URL"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

echo "[oar-eval] running planner-first evaluation"
python3 run_planner_first_eval.py \
  --resume \
  --mode "${EVAL_MODE:-leaf}" \
  --output "${EVAL_OUTPUT:-evaluation/planner_first_outputs_70b_3.jsonl}" \
  "$@"

echo "[oar-eval] done"
