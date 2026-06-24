#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

mkdir -p evaluation_relf1/logs_internal_knowledge

JOB_ID="${OAR_JOB_ID:-local}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_BIND_HOST="${OLLAMA_BIND_HOST:-127.0.0.1:${OLLAMA_PORT}}"
OLLAMA_API_URL="http://127.0.0.1:${OLLAMA_PORT}"
OLLAMA_LOG="evaluation_relf1/logs_internal_knowledge/ollama_internal_${JOB_ID}.log"
OLLAMA_BIN="${OLLAMA_BIN:-}"

export GATEWAY_LOG_PATH="${GATEWAY_LOG_PATH:-$REPO_ROOT/logs/provsql_gateway_logs.jsonl}"
export OLLAMA_REQUEST_TIMEOUT="${OLLAMA_REQUEST_TIMEOUT:-1800}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${JOB_ID}}"

for profile in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile"; do
  if [[ -f "$profile" ]]; then
    # shellcheck disable=SC1090
    source "$profile" || true
  fi
done

for candidate_dir in "$HOME/cicciara/ollama/bin" "$HOME/ollama/bin"; do
  if [[ -d "$candidate_dir" ]]; then
    export PATH="$candidate_dir:$PATH"
  fi
done

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    kill "$OLLAMA_PID"
    wait "$OLLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[oar-internal-eval] repo: $REPO_ROOT"
echo "[oar-internal-eval] job id: $JOB_ID"

if [[ -z "$OLLAMA_BIN" ]]; then
  OLLAMA_BIN="$(command -v ollama || true)"
fi

if [[ -z "$OLLAMA_BIN" ]]; then
  for candidate in "$HOME/cicciara/ollama/bin/ollama" "$HOME/ollama/bin/ollama" "$HOME/faresa/ollama"; do
    if [[ -x "$candidate" ]]; then
      OLLAMA_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$OLLAMA_BIN" ]]; then
  echo "[oar-internal-eval] ERROR: ollama command not found."
  echo "[oar-internal-eval] Load the Ollama module/package before submitting, or submit with:"
  echo "[oar-internal-eval]   OLLAMA_BIN=/absolute/path/to/ollama oarsub -l /gpu=1,walltime=4:0:0 './run_oar_internal_knowledge_eval.sh'"
  echo "[oar-internal-eval] On many clusters you can inspect available modules with: module avail ollama"
  exit 127
fi

echo "[oar-internal-eval] ollama binary: $OLLAMA_BIN"
echo "[oar-internal-eval] starting ollama on $OLLAMA_BIND_HOST"

OLLAMA_HOST="$OLLAMA_BIND_HOST" "$OLLAMA_BIN" serve >"$OLLAMA_LOG" 2>&1 &
OLLAMA_PID="$!"

echo "[oar-internal-eval] ollama pid: $OLLAMA_PID"
echo "[oar-internal-eval] ollama log: $OLLAMA_LOG"

for attempt in $(seq 1 60); do
  if curl -fsS "$OLLAMA_API_URL/api/tags" >/dev/null 2>&1; then
    echo "[oar-internal-eval] ollama is ready"
    break
  fi

  if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "[oar-internal-eval] ollama exited before becoming ready"
    tail -n 80 "$OLLAMA_LOG" || true
    exit 1
  fi

  if [[ "$attempt" == "60" ]]; then
    echo "[oar-internal-eval] timed out waiting for ollama"
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

EVAL_OUTPUT="${EVAL_OUTPUT:-evaluation_relf1/internal_knowledge_outputs_70b_${JOB_ID}.jsonl}"
METRICS_OUTPUT="${METRICS_OUTPUT:-evaluation_relf1/internal_knowledge_metrics_70b_${JOB_ID}.json}"
METRICS_CSV_OUTPUT="${METRICS_CSV_OUTPUT:-evaluation_relf1/internal_knowledge_metrics_70b_${JOB_ID}.csv}"
METRICS_PLOTS_DIR="${METRICS_PLOTS_DIR:-evaluation_relf1/internal_knowledge_plots_70b_${JOB_ID}}"
RUN_METRICS="${RUN_METRICS:-1}"
EVAL_MODE="${EVAL_MODE:-root}"
METRICS_MODE="${METRICS_MODE:-auto}"
PROMPT_DOMAIN="${PROMPT_DOMAIN:-relf}"

if [[ "$EVAL_MODE" == "leaf" ]]; then
  DEFAULT_EVAL_INPUT="evaluation_relf1/leaf_node_questions.json"
  DEFAULT_GROUND_TRUTH="evaluation_relf1/ground_truth_leaf_tasks.json"
else
  DEFAULT_EVAL_INPUT="evaluation_relf1/questions.json"
  DEFAULT_GROUND_TRUTH="evaluation_relf1/ground_truth_queries.json"
fi

echo "[oar-internal-eval] running internal-knowledge evaluation"
python3 run_internal_knowledge_eval.py \
  --resume \
  --input "${EVAL_INPUT:-$DEFAULT_EVAL_INPUT}" \
  --output "$EVAL_OUTPUT" \
  --mode "$EVAL_MODE" \
  --prompt-domain "$PROMPT_DOMAIN" \
  --ollama-model "${OLLAMA_MODEL_INTERNAL_KNOWLEDGE:-${OLLAMA_MODEL_BASE:-llama3:70b}}" \
  "$@"

if [[ "$RUN_METRICS" == "1" ]]; then
  echo "[oar-internal-eval] computing metrics"
  python3 evaluate_internal_knowledge_outputs.py \
    --predictions "$EVAL_OUTPUT" \
    --ground-truth "${GROUND_TRUTH:-$DEFAULT_GROUND_TRUTH}" \
    --csv-dir "${CSV_DIR:-rel-f1-csv}" \
    --output "$METRICS_OUTPUT" \
    --csv-output "$METRICS_CSV_OUTPUT" \
    --mode "$METRICS_MODE" \
    --plots-dir "$METRICS_PLOTS_DIR"
fi

echo "[oar-internal-eval] done"
