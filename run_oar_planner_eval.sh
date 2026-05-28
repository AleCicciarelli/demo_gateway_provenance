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

EVAL_MODEL_LABEL="${EVAL_MODEL_LABEL:-70b}"
EVAL_ITERATIONS="${EVAL_ITERATIONS:-4}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-evaluation/planner_first_runs}"
EVAL_RUN_DIR="${EVAL_RUN_DIR:-$EVAL_OUTPUT_ROOT/${EVAL_MODEL_LABEL}_${EVAL_ITERATIONS}iter}"
EVAL_OUTPUT="${EVAL_OUTPUT:-$EVAL_RUN_DIR/outputs.jsonl}"
EVAL_GATEWAY_STDOUT="${EVAL_GATEWAY_STDOUT:-$EVAL_RUN_DIR/gateway_stdout.log}"

case "$EVAL_MODEL_LABEL" in
  8b)
    EVAL_OLLAMA_MODEL="${EVAL_OLLAMA_MODEL:-${OLLAMA_MODEL_8B:-llama3:8b}}"
    ;;
  70b)
    EVAL_OLLAMA_MODEL="${EVAL_OLLAMA_MODEL:-${OLLAMA_MODEL_70B:-llama3:70b}}"
    ;;
  *)
    EVAL_OLLAMA_MODEL="${EVAL_OLLAMA_MODEL:-$EVAL_MODEL_LABEL}"
    ;;
esac

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    kill "$OLLAMA_PID"
    wait "$OLLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[oar-eval] repo: $REPO_ROOT"
echo "[oar-eval] job id: $JOB_ID"
echo "[oar-eval] model label: $EVAL_MODEL_LABEL"
echo "[oar-eval] iterations: $EVAL_ITERATIONS"
echo "[oar-eval] run dir: $EVAL_RUN_DIR"
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
mkdir -p "$EVAL_RUN_DIR"

MAX_ITERATIVE_RETRIEVALS="$EVAL_ITERATIONS" \
GATEWAY_LOG_PATH="${EVAL_GATEWAY_LOG_PATH:-$EVAL_RUN_DIR/gateway_events.jsonl}" \
python3 run_planner_first_eval.py \
  --resume \
  --mode "${EVAL_MODE:-leaf}" \
  --output "$EVAL_OUTPUT" \
  --gateway-log-output "$EVAL_GATEWAY_STDOUT" \
  --ollama-model "$EVAL_OLLAMA_MODEL" \
  "$@"

python3 evaluate_planner_first_outputs.py \
  --predictions "$EVAL_OUTPUT" \
  --output "$EVAL_RUN_DIR/metrics.json" \
  --csv-output "$EVAL_RUN_DIR/metrics.csv" \
  --plots-dir "$EVAL_RUN_DIR/plots"

echo "[oar-eval] done"
