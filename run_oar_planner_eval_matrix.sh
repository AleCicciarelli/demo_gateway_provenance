#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

mkdir -p evaluation_relf1/logs logs

JOB_ID="${OAR_JOB_ID:-local}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_BIND_HOST="${OLLAMA_BIND_HOST:-127.0.0.1:${OLLAMA_PORT}}"
OLLAMA_API_URL="http://127.0.0.1:${OLLAMA_PORT}"
OLLAMA_LOG="evaluation_relf1/logs/ollama_planner_matrix_${JOB_ID}.log"
OLLAMA_BIN="${OLLAMA_BIN:-}"

export CSV_DIR="${CSV_DIR:-$REPO_ROOT/rel-f1-csv}"
export FAISS_INDEX_FOLDER="${FAISS_INDEX_FOLDER:-$REPO_ROOT/faiss_index_relf1_rows_bge_m3}"
export EMB_MODEL="${EMB_MODEL:-BAAI/bge-m3}"
export EMB_STRATEGY="${EMB_STRATEGY:-bge-m3}"
export EMB_DEVICE="${EMB_DEVICE:-cuda}"
export OLLAMA_REQUEST_TIMEOUT="${OLLAMA_REQUEST_TIMEOUT:-1800}"
export OLLAMA_HOST="$OLLAMA_API_URL"

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

EVAL_MODE="${EVAL_MODE:-leaf}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-evaluation_relf1/planner_first_runs}"
EVAL_MODELS="${EVAL_MODELS:-70b}"
EVAL_ITERATIONS="${EVAL_ITERATIONS:-1 2 3 4}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"

model_name_for_label() {
  case "$1" in
    8b)
      printf '%s\n' "${OLLAMA_MODEL_8B:-llama3:8b}"
      ;;
    70b)
      printf '%s\n' "${OLLAMA_MODEL_70B:-llama3:70b}"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    kill "$OLLAMA_PID"
    wait "$OLLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[oar-matrix] repo: $REPO_ROOT"
echo "[oar-matrix] job id: $JOB_ID"
echo "[oar-matrix] models: $EVAL_MODELS"
echo "[oar-matrix] iterations: $EVAL_ITERATIONS"
echo "[oar-matrix] output root: $EVAL_OUTPUT_ROOT"

if [[ -z "$OLLAMA_BIN" ]]; then
  OLLAMA_BIN="$(command -v ollama || true)"
fi

if [[ -z "$OLLAMA_BIN" ]]; then
  for candidate in "$HOME/cicciara/ollama/bin/ollama" "$HOME/ollama/bin/ollama"; do
    if [[ -x "$candidate" ]]; then
      OLLAMA_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$OLLAMA_BIN" ]]; then
  echo "[oar-matrix] ERROR: ollama command not found."
  echo "[oar-matrix] Submit with OLLAMA_BIN=/absolute/path/to/ollama or load the Ollama module first."
  exit 127
fi

echo "[oar-matrix] ollama binary: $OLLAMA_BIN"
echo "[oar-matrix] starting ollama on $OLLAMA_BIND_HOST"

OLLAMA_HOST="$OLLAMA_BIND_HOST" "$OLLAMA_BIN" serve >"$OLLAMA_LOG" 2>&1 &
OLLAMA_PID="$!"

echo "[oar-matrix] ollama pid: $OLLAMA_PID"
echo "[oar-matrix] ollama log: $OLLAMA_LOG"

for attempt in $(seq 1 60); do
  if curl -fsS "$OLLAMA_API_URL/api/tags" >/dev/null 2>&1; then
    echo "[oar-matrix] ollama is ready"
    break
  fi

  if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "[oar-matrix] ollama exited before becoming ready"
    tail -n 80 "$OLLAMA_LOG" || true
    exit 1
  fi

  if [[ "$attempt" == "60" ]]; then
    echo "[oar-matrix] timed out waiting for ollama"
    tail -n 80 "$OLLAMA_LOG" || true
    exit 1
  fi

  sleep 2
done

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

mkdir -p "$EVAL_OUTPUT_ROOT"

for model_label in $EVAL_MODELS; do
  model_name="$(model_name_for_label "$model_label")"

  for iterations in $EVAL_ITERATIONS; do
    run_dir="$EVAL_OUTPUT_ROOT/${model_label}_${iterations}iter"
    mkdir -p "$run_dir"

    echo
    echo "[oar-matrix] running model=$model_label ($model_name) iterations=$iterations"
    echo "[oar-matrix] run dir: $run_dir"

    MAX_ITERATIVE_RETRIEVALS="$iterations" \
    GATEWAY_LOG_PATH="$run_dir/gateway_events.jsonl" \
    python3 run_planner_first_eval.py \
      --resume \
      --mode "$EVAL_MODE" \
      --output "$run_dir/outputs.jsonl" \
      --gateway-log-output "$run_dir/gateway_stdout.log" \
      --ollama-model "$model_name" \
      --temperature "$EVAL_TEMPERATURE" \
      "$@"

    python3 evaluate_planner_first_outputs.py \
      --predictions "$run_dir/outputs.jsonl" \
      --output "$run_dir/metrics.json" \
      --csv-output "$run_dir/metrics.csv" \
      --plots-dir "$run_dir/plots"
  done
done

echo "[oar-matrix] done"
