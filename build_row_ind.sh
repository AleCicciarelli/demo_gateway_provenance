#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
else
  echo "No virtual environment found at $VENV_DIR; using the current Python environment."
fi


JOB_TMP="${TMPDIR:-/tmp}/faiss_${OAR_JOB_ID:-manual}"
mkdir -p "$JOB_TMP"
export TMPDIR="$JOB_TMP"
export HF_HOME="${HF_HOME:-$JOB_TMP/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$JOB_TMP/sentence_transformers}"
export TORCH_HOME="${TORCH_HOME:-$JOB_TMP/torch}"
export TQDM_DISABLE=1

# The gateway image preloads BGE-M3 in HF_HOME/hub. Older Transformers
# configuration may point at a separate, incomplete TRANSFORMERS_CACHE tree.
if [[ -f "$HF_HOME/hub/models--BAAI--bge-m3/refs/main" ]]; then
  export TRANSFORMERS_CACHE="$HF_HOME/hub"
  export SENTENCE_TRANSFORMERS_HOME="$HF_HOME/hub"
fi

LOG_PATH="$JOB_TMP/faiss_build_events.jsonl"
CHECKPOINT_EVERY_BATCHES="${FAISS_CHECKPOINT_EVERY_BATCHES:-25}"
ENCODE_BATCH_SIZE="${FAISS_ENCODE_BATCH_SIZE:-64}"
FAISS_DEVICE="${FAISS_DEVICE:-cpu}"
ROW_TEXTUALIZATION_STRATEGY="${ROW_TEXTUALIZATION_STRATEGY:-rich}"
ROW_DOCUMENTS_OUT="${ROW_DOCUMENTS_OUT:-faiss_index_relf1_rows_bge_m3_docs/row_documents_relf1.jsonl}"
ROW_INDEX_FOLDER="${ROW_INDEX_FOLDER:-faiss_index_relf1_rows_bge_m3}"
ROW_CHECKPOINT_FOLDER="${ROW_CHECKPOINT_FOLDER:-${ROW_INDEX_FOLDER}.checkpoint}"

echo "=== Rebuilding FAISS vectors from row documents ==="
mkdir -p faiss_index_relf1_rows_bge_m3_docs

python3 build_row_index.py \
     --csv_dir rel-f1-csv \
     --schema_profile rel-f1-csv/schema_profile_relf1.json \
     --sep "," \
     --documents_out "$ROW_DOCUMENTS_OUT" \
     --textualization-strategy "$ROW_TEXTUALIZATION_STRATEGY" \
     --documents-only

python3 build_row_faiss_index.py \
     --documents "$ROW_DOCUMENTS_OUT" \
     --index-folder "$ROW_INDEX_FOLDER" \
     --checkpoint-folder "$ROW_CHECKPOINT_FOLDER" \
     --embedding-model BAAI/bge-m3 \
     --device "$FAISS_DEVICE" \
     --batch-size 256 \
     --encode-batch-size "$ENCODE_BATCH_SIZE" \
     --checkpoint-every-batches "$CHECKPOINT_EVERY_BATCHES" \
     --no-resume

echo "=== Row index completed ==="
