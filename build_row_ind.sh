#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/demo_gateway"   

source .venv/bin/activate


JOB_TMP="${TMPDIR:-/tmp}/faiss_${OAR_JOB_ID:-manual}"
mkdir -p "$JOB_TMP"
export TMPDIR="$JOB_TMP"
export HF_HOME="${HF_HOME:-$JOB_TMP/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$JOB_TMP/sentence_transformers}"
export TORCH_HOME="${TORCH_HOME:-$JOB_TMP/torch}"
export TQDM_DISABLE=1
LOG_PATH="$JOB_TMP/faiss_build_events.jsonl"
CHECKPOINT_EVERY_BATCHES="${FAISS_CHECKPOINT_EVERY_BATCHES:-25}"
ENCODE_BATCH_SIZE="${FAISS_ENCODE_BATCH_SIZE:-64}"
FAISS_DEVICE="${FAISS_DEVICE:-cpu}"

echo "=== Rebuilding FAISS vectors from row documents ==="
mkdir -p faiss_index_relf1_rows_bge_m3_docs

python3 build_row_index.py \
     --csv_dir rel-f1-csv \
     --schema_profile rel-f1-csv/schema_profile_relf1.json \
     --sep "," \
     --documents_out faiss_index_relf1_rows_bge_m3_docs/row_documents_relf1.jsonl \
     --documents-only

python3 build_row_faiss_index.py \
     --documents faiss_index_relf1_rows_bge_m3_docs/row_documents_relf1.jsonl \
     --index-folder faiss_index_relf1_rows_bge_m3 \
     --checkpoint-folder faiss_index_relf1_rows_bge_m3.checkpoint \
     --embedding-model BAAI/bge-m3 \
     --device "$FAISS_DEVICE" \
     --batch-size 256 \
     --encode-batch-size "$ENCODE_BATCH_SIZE" \
     --checkpoint-every-batches "$CHECKPOINT_EVERY_BATCHES" \
     --no-resume

echo "=== Row index completed ==="
