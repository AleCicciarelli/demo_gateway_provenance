#!/bin/bash

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

echo "=== Building FAISS index with BGE-M3 ==="

python3 build_faiss_index.py \
  --csv-dir tpch_no_provsql \
  --index-folder faiss_index_tpch_bge_m3 \
  --emb-model BAAI/bge-m3 \
  --emb-strategy bge-m3 \
  --log-path "$LOG_PATH" \
  --checkpoint-folder faiss_index_tpch_bge_m3.checkpoint \
  --checkpoint-every-batches "$CHECKPOINT_EVERY_BATCHES"

echo "=== BGE M3 completed ==="
