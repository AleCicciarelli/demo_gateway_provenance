#!/usr/bin/env bash
# oarsub -l /gpu=1,walltime=12:0:0 'bash ./run_oar_wikidata_movies_index.sh'
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

VENV_DIR="${VENV_DIR:-$PWD/.venv}"
PYTHON="${INDEX_PYTHON:-$VENV_DIR/bin/python}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
OUTPUT="${ROW_INDEX_FOLDER:-$PWD/faiss_index_wikidata_movies_rows_bge_m3_semantic_join}"
DOCUMENTS="$OUTPUT/row_documents_wikidata_movies.jsonl"
DEVICE="${FAISS_DEVICE:-cuda}"
mkdir -p "$OUTPUT"
LOG_PATH="${ROW_INDEX_LOG:-$OUTPUT/build_${OAR_JOB_ID:-local}.log}"
touch "$LOG_PATH"
exec > >(tee -a "$LOG_PATH") 2>&1

FINGERPRINT=$(sha256sum wikidata_movies/csv/*.csv wikidata_movies/metadata/schema.json prepare_wikidata_movies_documents.py build_row_faiss_index.py | sha256sum | cut -d ' ' -f 1)
if [[ -e "$OUTPUT/source.sha256" ]]; then
    [[ "$(cat "$OUTPUT/source.sha256")" == "$FINGERPRINT" ]] || {
        echo 'Source data or builder changed. Set ROW_INDEX_FOLDER to a fresh directory.'
        exit 1
    }
else
    [[ ! -e "$DOCUMENTS" && ! -e "$OUTPUT/index.faiss" && ! -e "$OUTPUT.checkpoint/manifest.json" ]] || {
        echo 'Existing artifacts have no source fingerprint. Use a fresh ROW_INDEX_FOLDER.'
        exit 1
    }
    printf '%s\n' "$FINGERPRINT" > "$OUTPUT/source.sha256"
fi

if [[ ! -f "$DOCUMENTS" ]]; then
    "$PYTHON" prepare_wikidata_movies_documents.py --documents-out "$DOCUMENTS"
fi
if [[ "${DOCUMENTS_ONLY:-0}" == 1 ]]; then
    exit 0
fi

"$PYTHON" - "$DEVICE" <<'PY'
import sys
import torch
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
if sys.argv[1].startswith('cuda') and not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Use an allocated GPU node, or set FAISS_DEVICE=cpu.')
print('Embedding device:', sys.argv[1])
PY

"$PYTHON" build_row_faiss_index.py \
    --documents "$DOCUMENTS" \
    --index-folder "$OUTPUT" \
    --checkpoint-folder "$OUTPUT.checkpoint" \
    --embedding-model BAAI/bge-m3 --device "$DEVICE" \
    --batch-size 256 \
    --encode-batch-size "${FAISS_ENCODE_BATCH_SIZE:-16}" \
    --checkpoint-every-batches "${FAISS_CHECKPOINT_EVERY_BATCHES:-100}"

"$PYTHON" - "$OUTPUT" <<'PY'
import csv
import json
import sys
from pathlib import Path
import faiss
folder = Path(sys.argv[1])
expected = 0
for path in Path('wikidata_movies/csv').glob('*.csv'):
    with path.open(newline='', encoding='utf-8') as handle:
        expected += sum(1 for _ in csv.DictReader(handle))
with (folder / 'row_documents_wikidata_movies.jsonl').open() as handle:
    actual = sum(1 for line in handle if line.strip())
index = faiss.read_index(str(folder / 'index.faiss'))
assert index.ntotal == actual == expected, (index.ntotal, actual, expected)
assert index.d == 1024, index.d
print(f'Validated FAISS index: {index.ntotal:,} vectors, dimension {index.d}')
PY
