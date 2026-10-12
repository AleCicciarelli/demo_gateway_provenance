#!/usr/bin/env bash
# Submit from the repository root: oarsub -l /gpu=1,walltime=4:0:0 'bash ./run_oar_mathe_index.sh'
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

VENV_DIR="${VENV_DIR:-$PWD/.venv}"
PYTHON="$VENV_DIR/bin/python"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
STRATEGY="${ROW_TEXTUALIZATION_STRATEGY:-semantic-join}"
OUTPUT="${ROW_INDEX_FOLDER:-$PWD/faiss_index_mathe_rows_bge_m3}"
DOCUMENTS="$OUTPUT/row_documents_mathe.jsonl"
if [[ ! -f mathe_csv/schema_profile_mathe.json ]]; then
    "$PYTHON" prepare_mathe.py
fi
mkdir -p "$OUTPUT" logs
exec > >(tee -a "logs/mathe_index_${OAR_JOB_ID:-local}.log") 2>&1

"$PYTHON" - <<'PY'
import torch
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable. Run this job on an allocated GPU node with CUDA-enabled PyTorch.')
print('GPU:', torch.cuda.get_device_name(0))
PY

"$PYTHON" - <<'PY'
import json
from pathlib import Path
profile = json.loads(Path('mathe_csv/schema_profile_mathe.json').read_text())
missing = [table for table in profile['tables'] if not Path(f'mathe_csv/{table}.csv').is_file()]
if missing:
    raise SystemExit(f'Missing CSV tables: {missing}. Run python3 prepare_mathe.py before submitting.')
PY

# Never reuse documents/checkpoints when the source data or text format changed.
FINGERPRINT=$( { sha256sum mathe_csv/*.csv mathe_csv/schema_profile_mathe.json build_row_index.py build_row_faiss_index.py; printf '%s\n' "$STRATEGY"; } | sha256sum | cut -d ' ' -f 1)
if [[ -e "$OUTPUT/source.sha256" ]]; then
    [[ "$(cat "$OUTPUT/source.sha256")" == "$FINGERPRINT" ]] || {
        echo 'Source data or document builder changed. Set ROW_INDEX_FOLDER to a fresh directory.'
        exit 1
    }
else
    [[ ! -e "$DOCUMENTS" && ! -e "$OUTPUT.checkpoint/manifest.json" ]] || {
        echo 'Existing artifacts have no source fingerprint. Use a fresh ROW_INDEX_FOLDER.'
        exit 1
    }
    printf '%s\n' "$FINGERPRINT" > "$OUTPUT/source.sha256"
fi

if [[ ! -f "$DOCUMENTS" ]]; then
    "$PYTHON" build_row_index.py \
        --csv_dir mathe_csv \
        --schema_profile mathe_csv/schema_profile_mathe.json \
        --documents_out "$DOCUMENTS" \
        --textualization-strategy "$STRATEGY" --documents-only
fi

"$PYTHON" build_row_faiss_index.py \
    --documents "$DOCUMENTS" \
    --index-folder "$OUTPUT" \
    --checkpoint-folder "$OUTPUT.checkpoint" \
    --embedding-model BAAI/bge-m3 --device cuda \
    --batch-size 256 \
    --encode-batch-size "${FAISS_ENCODE_BATCH_SIZE:-16}" \
    --checkpoint-every-batches "${FAISS_CHECKPOINT_EVERY_BATCHES:-25}"

"$PYTHON" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path
import faiss
folder = Path(sys.argv[1])
profile = json.loads(Path('mathe_csv/schema_profile_mathe.json').read_text())
expected = sum(t['row_count'] for t in profile['tables'].values())
index = faiss.read_index(str(folder / 'index.faiss'))
assert index.ntotal == expected, (index.ntotal, expected)
assert index.d == 1024, index.d
print(f'Validated FAISS index: {index.ntotal:,} vectors, dimension {index.d}')
PY
