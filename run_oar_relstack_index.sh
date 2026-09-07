#!/usr/bin/env bash
# Submit from the repository root: oarsub -l /gpu=1,walltime=48:0:0 'bash ./run_oar_relstack_index.sh'
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

VENV_DIR="${VENV_DIR:-$PWD/.venv}"
PYTHON="$VENV_DIR/bin/python"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
STRATEGY="${ROW_TEXTUALIZATION_STRATEGY:-semantic-join}"
OUTPUT="${ROW_INDEX_FOLDER:-$PWD/faiss_index_relstack_rows_bge_m3_semantic_join}"
DOCUMENTS="$OUTPUT/row_documents_relstack.jsonl"
mkdir -p "$OUTPUT" logs
exec > >(tee -a "logs/relstack_index_${OAR_JOB_ID:-local}.log") 2>&1

"$PYTHON" - <<'PY'
import torch
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable. Run this job on an allocated GPU node with CUDA-enabled PyTorch.')
print('GPU:', torch.cuda.get_device_name(0))
PY

# Never reuse documents/checkpoints when the source data or text format changed.
FINGERPRINT=$( { sha256sum rel-stack_csv/*.csv rel-stack_csv/schema_profile_relstack.json build_row_index.py; printf '%s\n' "$STRATEGY"; } | sha256sum | cut -d ' ' -f 1)
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
        --csv_dir rel-stack_csv \
        --schema_profile rel-stack_csv/schema_profile_relstack.json \
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
    --checkpoint-every-batches "${FAISS_CHECKPOINT_EVERY_BATCHES:-1000}"

"$PYTHON" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path
import faiss
folder = Path(sys.argv[1])
profile = json.loads(Path('rel-stack_csv/schema_profile_relstack.json').read_text())
expected = sum(t['num_rows'] for t in profile['tables'].values())
index = faiss.read_index(str(folder / 'index.faiss'))
assert index.ntotal == expected, (index.ntotal, expected)
assert index.d == 1024, index.d
print(f'Validated FAISS index: {index.ntotal:,} vectors, dimension {index.d}')
PY
