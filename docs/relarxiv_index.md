# rel-arxiv BGE-M3 index

The OAR job exports 2,733,846 row documents and builds a LangChain FAISS index
with normalized, 1024-dimensional `BAAI/bge-m3` dense embeddings. It defaults
to the `semantic-join` text format used by the rel-f1 semantic-join index.
Linked-row metadata uses the official RelBench primary/foreign keys recorded
in `rel-arxiv_csv/schema_profile_relarxiv.json`. Tables without primary keys
use their CSV row positions for document IDs. Text values retain the existing
builder's 120-character truncation; full values remain in document metadata.

On the cluster, place this repository and `rel-arxiv_csv/` on shared storage.
Prepare a virtual environment with CUDA-enabled PyTorch appropriate to the
cluster, then install the indexing dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-index.txt
```

If compute nodes lack internet access, download the model on a login node first:

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-m3", device="cpu")'
```

Submit from the repository root (adjust resource syntax and walltime to cluster policy):

```bash
oarsub -l /gpu=1,walltime=48:0:0 'bash ./run_oar_relarxiv_index.sh'
```

Allow at least 64 GB host RAM and 40 GB free disk as an initial resource budget;
actual usage depends on document metadata and the environment. The dense vectors
alone occupy about 10.4 GiB, and checkpoints duplicate the index and document store.
48 hours is a requested limit, not a runtime estimate. The default encoding batch
size is 16; reduce `FAISS_ENCODE_BATCH_SIZE` if GPU memory is insufficient.

Outputs are under `faiss_index_relarxiv_rows_bge_m3_semantic_join/`:

- `row_documents_relarxiv.jsonl`: documents and row metadata.
- `index.faiss`: dense vectors.
- `index.pkl`: LangChain document store and ID mapping.

Logs are in `logs/relarxiv_index_<job-id>.log`. Resubmit the same command to
reuse completed documents and resume the latest checkpoint. Checkpoints are
saved every 256,000 rows by default; an interrupted batch interval is recomputed.
Document generation restarts if interrupted before its atomic final rename.
Source fingerprints prevent reuse after CSV, schema, strategy, or builder changes;
set `ROW_INDEX_FOLDER` to a new directory for changed inputs.

The job verifies the final vector count and dimension. Gateway configuration and
Docker mounts must be updated separately before serving this index.
