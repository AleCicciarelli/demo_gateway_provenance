# rel-stack BGE-M3 index

The full rel-stack database is exported to `rel-stack_csv/`, including records
following the RelBench test timestamp. Its schema profile uses RelBench's declared
primary and foreign keys, with lower-case table names matching the document builder.

To export again (requires RelBench and network access unless cached):

```bash
.venv/bin/python -m pip install relbench
HF_HOME=/tmp/relbench-hf .venv/bin/python relbench_data.py --dataset rel-stack
```

Copy the CSV folder, schema profile, scripts, and dependency file to the shared
cluster working directory. CSVs and generated indexes are ignored by Git, so they
must be transferred separately or regenerated on the cluster.

Use a virtual environment with CUDA-enabled PyTorch and install the dependencies:

```bash
.venv/bin/python -m pip install -r requirements-index.txt
```

If GPU nodes cannot download models, cache BGE-M3 on the login node first:

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-m3", device="cpu")'
```

Submit from the repository root, adjusting resources to your cluster's policy:

```bash
oarsub -l /gpu=1,walltime=48:0:0 'bash ./run_oar_relstack_index.sh'
```

The job generates row documents using the same `semantic-join` format as the
rel-arxiv job, then normalized 1024-dimensional BGE-M3 dense vectors on CUDA.
FAISS runs on the CPU; `faiss-cpu` is intentional. The existing 120-character
per-value text limit is retained for comparability; full values remain in metadata.

Output folder: `faiss_index_relstack_rows_bge_m3_semantic_join/`

- `row_documents_relstack.jsonl`: row text, source IDs, and linked-row metadata.
- `index.faiss`: embeddings in the FAISS index.
- `index.pkl`: LangChain document store and ID mapping.

Logs: `logs/relstack_index_<job-id>.log`.
Checkpoints: `faiss_index_relstack_rows_bge_m3_semantic_join.checkpoint/`.
Resubmit the same command to resume; incomplete document generation starts again.
Do not change input CSVs, the schema, the builder, or the text strategy while a
build is running. After changes, choose a fresh folder:

```bash
oarsub -l /gpu=1,walltime=48:0:0 \
  'ROW_INDEX_FOLDER="$PWD/faiss_index_relstack_rows_bge_m3_semantic_join_v2" bash ./run_oar_relstack_index.sh'
```

Default encoding batch size is 16; lower `FAISS_ENCODE_BATCH_SIZE` if needed.
Host RAM is a separate requirement from GPU VRAM: all vectors and document
metadata accumulate in RAM. For approximately 5.40 million rows, dense vectors
alone take about 20.6 GiB, before metadata, source tables, and checkpoint overhead.
Use a high-memory node (128 GiB is an initial planning target, not a measured
requirement), and budget disk for both the final index and a full checkpoint copy.
The 48-hour walltime is a requested limit, not a measured runtime.

The job checks final vector count against the schema and verifies dimension 1024.
No OAR job is submitted by preparing these files. Serving this dataset in the
Gateway requires a separate configuration change.
