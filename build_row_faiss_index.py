#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:  # Compatibility with the existing gateway image.
    from langchain_community.embeddings import HuggingFaceEmbeddings


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(metadata)

    # gateway.py expects metadata["table"] and metadata["rid"] so it can look up
    # the original CSV row in _CSV_RID_INDEX.
    if not metadata.get("rid"):
        metadata["rid"] = metadata.get("rownum_value") or metadata.get("row_id")

    return metadata


def iter_documents(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            yield line_number, Document(
                page_content=str(item.get("page_content") or ""),
                metadata=normalize_metadata(item.get("metadata") or {}),
            )


def manifest_path(checkpoint_folder: Path) -> Path:
    return checkpoint_folder / "manifest.json"


def load_checkpoint(
    checkpoint_folder: Path,
    embeddings: HuggingFaceEmbeddings,
    documents_path: Path,
    embedding_model: str,
):
    manifest_file = manifest_path(checkpoint_folder)
    if not manifest_file.exists():
        return None

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("documents") != str(documents_path):
        print("Ignoring checkpoint with different documents path", flush=True)
        return None
    if manifest.get("embedding_model") != embedding_model:
        print("Ignoring checkpoint with different embedding model", flush=True)
        return None

    vectorstore = FAISS.load_local(
        str(checkpoint_folder),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    n_docs = int(manifest.get("n_docs", 0))
    batches = int(manifest.get("batches", 0))
    print(f"Resuming checkpoint docs={n_docs} batches={batches}", flush=True)
    return vectorstore, n_docs, batches


def save_checkpoint(
    vectorstore: FAISS,
    checkpoint_folder: Path,
    documents_path: Path,
    embedding_model: str,
    n_docs: int,
    batches: int,
    status: str,
) -> None:
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(checkpoint_folder))
    manifest_path(checkpoint_folder).write_text(
        json.dumps(
            {
                "documents": str(documents_path),
                "embedding_model": embedding_model,
                "n_docs": n_docs,
                "batches": batches,
                "status": status,
                "updated_at": int(time.time()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS from completed row documents JSONL.")
    parser.add_argument("--documents", default="faiss_index_tpch_rows_bge_m3/documents.jsonl")
    parser.add_argument("--index-folder", default="faiss_index_tpch_rows_bge_m3")
    parser.add_argument("--checkpoint-folder", default="faiss_index_tpch_rows_bge_m3.checkpoint")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every-batches", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    documents_path = Path(args.documents)
    index_folder = Path(args.index_folder)
    checkpoint_folder = Path(args.checkpoint_folder)

    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": args.device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": args.encode_batch_size,
        },
    )

    vectorstore = None
    indexed_docs = 0
    batch_count = 0

    if not args.no_resume:
        checkpoint = load_checkpoint(
            checkpoint_folder,
            embeddings,
            documents_path,
            args.embedding_model,
        )
        if checkpoint is not None:
            vectorstore, indexed_docs, batch_count = checkpoint

    batch: list[Document] = []
    started_at = time.monotonic()

    for line_number, doc in iter_documents(documents_path):
        if line_number <= indexed_docs:
            continue

        batch.append(doc)
        if len(batch) < args.batch_size:
            continue

        batch_count += 1
        print(
            f"Embedding batch={batch_count} rows={len(batch)} "
            f"indexed_before={indexed_docs}",
            flush=True,
        )
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        indexed_docs += len(batch)
        batch = []

        elapsed = time.monotonic() - started_at
        print(f"Indexed docs={indexed_docs} elapsed={elapsed:.1f}s", flush=True)

        if args.checkpoint_every_batches and batch_count % args.checkpoint_every_batches == 0:
            print(f"Saving checkpoint docs={indexed_docs}", flush=True)
            save_checkpoint(
                vectorstore,
                checkpoint_folder,
                documents_path,
                args.embedding_model,
                indexed_docs,
                batch_count,
                "partial",
            )

    if batch:
        batch_count += 1
        print(
            f"Embedding batch={batch_count} rows={len(batch)} "
            f"indexed_before={indexed_docs}",
            flush=True,
        )
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        indexed_docs += len(batch)

    if vectorstore is None:
        raise RuntimeError("No documents indexed.")

    index_folder.mkdir(parents=True, exist_ok=True)
    print(f"Saving final FAISS index to {index_folder}", flush=True)
    vectorstore.save_local(str(index_folder))
    save_checkpoint(
        vectorstore,
        checkpoint_folder,
        documents_path,
        args.embedding_model,
        indexed_docs,
        batch_count,
        "complete",
    )
    print(f"Completed FAISS row index docs={indexed_docs}", flush=True)


if __name__ == "__main__":
    main()
