#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


LogEvent = Callable[[Dict[str, Any]], None]


class FaissIndexManager:
    def __init__(
        self,
        index_folder: str,
        embeddings_factory: Callable[[], Embeddings],
        csv_cache: Dict[str, List[Dict[str, Any]]],
        load_csvs: Callable[[], None],
        index_set: Set[str],
        batch_size: int,
        emb_model: str,
        emb_strategy: str,
        log_event: Optional[LogEvent] = None,
        checkpoint_folder: Optional[str] = None,
        checkpoint_every_batches: int = 0,
        resume_checkpoint: bool = True,
    ):
        self.index_folder = index_folder
        self.embeddings_factory = embeddings_factory
        self.csv_cache = csv_cache
        self.load_csvs = load_csvs
        self.index_set = index_set
        self.batch_size = batch_size
        self.emb_model = emb_model
        self.emb_strategy = emb_strategy
        self.log_event = log_event
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_every_batches = checkpoint_every_batches
        self.resume_checkpoint = resume_checkpoint
        self.vector_store: Optional[FAISS] = None

    @staticmethod
    def row_to_text(row: Dict[str, Any], table: Optional[str] = None) -> str:
        parts = []

        if table:
            parts.append(f"table: {table}")

        for key, value in row.items():
            if key == "__rid__" or key.endswith("_rownum"):
                continue
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            parts.append(f"{key}: {text}")

        return " | ".join(parts)

    def get_or_build(self) -> FAISS:
        if self.vector_store is not None:
            return self.vector_store

        embedding_model = self.embeddings_factory()
        index_path = Path(self.index_folder)

        if index_path.exists() and index_path.is_dir() and any(index_path.iterdir()):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_folder,
                    embedding_model,
                    allow_dangerous_deserialization=True,
                )
                self._log({
                    "type": "faiss_loaded",
                    "path": self.index_folder,
                    "emb_model": self.emb_model,
                    "emb_strategy": self.emb_strategy,
                })
                return self.vector_store
            except Exception as exc:
                self._log({
                    "type": "faiss_load_failed",
                    "path": self.index_folder,
                    "emb_model": self.emb_model,
                    "emb_strategy": self.emb_strategy,
                    "error": str(exc),
                })

        self.load_csvs()
        self.vector_store = self.build_index(embedding_model)
        return self.vector_store

    def build_index(self, embedding_model: Embeddings) -> FAISS:
        started_at = time.monotonic()
        index_path = Path(self.index_folder)
        vector_store: Optional[FAISS] = None
        batch_docs: List[Document] = []
        total_docs = 0
        batch_count = 0
        target_docs = self._count_target_rows()
        resume_docs = 0

        if self.resume_checkpoint:
            checkpoint = self._load_checkpoint(embedding_model, target_docs)
            if checkpoint is not None:
                vector_store, resume_docs, batch_count = checkpoint
                total_docs = resume_docs

        self._progress(
            "Starting FAISS build "
            f"path={self.index_folder} emb_model={self.emb_model} "
            f"emb_strategy={self.emb_strategy} batch_size={self.batch_size} "
            f"target_rows={target_docs} resume_docs={resume_docs}"
        )

        for table, rows in self.csv_cache.items():
            if self.index_set and table not in self.index_set:
                self._progress(f"Skipping table={table} rows={len(rows)}")
                continue

            table_started_at = time.monotonic()
            table_docs = 0
            self._progress(f"Indexing table={table} rows={len(rows)}")

            for row in rows:
                rid = row.get("__rid__")
                if not rid:
                    continue

                text = self.row_to_text(row, table)
                if not text:
                    continue

                if resume_docs > 0:
                    resume_docs -= 1
                    continue

                batch_docs.append(
                    Document(
                        page_content=text,
                        metadata={"table": table, "rid": rid},
                    )
                )
                table_docs += 1

                if len(batch_docs) >= self.batch_size:
                    batch_count += 1
                    self._progress_batch_start(batch_count, total_docs, len(batch_docs), target_docs)
                    vector_store = self._add_batch(vector_store, batch_docs, embedding_model)
                    total_docs += len(batch_docs)
                    batch_docs = []
                    self._progress_batch_done(batch_count, total_docs, target_docs, started_at)
                    self._checkpoint_if_due(vector_store, total_docs, batch_count, target_docs)

            self._progress(
                f"Finished table={table} queued_rows={table_docs} "
                f"elapsed={self._format_elapsed(time.monotonic() - table_started_at)}"
            )

        if batch_docs:
            batch_count += 1
            self._progress_batch_start(batch_count, total_docs, len(batch_docs), target_docs)
            vector_store = self._add_batch(vector_store, batch_docs, embedding_model)
            total_docs += len(batch_docs)
            self._progress_batch_done(batch_count, total_docs, target_docs, started_at)
            self._checkpoint_if_due(vector_store, total_docs, batch_count, target_docs, force=True)

        if vector_store is None:
            raise RuntimeError("No documents indexed. Check CSV_DIR / INDEX_TABLES / file contents.")

        self._progress(f"Saving FAISS index to {self.index_folder}")
        save_started_at = time.monotonic()
        index_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(self.index_folder)
        self._progress(f"Saved FAISS index elapsed={self._format_elapsed(time.monotonic() - save_started_at)}")
        self._save_checkpoint(vector_store, total_docs, batch_count, target_docs, status="complete")

        self._log({
            "type": "faiss_built",
            "path": self.index_folder,
            "n_docs": total_docs,
            "emb_model": self.emb_model,
            "emb_strategy": self.emb_strategy,
        })

        self._progress(
            f"Completed FAISS build docs={total_docs} batches={batch_count} "
            f"elapsed={self._format_elapsed(time.monotonic() - started_at)}"
        )
        return vector_store

    @staticmethod
    def _add_batch(
        vector_store: Optional[FAISS],
        batch_docs: List[Document],
        embedding_model: Embeddings,
    ) -> FAISS:
        if vector_store is None:
            return FAISS.from_documents(batch_docs, embedding=embedding_model)

        vector_store.add_documents(batch_docs)
        return vector_store

    def _log(self, event: Dict[str, Any]) -> None:
        if self.log_event is not None:
            self.log_event(event)

    def _load_checkpoint(
        self,
        embedding_model: Embeddings,
        target_docs: int,
    ) -> Optional[tuple[FAISS, int, int]]:
        if not self.checkpoint_folder:
            return None

        checkpoint_path = Path(self.checkpoint_folder)
        manifest_path = self._checkpoint_manifest_path()
        if not manifest_path.exists():
            return None

        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception as exc:
            self._progress(f"Ignoring unreadable checkpoint manifest={manifest_path} error={exc}")
            return None

        if manifest.get("emb_model") != self.emb_model or manifest.get("emb_strategy") != self.emb_strategy:
            self._progress(
                "Ignoring checkpoint with different embedding config "
                f"manifest_model={manifest.get('emb_model')} manifest_strategy={manifest.get('emb_strategy')}"
            )
            return None

        if manifest.get("target_docs") != target_docs:
            self._progress(
                "Ignoring checkpoint because target row count changed "
                f"manifest_target={manifest.get('target_docs')} current_target={target_docs}"
            )
            return None

        try:
            vector_store = FAISS.load_local(
                str(checkpoint_path),
                embedding_model,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            self._progress(f"Ignoring unreadable checkpoint path={checkpoint_path} error={exc}")
            return None

        manifest_docs = int(manifest.get("n_docs", 0))
        actual_docs = len(vector_store.index_to_docstore_id)
        if actual_docs != manifest_docs:
            self._progress(
                "Checkpoint manifest/docstore count mismatch "
                f"manifest_docs={manifest_docs} actual_docs={actual_docs}; using actual_docs"
            )

        n_docs = min(actual_docs, target_docs)
        batch_count = int(manifest.get("batch_count", n_docs // max(self.batch_size, 1)))
        self._progress(
            f"Loaded checkpoint path={checkpoint_path} docs={n_docs} "
            f"batches={batch_count} status={manifest.get('status', 'unknown')}"
        )
        return vector_store, n_docs, batch_count

    def _checkpoint_if_due(
        self,
        vector_store: FAISS,
        total_docs: int,
        batch_count: int,
        target_docs: int,
        force: bool = False,
    ) -> None:
        if self.checkpoint_every_batches <= 0:
            return
        if not force and batch_count % self.checkpoint_every_batches != 0:
            return
        self._save_checkpoint(vector_store, total_docs, batch_count, target_docs, status="partial")

    def _save_checkpoint(
        self,
        vector_store: FAISS,
        total_docs: int,
        batch_count: int,
        target_docs: int,
        status: str,
    ) -> None:
        if not self.checkpoint_folder:
            return

        checkpoint_path = Path(self.checkpoint_folder)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint_started_at = time.monotonic()
        self._progress(
            f"Saving FAISS checkpoint to {checkpoint_path} "
            f"docs={self._format_progress(total_docs, target_docs)} status={status}"
        )
        vector_store.save_local(str(checkpoint_path))
        manifest = {
            "status": status,
            "n_docs": total_docs,
            "batch_count": batch_count,
            "target_docs": target_docs,
            "batch_size": self.batch_size,
            "emb_model": self.emb_model,
            "emb_strategy": self.emb_strategy,
            "index_set": sorted(self.index_set),
            "index_folder": self.index_folder,
            "updated_at": int(time.time()),
        }
        with self._checkpoint_manifest_path().open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        self._progress(
            f"Saved FAISS checkpoint elapsed={self._format_elapsed(time.monotonic() - checkpoint_started_at)}"
        )

    def _checkpoint_manifest_path(self) -> Path:
        if not self.checkpoint_folder:
            raise RuntimeError("checkpoint_folder is not configured")
        return Path(self.checkpoint_folder) / "manifest.json"

    def _count_target_rows(self) -> int:
        total = 0
        for table, rows in self.csv_cache.items():
            if self.index_set and table not in self.index_set:
                continue
            for row in rows:
                rid = row.get("__rid__")
                if not rid:
                    continue
                if not self.row_to_text(row, table):
                    continue
                total += 1
        return total

    def _progress_batch_start(
        self,
        batch_count: int,
        total_docs: int,
        batch_size: int,
        target_docs: int,
    ) -> None:
        self._progress(
            f"Embedding batch={batch_count} batch_rows={batch_size} "
            f"indexed_before={self._format_progress(total_docs, target_docs)}"
        )

    def _progress_batch_done(
        self,
        batch_count: int,
        total_docs: int,
        target_docs: int,
        started_at: float,
    ) -> None:
        self._progress(
            f"Indexed batch={batch_count} total={self._format_progress(total_docs, target_docs)} "
            f"elapsed={self._format_elapsed(time.monotonic() - started_at)}"
        )

    @staticmethod
    def _format_progress(done: int, total: int) -> str:
        if total <= 0:
            return str(done)
        pct = done / total * 100
        return f"{done}/{total} ({pct:.1f}%)"

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"

        minutes, remaining_seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h{minutes:02d}m{remaining_seconds:02d}s"
        return f"{minutes}m{remaining_seconds:02d}s"

    @staticmethod
    def _progress(message: str) -> None:
        print(f"[FAISS] {message}", flush=True)
