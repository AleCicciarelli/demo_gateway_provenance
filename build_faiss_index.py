#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List


def load_csvs(csv_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    csv_cache: Dict[str, List[Dict[str, Any]]] = {}
    base = Path(csv_dir)
    if not base.exists():
        raise RuntimeError(f"CSV_DIR not found: {csv_dir}")

    for path in sorted(base.glob("*.csv")):
        table = path.stem
        id_col = f"{table}_rownum"
        rows: List[Dict[str, Any]] = []

        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="|")
            if reader.fieldnames is None or id_col not in reader.fieldnames:
                raise RuntimeError(f"Missing required id column '{id_col}' in {path}")

            for row in reader:
                clean_row = dict(row)
                if None in clean_row:
                    clean_row.pop(None, None)

                rid = str(clean_row.get(id_col, "")).strip()
                if not rid:
                    continue

                clean_row["__rid__"] = rid
                rows.append(clean_row)

        csv_cache[table] = rows

    return csv_cache


def append_log(log_path: str, event: Dict[str, Any]) -> None:
    event["ts"] = int(time.time())
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def check_torchvision_compatibility() -> None:
    try:
        import importlib.metadata as metadata

        torch_version = _version_tuple(metadata.version("torch"))
        torchvision_version = _version_tuple(metadata.version("torchvision"))
    except metadata.PackageNotFoundError:
        return

    if torch_version >= (2, 6) and torchvision_version < (0, 21):
        raise RuntimeError(
            "Incompatible torch/torchvision versions detected: "
            f"torch={metadata.version('torch')}, torchvision={metadata.version('torchvision')}. "
            "Upgrade with: pip install -U 'torch>=2.6' 'torchvision>=0.21'"
        )


def _version_tuple(version: str) -> tuple[int, int]:
    main = version.split("+", 1)[0]
    parts = main.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index from TPCH CSV files.")
    parser.add_argument("--csv-dir", default=os.getenv("CSV_DIR", "/app/tpch_no_provsql"))
    parser.add_argument("--index-folder", default=os.getenv("FAISS_INDEX_FOLDER", "/app/faiss_index_tpch"))
    parser.add_argument("--emb-model", default=os.getenv("EMB_MODEL", "sentence-transformers/all-mpnet-base-v2"))
    parser.add_argument("--emb-strategy", default=os.getenv("EMB_STRATEGY", "auto"))
    parser.add_argument("--emb-device", default=os.getenv("EMB_DEVICE", "auto"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("INDEX_BATCH_SIZE", "500")))
    parser.add_argument("--index-tables", default=os.getenv("INDEX_TABLES", ""))
    parser.add_argument("--log-path", default=os.getenv("GATEWAY_LOG_PATH", "/app/logs/provsql_gateway_logs.jsonl"))
    parser.add_argument("--checkpoint-folder", default=os.getenv("FAISS_CHECKPOINT_FOLDER", ""))
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=int(os.getenv("FAISS_CHECKPOINT_EVERY_BATCHES", "0")),
        help="Save a resumable checkpoint every N batches. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-resume-checkpoint",
        action="store_false",
        dest="resume_checkpoint",
        help="Ignore any existing checkpoint and rebuild from scratch.",
    )
    parser.set_defaults(resume_checkpoint=True)
    args = parser.parse_args()

    check_torchvision_compatibility()

    from embedding_strategies import EmbeddingStrategies
    from faiss_index_manager import FaissIndexManager

    csv_cache = load_csvs(args.csv_dir)
    index_set = {table.strip() for table in args.index_tables.split(",") if table.strip()}
    embeddings = EmbeddingStrategies(
        model_name=args.emb_model,
        strategy=args.emb_strategy,
        device=args.emb_device,
        batch_size=args.batch_size,
    )

    manager = FaissIndexManager(
        index_folder=args.index_folder,
        embeddings_factory=lambda: embeddings,
        csv_cache=csv_cache,
        load_csvs=lambda: None,
        index_set=index_set,
        batch_size=args.batch_size,
        emb_model=args.emb_model,
        emb_strategy=args.emb_strategy,
        log_event=lambda event: append_log(args.log_path, event),
        checkpoint_folder=args.checkpoint_folder or None,
        checkpoint_every_batches=args.checkpoint_every_batches,
        resume_checkpoint=args.resume_checkpoint,
    )
    manager.build_index(embeddings)
    print(f"Wrote FAISS index: {Path(args.index_folder).resolve()}")


if __name__ == "__main__":
    main()
