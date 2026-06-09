#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DummyEmbeddings(Embeddings):
    """FAISS.load_local needs an embeddings object, but visualization does not."""

    def embed_documents(self, texts):
        raise NotImplementedError("This script only reads existing vectors.")

    def embed_query(self, text):
        raise NotImplementedError("This script only reads existing vectors.")


def default_index_path() -> str:
    if Path("faiss_index_tpch").exists():
        return "faiss_index_tpch"
    return "faiss_index"


def label_for_doc(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return metadata.get("table") or metadata.get("source") or "unknown"


def sample_ids_by_label(labels: list[str], sample_size: int, seed: int) -> list[int]:
    if sample_size <= 0 or sample_size >= len(labels):
        return list(range(len(labels)))

    rng = np.random.default_rng(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)

    selected: list[int] = []
    quota = max(1, sample_size // max(1, len(by_label)))
    leftovers: list[int] = []

    for ids in by_label.values():
        if len(ids) <= quota:
            selected.extend(ids)
        else:
            selected.extend(rng.choice(ids, size=quota, replace=False).tolist())
            leftovers.extend(set(ids) - set(selected))

    remaining = sample_size - len(selected)
    if remaining > 0 and leftovers:
        selected.extend(rng.choice(leftovers, size=min(remaining, len(leftovers)), replace=False).tolist())

    selected = sorted(set(selected))
    if len(selected) > sample_size:
        selected = sorted(rng.choice(selected, size=sample_size, replace=False).tolist())
    return selected


def reconstruct_vectors(index, ids: list[int]) -> np.ndarray:
    vectors = np.empty((len(ids), index.d), dtype=np.float32)
    for out_idx, faiss_id in enumerate(ids):
        vectors[out_idx] = index.reconstruct(int(faiss_id))
    return vectors


def write_points_csv(path: Path, points: np.ndarray, ids: list[int], labels: list[str], doc_ids: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_row", "faiss_id", "docstore_id", "label", "x", "y"])
        for sample_row, (point, faiss_id, label, doc_id) in enumerate(zip(points, ids, labels, doc_ids)):
            writer.writerow([sample_row, faiss_id, doc_id, label, point[0], point[1]])


def plot_points(path: Path, points: np.ndarray, labels: list[str], title: str) -> None:
    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    color_ids = [label_to_id[label] for label in labels]

    cmap = plt.get_cmap("tab20", max(1, len(unique_labels)))
    plt.figure(figsize=(12, 8))
    plt.scatter(points[:, 0], points[:, 1], c=color_ids, cmap=cmap, s=8, alpha=0.65, linewidths=0)
    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=cmap(label_to_id[label]),
            label=f"{label} ({labels.count(label)})",
            markersize=6,
        )
        for label in unique_labels
    ]
    plt.legend(handles=handles, title="Label", loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a FAISS index with a sampled t-SNE plot.")
    parser.add_argument("--index", default=default_index_path(), help="FAISS index folder.")
    parser.add_argument("--sample", type=int, default=5000, help="Number of vectors to sample. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and t-SNE.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--output", default="faiss_clusters_tsne_20000.png", help="Output PNG path.")
    parser.add_argument("--csv", default="faiss_clusters_tsne_20000.csv", help="Output CSV path.")
    args = parser.parse_args()

    vectorstore = FAISS.load_local(
        args.index,
        DummyEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    faiss_index = vectorstore.index
    total = faiss_index.ntotal

    doc_ids = [vectorstore.index_to_docstore_id[i] for i in range(total)]
    docs = [vectorstore.docstore.search(doc_id) for doc_id in doc_ids]
    all_labels = [label_for_doc(doc) for doc in docs]

    ids = sample_ids_by_label(all_labels, args.sample, args.seed)
    labels = [all_labels[i] for i in ids]
    sampled_doc_ids = [doc_ids[i] for i in ids]

    print(f"Loaded {total} vectors from {args.index} (dimension={faiss_index.d}).")
    print(f"Visualizing {len(ids)} sampled vectors.")
    print("Sampled label counts:")
    for label in sorted(set(labels)):
        print(f"  {label}: {labels.count(label)}")

    vectors = reconstruct_vectors(faiss_index, ids)

    reduced = vectors
    if vectors.shape[1] > 50 and len(ids) > 50:
        reduced = PCA(n_components=50, random_state=args.seed).fit_transform(vectors)

    perplexity = min(args.perplexity, max(1, (len(ids) - 1) / 3))
    points = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(reduced)

    output_path = Path(args.output)
    csv_path = Path(args.csv)
    write_points_csv(csv_path, points, ids, labels, sampled_doc_ids)
    plot_points(
        output_path,
        points,
        labels,
        f"FAISS embeddings visualized with t-SNE ({len(ids)} of {total} vectors)",
    )

    print(f"Wrote plot: {output_path.resolve()}")
    print(f"Wrote coordinates: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
