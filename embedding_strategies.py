#!/usr/bin/env python3
from __future__ import annotations

import os
from contextlib import redirect_stderr
from typing import Literal, Optional

import torch
from FlagEmbedding import BGEM3FlagModel, FlagModel
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

EmbeddingStrategy = Literal["sentence-transformer", "bge-v1.5", "bge-m3"]


class EmbeddingStrategies(Embeddings):
    """LangChain embeddings wrapper with selectable embedding backends."""

    BGE_V15_MODELS = {
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
    }

    def __init__(
        self,
        model_name: str,
        strategy: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 500,
    ):
        self.model_name = model_name
        self.strategy = self._resolve_strategy(model_name, strategy)
        self.device = self._resolve_device(device)
        self.batch_size = batch_size

        if self.strategy == "sentence-transformer":
            self.model = SentenceTransformer(model_name, device=self.device)
            print(f"[EMB] SentenceTransformer using device: {self.device}", flush=True)
            return

        if self.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            use_fp16 = False
        else:
            use_fp16 = True

        if self.strategy == "bge-m3":
            self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        else:
            query_instruction = ""
            if model_name in self.BGE_V15_MODELS:
                query_instruction = "Represent this sentence for searching relevant passages: "

            self.model = FlagModel(
                model_name,
                query_instruction_for_retrieval=query_instruction,
                use_fp16=use_fp16,
            )

        print(
            f"[EMB] FlagEmbedding model={model_name} "
            f"strategy={self.strategy} device={self.device} use_fp16={use_fp16}",
            flush=True,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @classmethod
    def _resolve_strategy(cls, model_name: str, strategy: Optional[str]) -> EmbeddingStrategy:
        normalized = (strategy or "auto").strip().lower()
        aliases = {
            "sentence": "sentence-transformer",
            "sentence_transformer": "sentence-transformer",
            "sentence-transformers": "sentence-transformer",
            "st": "sentence-transformer",
            "bge": "bge-v1.5",
            "bge-v15": "bge-v1.5",
            "bge-v1_5": "bge-v1.5",
            "bge-m3": "bge-m3",
            "m3": "bge-m3",
        }
        normalized = aliases.get(normalized, normalized)

        if normalized == "auto":
            if model_name == "BAAI/bge-m3":
                return "bge-m3"
            if model_name in cls.BGE_V15_MODELS or model_name.startswith("BAAI/bge-"):
                return "bge-v1.5"
            return "sentence-transformer"

        if normalized not in {"sentence-transformer", "bge-v1.5", "bge-m3"}:
            raise ValueError(
                "Unknown embedding strategy "
                f"'{strategy}'. Use sentence-transformer, bge-v1.5, bge-m3, or auto."
            )

        return normalized  # type: ignore[return-value]

    def embed_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        if self.strategy == "sentence-transformer":
            with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
                return self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).tolist()

        if self.strategy == "bge-m3":
            with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
                output = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    max_length=8192,
                )
            return output["dense_vecs"].tolist()

        with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
            embeddings = self.model.encode_corpus(
                texts,
                batch_size=self.batch_size,
            )
        return embeddings.tolist()

    def embed_query(self, text):
        if self.strategy == "sentence-transformer":
            with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
                return self.model.encode(
                    [text],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )[0].tolist()

        if self.strategy == "bge-m3":
            with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
                output = self.model.encode(
                    [text],
                    batch_size=1,
                    max_length=8192,
                )
            return output["dense_vecs"][0].tolist()

        with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
            embedding = self.model.encode_queries(
                [text],
                batch_size=1,
            )
        return embedding[0].tolist()
