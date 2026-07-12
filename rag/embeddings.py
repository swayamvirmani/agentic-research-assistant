"""
Embedding model abstraction with batching and caching.
Supports OpenAI and local SentenceTransformers.
"""

from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from typing import Literal

import numpy as np
import structlog

from utils.config import settings

logger = structlog.get_logger()


class EmbeddingCache:
    """Simple in-memory LRU cache for embeddings."""

    def __init__(self, max_size: int = 10_000):
        self._cache: dict[str, list[float]] = {}
        self._max = max_size

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        return self._cache.get(self._key(text))

    def set(self, text: str, embedding: list[float]) -> None:
        if len(self._cache) >= self._max:
            # Evict oldest
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[self._key(text)] = embedding

    @property
    def size(self) -> int:
        return len(self._cache)


class OpenAIEmbedder:
    """
    OpenAI text-embedding-3-* embedder with:
    - Automatic batching (max 2048 texts per request)
    - In-memory caching
    - Retry with exponential backoff
    """

    MAX_BATCH = 512

    def __init__(
        self,
        model: str = settings.openai_embedding_model,
        dimensions: int | None = None,
    ):
        from openai import OpenAI

        self.model = model
        self.dimensions = dimensions
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._cache = EmbeddingCache()
        self.dim = self._infer_dim()

    def _infer_dim(self) -> int:
        dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return self.dimensions or dims.get(self.model, 1536)

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed a list of texts, using cache where possible."""
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            embeddings = self._embed_batch(uncached_texts)
            for idx, emb, text in zip(uncached_indices, embeddings, uncached_texts):
                results[idx] = emb
                self._cache.set(text, emb)

        return np.array(results, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.MAX_BATCH):
            batch = texts[i : i + self.MAX_BATCH]
            kwargs: dict = {"model": self.model, "input": batch}
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions

            for attempt in range(3):
                try:
                    response = self._client.embeddings.create(**kwargs)
                    batch_embs = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embs)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    wait = 2 ** attempt
                    logger.warning("embed_retry", attempt=attempt, error=str(e))
                    time.sleep(wait)

        return all_embeddings


class SentenceTransformerEmbedder:
    """
    Local SentenceTransformer embedder — no API costs.
    Great for offline/on-prem deployments.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self._cache = EmbeddingCache()
        logger.info("local_embedder_loaded", model=model_name, dim=self.dim)

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices, uncached_texts = [], []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            embeddings = self._model.encode(
                uncached_texts,
                batch_size=64,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
            )
            for idx, emb, text in zip(uncached_indices, embeddings, uncached_texts):
                results[idx] = emb.tolist()
                self._cache.set(text, emb.tolist())

        return np.array(results, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ── Factory ──────────────────────────────────────────────────────────────────

EmbedderType = OpenAIEmbedder | SentenceTransformerEmbedder


@lru_cache(maxsize=1)
def get_embedder(
    provider: Literal["openai", "local"] = "openai",
) -> EmbedderType:
    if provider == "local":
        return SentenceTransformerEmbedder()
    return OpenAIEmbedder()
