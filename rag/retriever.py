"""
Hybrid Retriever: BM25 sparse + FAISS dense + Reciprocal Rank Fusion (RRF).

RRF Formula: score(d) = Σ 1 / (k + rank(d))
where k=60 is a smoothing constant.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import structlog
from rank_bm25 import BM25Okapi

from rag.chunker import Chunk
from rag.embeddings import EmbedderType, get_embedder
from utils.config import settings

logger = structlog.get_logger()


class SearchResult:
    """A retrieved document chunk with scores."""

    def __init__(
        self,
        chunk: Chunk,
        dense_score: float = 0.0,
        bm25_score: float = 0.0,
        rrf_score: float = 0.0,
    ):
        self.chunk = chunk
        self.dense_score = dense_score
        self.bm25_score = bm25_score
        self.rrf_score = rrf_score

    def to_dict(self) -> dict:
        return {
            **self.chunk.to_dict(),
            "dense_score": round(self.dense_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "rrf_score": round(self.rrf_score, 4),
        }


class HybridRetriever:
    """
    Production hybrid retriever combining:
    1. BM25 (lexical, exact match) — great for specific terms/names
    2. FAISS (semantic, dense) — great for paraphrase & context
    3. RRF fusion — combines both rank lists optimally

    Usage:
        retriever = HybridRetriever(embedder)
        retriever.add_chunks(chunks)
        results = retriever.search("What is attention mechanism?", top_k=5)
    """

    RRF_K = 60  # standard RRF smoothing constant

    def __init__(
        self,
        embedder: EmbedderType | None = None,
        alpha: float = settings.hybrid_alpha,
        index_path: str = settings.faiss_index_path,
    ):
        self.embedder = embedder or get_embedder()
        self.alpha = alpha  # weight for dense (1-alpha for BM25)
        self.index_path = Path(index_path)

        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._tokenized_corpus: list[list[str]] = []

    # ── Indexing ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to both BM25 and FAISS indexes."""
        if not chunks:
            return

        start_idx = len(self._chunks)
        self._chunks.extend(chunks)
        texts = [c.text for c in chunks]

        # 1. BM25 index
        new_tokenized = [self._tokenize(t) for t in texts]
        self._tokenized_corpus.extend(new_tokenized)
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        # 2. FAISS index
        embeddings = self.embedder.embed(texts)
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        if self._faiss_index is None:
            dim = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)

        self._faiss_index.add(embeddings.astype(np.float32))
        logger.info(
            "index_updated",
            added=len(chunks),
            total=len(self._chunks),
        )

    def search(
        self,
        query: str,
        top_k: int = settings.top_k_retrieval,
    ) -> list[SearchResult]:
        """Hybrid search returning top_k results via RRF fusion."""
        if not self._chunks:
            logger.warning("retriever_empty")
            return []

        top_k = min(top_k, len(self._chunks))

        # Parallel retrieval
        dense_results = self._dense_search(query, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)

        # RRF fusion
        fused = self._rrf_fusion(dense_results, bm25_results, top_k)
        return fused

    # ── Private ──────────────────────────────────────────────────────────────

    def _dense_search(
        self, query: str, top_k: int
    ) -> list[tuple[int, float]]:
        """Return (chunk_index, score) from FAISS."""
        q_emb = self.embedder.embed_one(query).reshape(1, -1)
        norm = np.linalg.norm(q_emb)
        if norm > 1e-8:
            q_emb = q_emb / norm

        scores, indices = self._faiss_index.search(
            q_emb.astype(np.float32), top_k
        )
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

    def _bm25_search(
        self, query: str, top_k: int
    ) -> list[tuple[int, float]]:
        """Return (chunk_index, score) from BM25."""
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    def _rrf_fusion(
        self,
        dense: list[tuple[int, float]],
        bm25: list[tuple[int, float]],
        top_k: int,
    ) -> list[SearchResult]:
        """Combine rank lists via Reciprocal Rank Fusion."""
        rrf_scores: dict[int, float] = {}

        # Dense ranks
        for rank, (idx, score) in enumerate(dense):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (
                self.alpha / (self.RRF_K + rank)
            )

        # BM25 ranks
        for rank, (idx, score) in enumerate(bm25):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (
                (1 - self.alpha) / (self.RRF_K + rank)
            )

        # Build score maps for raw scores
        dense_map = {idx: score for idx, score in dense}
        bm25_map = {idx: score for idx, score in bm25}

        # Sort by RRF score
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, rrf_score in sorted_items[:top_k]:
            if idx < len(self._chunks):
                results.append(
                    SearchResult(
                        chunk=self._chunks[idx],
                        dense_score=dense_map.get(idx, 0.0),
                        bm25_score=bm25_map.get(idx, 0.0),
                        rrf_score=rrf_score,
                    )
                )

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer for BM25."""
        import re
        tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
        # Remove very short tokens
        return [t for t in tokens if len(t) > 1]

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist index to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(
            self._faiss_index,
            str(self.index_path / "faiss.index"),
        )

        # Save BM25 + chunks
        state = {
            "chunks": self._chunks,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(self.index_path / "state.pkl", "wb") as f:
            pickle.dump(state, f)

        logger.info("index_saved", path=str(self.index_path))

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        faiss_path = self.index_path / "faiss.index"
        state_path = self.index_path / "state.pkl"

        if not faiss_path.exists() or not state_path.exists():
            return False

        self._faiss_index = faiss.read_index(str(faiss_path))
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        self._chunks = state["chunks"]
        self._tokenized_corpus = state["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info("index_loaded", total_chunks=len(self._chunks))
        return True

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
