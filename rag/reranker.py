"""
Cross-encoder reranker for precision improvement after retrieval.

Flow: Retriever returns top-K (e.g. 10) → Reranker scores all K
pairs → Return top-N (e.g. 4) with highest cross-attention scores.

Supports:
- Local cross-encoder (ms-marco-MiniLM-L-6-v2)
- Cohere Rerank API
"""

from __future__ import annotations

import structlog

from rag.retriever import SearchResult
from utils.config import settings

logger = structlog.get_logger()


class CrossEncoderReranker:
    """
    Local cross-encoder reranker using sentence-transformers.
    Slower than bi-encoder but significantly more accurate.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        logger.info("cross_encoder_loaded", model=model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = settings.top_k_rerank,
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs)

        for result, score in zip(results, scores):
            result.rrf_score = float(score)  # overwrite with reranker score

        reranked = sorted(results, key=lambda r: r.rrf_score, reverse=True)
        return reranked[:top_n]


class CohereReranker:
    """
    Cohere Rerank API — highest quality, requires API key.
    Falls back to CrossEncoderReranker if no key is set.
    """

    def __init__(self, model: str = "rerank-english-v3.0"):
        import cohere

        if not settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY not set")
        self._client = cohere.Client(settings.cohere_api_key)
        self._model = model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = settings.top_k_rerank,
    ) -> list[SearchResult]:
        if not results:
            return []

        docs = [r.chunk.text for r in results]
        response = self._client.rerank(
            query=query,
            documents=docs,
            top_n=min(top_n, len(docs)),
            model=self._model,
        )

        reranked = []
        for hit in response.results:
            result = results[hit.index]
            result.rrf_score = float(hit.relevance_score)
            reranked.append(result)

        return reranked


def get_reranker(prefer_cohere: bool = True):
    """Get best available reranker."""
    if prefer_cohere and settings.has_cohere:
        try:
            return CohereReranker()
        except Exception as e:
            logger.warning("cohere_reranker_failed", error=str(e))

    return CrossEncoderReranker()
