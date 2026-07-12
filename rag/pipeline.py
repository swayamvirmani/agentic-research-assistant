"""
End-to-end RAG pipeline.

Ingestion:  File → Extract Text → Chunk → Embed → Index
Retrieval:  Query → Hybrid Search → Rerank → Context → LLM
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Iterator

import structlog

from rag.chunker import Chunk, DocumentChunker
from rag.embeddings import get_embedder
from rag.reranker import get_reranker
from rag.retriever import HybridRetriever, SearchResult
from utils.config import settings

logger = structlog.get_logger()


# ── Document Loaders ─────────────────────────────────────────────────────────


def load_pdf(path: Path) -> str:
    """Extract text from PDF."""
    try:
        import PyPDF2

        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n\n".join(text)
    except ImportError:
        raise ImportError("pip install PyPDF2")


def load_docx(path: Path) -> str:
    """Extract text from .docx file."""
    try:
        from docx import Document

        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        raise ImportError("pip install python-docx")


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_html(path: Path) -> str:
    """Extract text from HTML, stripping tags."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    except ImportError:
        raise ImportError("pip install beautifulsoup4")


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".doc": load_docx,
    ".txt": load_txt,
    ".md": load_txt,
    ".html": load_html,
    ".htm": load_html,
}


def extract_text(path: Path) -> str:
    """Dispatch to the correct loader based on file extension."""
    suffix = path.suffix.lower()
    loader = LOADERS.get(suffix)
    if not loader:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader(path)


# ── RAG Pipeline ─────────────────────────────────────────────────────────────


class RAGPipeline:
    """
    Full RAG pipeline with:
    - Multi-format document ingestion
    - Recursive chunking
    - Hybrid retrieval (BM25 + FAISS)
    - Cross-encoder reranking
    - Context building with citations
    """

    def __init__(self):
        self.embedder = get_embedder()
        self.chunker = DocumentChunker()
        self.retriever = HybridRetriever(embedder=self.embedder)
        self.reranker = get_reranker()
        self._doc_registry: dict[str, dict] = {}  # doc_id → metadata

        # Try to load existing index
        if self.retriever.load():
            logger.info("existing_index_loaded", chunks=self.retriever.chunk_count)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_file(
        self,
        path: str | Path,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Ingest a single file into the knowledge base."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Generate stable doc_id from file content hash
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        doc_id = f"doc_{content_hash}"

        if doc_id in self._doc_registry:
            logger.info("doc_already_indexed", doc_id=doc_id, path=str(path))
            return {"doc_id": doc_id, "status": "already_indexed", "chunks": 0}

        logger.info("ingesting_file", path=str(path))

        # Extract text
        text = extract_text(path)
        if not text.strip():
            logger.warning("empty_document", path=str(path))
            return {"doc_id": doc_id, "status": "empty", "chunks": 0}

        # Chunk
        doc_metadata = {
            "source": str(path),
            "filename": path.name,
            "file_type": path.suffix.lower(),
            "doc_id": doc_id,
            **(metadata or {}),
        }
        chunks = self.chunker.chunk_document(text, doc_id, doc_metadata)

        # Index
        self.retriever.add_chunks(chunks)
        self._doc_registry[doc_id] = {
            **doc_metadata,
            "chunk_count": len(chunks),
            "char_count": len(text),
        }

        logger.info("file_ingested", doc_id=doc_id, chunks=len(chunks))
        return {"doc_id": doc_id, "status": "indexed", "chunks": len(chunks)}

    def ingest_directory(
        self,
        directory: str | Path,
        glob: str = "**/*",
        metadata: dict | None = None,
    ) -> list[dict]:
        """Ingest all supported files in a directory."""
        directory = Path(directory)
        results = []
        for path in sorted(directory.glob(glob)):
            if path.suffix.lower() in LOADERS and path.is_file():
                try:
                    result = self.ingest_file(path, metadata)
                    results.append(result)
                except Exception as e:
                    logger.error("ingest_failed", path=str(path), error=str(e))
                    results.append({"path": str(path), "status": "error", "error": str(e)})

        self.retriever.save()
        return results

    def ingest_text(
        self,
        text: str,
        doc_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Ingest raw text directly."""
        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
        chunks = self.chunker.chunk_document(
            text, doc_id, metadata or {"source": "direct_input"}
        )
        self.retriever.add_chunks(chunks)
        self._doc_registry[doc_id] = {"doc_id": doc_id, "chunk_count": len(chunks)}
        return {"doc_id": doc_id, "status": "indexed", "chunks": len(chunks)}

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = settings.top_k_retrieval,
        rerank: bool = True,
        top_n_rerank: int = settings.top_k_rerank,
    ) -> list[SearchResult]:
        """Full retrieval pipeline: search → rerank."""
        results = self.retriever.search(query, top_k=top_k)
        logger.debug("retrieved", query=query[:60], count=len(results))

        if rerank and results:
            results = self.reranker.rerank(query, results, top_n=top_n_rerank)
            logger.debug("reranked", count=len(results))

        return results

    def build_context(
        self,
        results: list[SearchResult],
        max_tokens: int = 3000,
    ) -> tuple[str, list[dict]]:
        """
        Build a numbered context string from search results.
        Returns (context_text, citations).
        """
        context_parts = []
        citations = []
        total_tokens = 0

        for i, result in enumerate(results, 1):
            chunk = result.chunk
            token_est = chunk.token_count

            if total_tokens + token_est > max_tokens:
                break

            context_parts.append(
                f"[{i}] {chunk.text}"
            )
            citations.append({
                "citation_id": i,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "source": chunk.metadata.get("source", "unknown"),
                "filename": chunk.metadata.get("filename", "unknown"),
                "relevance_score": round(result.rrf_score, 4),
                "excerpt": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            })
            total_tokens += token_est

        context = "\n\n---\n\n".join(context_parts)
        return context, citations

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "total_chunks": self.retriever.chunk_count,
            "total_documents": len(self._doc_registry),
            "documents": list(self._doc_registry.values()),
        }
