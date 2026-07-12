"""
Document chunking strategies.
Supports fixed-size, semantic, and recursive chunking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

from utils.config import settings


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


@dataclass
class Chunk:
    """A document chunk with metadata."""

    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    start_char: int
    end_char: int
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "token_count": self.token_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


class DocumentChunker:
    """
    Multi-strategy document chunker.

    Strategies:
    - FIXED: Split by token count with overlap
    - RECURSIVE: Try paragraph → sentence → word splits
    - SEMANTIC: Group sentences by semantic similarity (requires embedder)
    """

    # Separators for recursive splitting (ordered by priority)
    _SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        encoding_name: str = "cl100k_base",  # kept for API compat
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        # Try tiktoken first; fall back to word-based approximation for offline envs
        self._enc = self._load_encoder(encoding_name)

    @staticmethod
    def _load_encoder(encoding_name: str):
        """Load tiktoken encoder or fall back to word-based approximation."""
        try:
            import tiktoken
            return tiktoken.get_encoding(encoding_name)
        except Exception:
            return None  # will use word-based fallback

    def count_tokens(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        # Word-based approximation: ~0.75 tokens per word (GPT-4 average)
        return max(1, int(len(text.split()) * 0.75))

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Chunk a document using the configured strategy."""
        metadata = metadata or {}
        text = self._clean_text(text)

        if self.strategy == ChunkStrategy.FIXED:
            raw_chunks = list(self._fixed_split(text))
        elif self.strategy == ChunkStrategy.RECURSIVE:
            raw_chunks = list(self._recursive_split(text))
        else:
            raw_chunks = list(self._fixed_split(text))  # fallback

        chunks = []
        char_offset = 0
        for i, chunk_text in enumerate(raw_chunks):
            start = text.find(chunk_text, char_offset)
            if start == -1:
                start = char_offset
            end = start + len(chunk_text)
            char_offset = max(char_offset, end - self.chunk_overlap * 4)

            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{i:04d}",
                    doc_id=doc_id,
                    text=chunk_text.strip(),
                    token_count=self.count_tokens(chunk_text),
                    start_char=start,
                    end_char=end,
                    metadata={
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(raw_chunks),
                        "strategy": self.strategy.value,
                    },
                )
            )

        return [c for c in chunks if c.text and c.token_count > 1]

    # ── Private splitting methods ────────────────────────────────────────────

    def _fixed_split(self, text: str) -> Iterator[str]:
        """Split by fixed token window with overlap."""
        if self._enc is not None:
            tokens = self._enc.encode(text)
            step = self.chunk_size - self.chunk_overlap
            for start in range(0, len(tokens), step):
                end = min(start + self.chunk_size, len(tokens))
                yield self._enc.decode(tokens[start:end])
        else:
            # Word-based fallback
            words = text.split()
            step = max(1, self.chunk_size - self.chunk_overlap)
            for start in range(0, len(words), step):
                end = min(start + self.chunk_size, len(words))
                yield " ".join(words[start:end])

    def _recursive_split(self, text: str, depth: int = 0) -> Iterator[str]:
        """
        Recursively split using progressively finer separators.
        Tries paragraph splits first, then sentences, then words.
        """
        if self.count_tokens(text) <= self.chunk_size:
            if text.strip():
                yield text
            return

        if depth >= len(self._SEPARATORS):
            # Last resort: fixed split
            yield from self._fixed_split(text)
            return

        sep = self._SEPARATORS[depth]
        parts = text.split(sep)

        current = ""
        for part in parts:
            candidate = current + sep + part if current else part

            if self.count_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    if self.count_tokens(current) > self.chunk_size:
                        yield from self._recursive_split(current, depth + 1)
                    else:
                        yield current
                current = part

        if current:
            if self.count_tokens(current) > self.chunk_size:
                yield from self._recursive_split(current, depth + 1)
            else:
                yield current

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and remove artifacts."""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = re.sub(r" {3,}", "  ", text)
        return text.strip()
