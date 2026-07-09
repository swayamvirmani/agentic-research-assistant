"""
Pydantic schemas for request/response validation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Research question")
    stream: bool = Field(False, description="Stream response via WebSocket")
    top_k: int = Field(10, ge=1, le=50, description="Number of docs to retrieve")
    rerank: bool = Field(True, description="Enable cross-encoder reranking")

    model_config = {"json_schema_extra": {"example": {
        "query": "What are the key principles of transformer architecture?",
        "stream": False,
        "top_k": 10,
        "rerank": True,
    }}}


class IngestURLRequest(BaseModel):
    url: str = Field(..., description="URL to fetch and ingest")
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Text to ingest")
    doc_id: str | None = Field(None, description="Optional custom document ID")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Response Models ───────────────────────────────────────────────────────────


class Citation(BaseModel):
    citation_id: int
    doc_id: str
    chunk_id: str
    source: str
    filename: str
    relevance_score: float
    excerpt: str


class AgentStepResponse(BaseModel):
    step_id: str
    agent: str
    action: str
    latency_ms: float
    quality_score: float | None = None
    tool_calls: list[dict] = []


class QueryResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    citations: list[Citation]
    quality_score: float
    iterations: int
    total_latency_ms: float
    agent_steps: list[AgentStepResponse]
    tool_calls: list[dict] = []

    model_config = {"json_schema_extra": {"example": {
        "query_id": "abc123",
        "query": "What is attention?",
        "answer": "Attention is a mechanism...",
        "citations": [],
        "quality_score": 0.92,
        "iterations": 1,
        "total_latency_ms": 2341.5,
        "agent_steps": [],
    }}}


class IngestResponse(BaseModel):
    doc_id: str
    status: Literal["indexed", "already_indexed", "empty", "error"]
    chunks: int
    message: str = ""


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    filename: str
    chunk_count: int
    file_type: str = ""


class KnowledgeBaseStats(BaseModel):
    total_documents: int
    total_chunks: int
    documents: list[dict]


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    knowledge_base: KnowledgeBaseStats
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    avg_quality_score: float
    cache_hit_rate: float
    success_rate: float


# ── WebSocket Messages ────────────────────────────────────────────────────────


class WSMessage(BaseModel):
    type: Literal["token", "step", "done", "error"]
    data: Any
    trace_id: str
