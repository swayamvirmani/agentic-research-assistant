"""
FastAPI application — main entry point.

Endpoints:
  POST /query           → Run full agent pipeline
  POST /ingest/file     → Ingest uploaded file
  POST /ingest/text     → Ingest raw text
  GET  /documents       → List indexed documents
  GET  /traces          → Recent reasoning traces
  GET  /metrics         → System metrics
  GET  /health          → Health check
  WS   /ws/stream/{id}  → WebSocket streaming
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from api.schemas import (
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    KnowledgeBaseStats,
    QueryRequest,
    QueryResponse,
    Citation,
    AgentStepResponse,
)
from utils.config import settings
from utils.tracing import trace_store

logger = structlog.get_logger()

# ── Prometheus Metrics ────────────────────────────────────────────────────────

QUERY_COUNTER = Counter("ara_queries_total", "Total queries processed")
QUERY_ERRORS = Counter("ara_query_errors_total", "Total query errors")
QUERY_LATENCY = Histogram("ara_query_latency_seconds", "Query latency", buckets=[
    0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0
])
INGEST_COUNTER = Counter("ara_ingestions_total", "Total documents ingested")

# ── Global State ──────────────────────────────────────────────────────────────

_pipeline = None
_orchestrator = None
_start_time = time.time()
_metrics = {
    "total_queries": 0,
    "total_latency_ms": 0.0,
    "total_quality": 0.0,
    "errors": 0,
}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline and orchestrator on startup."""
    global _pipeline, _orchestrator
    logger.info("startup_begin")

    try:
        from rag.pipeline import RAGPipeline
        from agents.orchestrator import MultiAgentOrchestrator

        _pipeline = RAGPipeline()
        _orchestrator = MultiAgentOrchestrator(rag_pipeline=_pipeline)
        logger.info("startup_complete", chunks=_pipeline.retriever.chunk_count)
    except Exception as e:
        logger.error("startup_failed", error=str(e))

    yield

    # Persist index on shutdown
    if _pipeline:
        _pipeline.retriever.save()
        logger.info("index_persisted_on_shutdown")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic Research Assistant",
    description="Multi-agent RAG system with self-reflection",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """System health check."""
    stats = _pipeline.stats if _pipeline else {"total_chunks": 0, "total_documents": 0, "documents": []}
    return HealthResponse(
        status="healthy" if _orchestrator else "degraded",
        version="1.0.0",
        knowledge_base=KnowledgeBaseStats(**stats),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Run the full multi-agent research pipeline.
    
    The query goes through:
    1. Router → classifies query type
    2. Researcher → RAG retrieval with query expansion
    3. Analyst → extracts insights, optionally uses tools
    4. Synthesizer → drafts answer
    5. Critic → evaluates quality, may trigger retry
    6. Synthesizer → produces final polished answer
    """
    if not _orchestrator:
        raise HTTPException(503, "Orchestrator not initialized")

    query_id = str(uuid.uuid4())
    start = time.perf_counter()
    QUERY_COUNTER.inc()

    try:
        result = await _orchestrator.arun(
            query=request.query,
            trace_id=query_id,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # Update metrics
        _metrics["total_queries"] += 1
        _metrics["total_latency_ms"] += latency_ms
        _metrics["total_quality"] += result.get("quality_score", 0)

        QUERY_LATENCY.observe(latency_ms / 1000)

        # Format response
        trace = result.get("trace", {})
        steps = [
            AgentStepResponse(
                step_id=s["step_id"],
                agent=s["agent"],
                action=s["action"],
                latency_ms=s["latency_ms"],
                quality_score=s.get("quality_score"),
                tool_calls=s.get("tool_calls", []),
            )
            for s in trace.get("steps", [])
        ]

        citations = [
            Citation(**c)
            for c in result.get("citations", [])
            if all(k in c for k in ["citation_id", "doc_id", "chunk_id", "source", "filename", "relevance_score", "excerpt"])
        ]

        return QueryResponse(
            query_id=query_id,
            query=request.query,
            answer=result["answer"],
            citations=citations,
            quality_score=result.get("quality_score", 0.0),
            iterations=result.get("iterations", 1),
            total_latency_ms=round(latency_ms, 2),
            agent_steps=steps,
            tool_calls=result.get("tool_calls", []),
        )

    except Exception as e:
        QUERY_ERRORS.inc()
        _metrics["errors"] += 1
        logger.error("query_failed", query=request.query[:100], error=str(e))
        raise HTTPException(500, f"Query processing failed: {e}")


@app.post("/ingest/file", response_model=IngestResponse, tags=["Knowledge Base"])
async def ingest_file(file: UploadFile = File(...)):
    """Ingest an uploaded file (PDF, DOCX, TXT, MD, HTML)."""
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    # Save to temp location
    tmp_path = Path(f"/tmp/{uuid.uuid4()}_{file.filename}")
    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        result = _pipeline.ingest_file(
            tmp_path,
            metadata={"original_filename": file.filename},
        )
        INGEST_COUNTER.inc()
        return IngestResponse(
            doc_id=result["doc_id"],
            status=result["status"],
            chunks=result["chunks"],
            message=f"Successfully ingested {file.filename}",
        )
    except Exception as e:
        logger.error("ingest_failed", filename=file.filename, error=str(e))
        raise HTTPException(500, f"Ingestion failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/ingest/text", response_model=IngestResponse, tags=["Knowledge Base"])
async def ingest_text(request: IngestTextRequest):
    """Ingest raw text directly into the knowledge base."""
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    result = _pipeline.ingest_text(
        text=request.text,
        doc_id=request.doc_id,
        metadata=request.metadata,
    )
    INGEST_COUNTER.inc()
    return IngestResponse(
        doc_id=result["doc_id"],
        status=result["status"],
        chunks=result["chunks"],
        message="Text ingested successfully",
    )


@app.get("/documents", tags=["Knowledge Base"])
async def list_documents():
    """List all indexed documents."""
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return _pipeline.stats


@app.get("/traces", tags=["Observability"])
async def list_traces(n: int = 20):
    """Get the most recent reasoning traces."""
    return {"traces": trace_store.list_recent(n)}


@app.get("/traces/{trace_id}", tags=["Observability"])
async def get_trace(trace_id: str):
    """Get a specific reasoning trace."""
    trace = trace_store.get(trace_id)
    if not trace:
        raise HTTPException(404, f"Trace {trace_id} not found")
    return trace.to_dict()


@app.get("/system/metrics", tags=["Observability"])
async def system_metrics():
    """High-level system metrics."""
    total = _metrics["total_queries"]
    return {
        "total_queries": total,
        "avg_latency_ms": round(_metrics["total_latency_ms"] / max(total, 1), 2),
        "avg_quality_score": round(_metrics["total_quality"] / max(total, 1), 3),
        "error_rate": round(_metrics["errors"] / max(total, 1), 3),
        "knowledge_base": _pipeline.stats if _pipeline else {},
    }


@app.websocket("/ws/stream/{trace_id}")
async def websocket_stream(websocket: WebSocket, trace_id: str):
    """WebSocket endpoint for streaming agent responses."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        query = data.get("query", "")

        if not query or not _orchestrator:
            await websocket.send_json({"type": "error", "data": "Invalid request"})
            return

        # Run pipeline (streaming not implemented in this minimal version)
        # In production: stream tokens from LLM via asyncio queues
        result = await _orchestrator.arun(query=query, trace_id=trace_id)

        await websocket.send_json({
            "type": "done",
            "trace_id": trace_id,
            "data": {
                "answer": result["answer"],
                "quality_score": result["quality_score"],
                "iterations": result["iterations"],
            }
        })

    except Exception as e:
        await websocket.send_json({"type": "error", "data": str(e)})
    finally:
        await websocket.close()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Redirect to dashboard."""
    return """
    <html><head><meta http-equiv="refresh" content="0; url=/docs"></head>
    <body><a href="/docs">Go to API Docs</a></body></html>
    """
