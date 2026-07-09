"""
Structured logging and reasoning trace capture.
Every agent step is recorded for full observability.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from utils.config import settings

# ── Configure structlog ──────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if settings.log_level == "DEBUG"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(__import__("logging"), settings.log_level)
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


# ── Trace Data Models ────────────────────────────────────────────────────────


@dataclass
class AgentStep:
    """A single reasoning step performed by an agent."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent: str = ""
    action: str = ""
    input_summary: str = ""
    output_summary: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    retrieved_docs: list[dict] = field(default_factory=list)
    quality_score: float | None = None
    latency_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "agent": self.agent,
            "action": self.action,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "tool_calls": self.tool_calls,
            "retrieved_docs": self.retrieved_docs,
            "quality_score": self.quality_score,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ReasoningTrace:
    """Complete trace for a single user query."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_latency_ms: float = 0.0
    iteration_count: int = 0
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None
    success: bool = False
    error: str | None = None

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)
        logger.info(
            "agent_step",
            trace_id=self.trace_id,
            agent=step.agent,
            action=step.action,
            latency_ms=step.latency_ms,
            quality_score=step.quality_score,
        )

    def complete(self, answer: str, success: bool = True, error: str | None = None):
        self.final_answer = answer
        self.success = success
        self.error = error
        self.completed_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            "trace_complete",
            trace_id=self.trace_id,
            success=success,
            total_steps=len(self.steps),
            total_latency_ms=self.total_latency_ms,
        )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "total_latency_ms": self.total_latency_ms,
            "iteration_count": self.iteration_count,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "success": self.success,
            "error": self.error,
        }


# ── Context Manager for Step Timing ─────────────────────────────────────────


class StepTimer:
    """Context manager that auto-records latency for an AgentStep."""

    def __init__(self, trace: ReasoningTrace, step: AgentStep):
        self.trace = trace
        self.step = step
        self._start: float = 0.0

    def __enter__(self) -> AgentStep:
        self._start = time.perf_counter()
        return self.step

    def __exit__(self, *_):
        elapsed = (time.perf_counter() - self._start) * 1000
        self.step.latency_ms = round(elapsed, 2)
        self.trace.total_latency_ms += elapsed
        self.trace.add_step(self.step)


# ── In-memory Trace Store (swap for Redis in prod) ───────────────────────────


class TraceStore:
    def __init__(self, max_traces: int = 1000):
        self._traces: dict[str, ReasoningTrace] = {}
        self._max = max_traces

    def save(self, trace: ReasoningTrace) -> None:
        if len(self._traces) >= self._max:
            oldest = next(iter(self._traces))
            del self._traces[oldest]
        self._traces[trace.trace_id] = trace

    def get(self, trace_id: str) -> ReasoningTrace | None:
        return self._traces.get(trace_id)

    def list_recent(self, n: int = 20) -> list[dict]:
        traces = list(self._traces.values())[-n:]
        return [t.to_dict() for t in reversed(traces)]


trace_store = TraceStore()
