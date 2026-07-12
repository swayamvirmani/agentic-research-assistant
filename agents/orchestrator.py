from __future__ import annotations
from typing import Annotated, Any, Literal, TypedDict

import structlog
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.analyst import AnalystAgent
from agents.critic import CriticAgent
from agents.researcher import ResearcherAgent
from agents.synthesizer import SynthesizerAgent
from agents.tools import tool_registry
from utils.config import settings
from utils.tracing import AgentStep, ReasoningTrace, StepTimer, trace_store

logger = structlog.get_logger()


# ── Graph State ───────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    # Core
    query: str
    trace_id: str

    # Agent outputs
    research_context: str
    research_citations: list[dict]
    analysis_result: str
    tool_calls_made: list[dict]
    draft_answer: str
    quality_score: float
    quality_feedback: str
    final_answer: str

    # Control flow
    iteration: int
    max_iterations: int
    query_type: Literal["factual", "analytical", "comparative", "procedural", "open_ended"]
    active_agents: list[str]

    # Streaming
    stream_tokens: list[str]

    # Trace
    trace: ReasoningTrace


# ── Orchestrator ──────────────────────────────────────────────────────────────


class MultiAgentOrchestrator:

    def __init__(self, rag_pipeline=None):
        self._researcher = ResearcherAgent(rag_pipeline=rag_pipeline)
        self._analyst = AnalystAgent()
        self._critic = CriticAgent()
        self._synthesizer = SynthesizerAgent()
        self._graph = self._build_graph()

    # ── Graph Builder ─────────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)

        # Nodes
        graph.add_node("route", self._route_node)
        graph.add_node("research", self._research_node)
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("draft", self._draft_node)
        graph.add_node("critique", self._critique_node)
        graph.add_node("finalize", self._finalize_node)

        # Edges
        graph.set_entry_point("route")
        graph.add_edge("route", "research")
        graph.add_edge("research", "analyze")
        graph.add_edge("analyze", "draft")
        graph.add_edge("draft", "critique")

        graph.add_conditional_edges(
            "critique",
            self._should_retry,
            {
                "retry": "research",
                "finalize": "finalize",
            },
        )
        graph.add_edge("finalize", END)

        return graph.compile()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _route_node(self, state: AgentState) -> AgentState:
        trace = state["trace"]
        with StepTimer(trace, AgentStep(agent="Router", action="classify_query")) as step:
            query = state["query"]

            # Classify using keywords (upgrade to LLM classifier for prod)
            q_lower = query.lower()
            if any(w in q_lower for w in ["compare", "vs", "difference", "versus"]):
                query_type = "comparative"
            elif any(w in q_lower for w in ["how to", "steps", "procedure", "process"]):
                query_type = "procedural"
            elif any(w in q_lower for w in ["analyze", "analyse", "evaluate", "assess"]):
                query_type = "analytical"
            elif any(w in q_lower for w in ["what", "who", "when", "where", "define"]):
                query_type = "factual"
            else:
                query_type = "open_ended"

            step.input_summary = query[:100]
            step.output_summary = f"Query type: {query_type}"
            step.metadata = {"query_type": query_type}

        return {
            **state,
            "query_type": query_type,
            "iteration": state.get("iteration", 0),
            "max_iterations": settings.max_iterations,
        }

    def _research_node(self, state: AgentState) -> AgentState:
        """Researcher agent: hybrid RAG retrieval."""
        trace = state["trace"]
        with StepTimer(trace, AgentStep(agent="Researcher", action="rag_retrieval")) as step:
            context, citations = self._researcher.research(
                query=state["query"],
                query_type=state.get("query_type", "factual"),
                iteration=state.get("iteration", 0),
                prior_feedback=state.get("quality_feedback", ""),
            )
            step.input_summary = state["query"][:100]
            step.output_summary = f"Retrieved {len(citations)} sources"
            step.retrieved_docs = citations[:3]

        return {
            **state,
            "research_context": context,
            "research_citations": citations,
        }

    def _analyze_node(self, state: AgentState) -> AgentState:
        """Analyst agent: extract insights, run calculations if needed."""
        trace = state["trace"]
        with StepTimer(trace, AgentStep(agent="Analyst", action="analyze_context")) as step:
            analysis, tool_calls = self._analyst.analyze(
                query=state["query"],
                context=state["research_context"],
                query_type=state.get("query_type", "factual"),
            )
            step.input_summary = f"Context ({len(state['research_context'])} chars)"
            step.output_summary = analysis[:150]
            step.tool_calls = tool_calls

        return {
            **state,
            "analysis_result": analysis,
            "tool_calls_made": state.get("tool_calls_made", []) + tool_calls,
        }

    def _draft_node(self, state: AgentState) -> AgentState:
        """Synthesizer agent: produce a draft answer."""
        trace = state["trace"]
        with StepTimer(trace, AgentStep(agent="Synthesizer", action="draft_answer")) as step:
            draft = self._synthesizer.synthesize(
                query=state["query"],
                context=state["research_context"],
                analysis=state["analysis_result"],
                citations=state["research_citations"],
                is_final=False,
            )
            step.input_summary = f"Iter {state.get('iteration', 0)}"
            step.output_summary = draft[:200]

        return {**state, "draft_answer": draft}

    def _critique_node(self, state: AgentState) -> AgentState:
        """Critic agent: evaluate draft quality and provide feedback."""
        trace = state["trace"]
        with StepTimer(trace, AgentStep(agent="Critic", action="evaluate_quality")) as step:
            score, feedback = self._critic.evaluate(
                query=state["query"],
                answer=state["draft_answer"],
                context=state["research_context"],
                citations=state["research_citations"],
            )
            step.input_summary = state["draft_answer"][:100]
            step.output_summary = f"Score: {score:.2f} | {feedback[:100]}"
            step.quality_score = score

        return {
            **state,
            "quality_score": score,
            "quality_feedback": feedback,
            "iteration": state.get("iteration", 0) + 1,
        }

    def _finalize_node(self, state: AgentState) -> AgentState:
        """Synthesizer agent: produce polished final answer."""
        trace = state["trace"]
        with StepTimer(trace, AgentStep(agent="Synthesizer", action="finalize_answer")) as step:
            final = self._synthesizer.synthesize(
                query=state["query"],
                context=state["research_context"],
                analysis=state["analysis_result"],
                citations=state["research_citations"],
                is_final=True,
                quality_feedback=state.get("quality_feedback", ""),
            )
            step.output_summary = final[:200]

        trace.complete(answer=final, success=True)
        trace_store.save(trace)

        return {**state, "final_answer": final}

    # ── Conditional Edge ──────────────────────────────────────────────────────

    def _should_retry(self, state: AgentState) -> Literal["retry", "finalize"]:
        """Decide whether to retry or finalize based on quality score."""
        score = state.get("quality_score", 0.0)
        iteration = state.get("iteration", 1)
        max_iter = state.get("max_iterations", settings.max_iterations)

        if score >= settings.reflection_threshold or iteration >= max_iter:
            logger.info(
                "critique_decision",
                decision="finalize",
                score=score,
                iteration=iteration,
            )
            return "finalize"

        logger.info(
            "critique_decision",
            decision="retry",
            score=score,
            iteration=iteration,
        )
        return "retry"

    # ── Public API ────────────────────────────────────────────────────────────

    async def arun(self, query: str, trace_id: str | None = None) -> dict:
        """Async execution of the full agent pipeline."""
        import uuid

        trace_id = trace_id or str(uuid.uuid4())
        trace = ReasoningTrace(trace_id=trace_id, query=query)

        initial_state: AgentState = {
            "query": query,
            "trace_id": trace_id,
            "research_context": "",
            "research_citations": [],
            "analysis_result": "",
            "tool_calls_made": [],
            "draft_answer": "",
            "quality_score": 0.0,
            "quality_feedback": "",
            "final_answer": "",
            "iteration": 0,
            "max_iterations": settings.max_iterations,
            "query_type": "factual",
            "active_agents": [],
            "stream_tokens": [],
            "trace": trace,
        }

        final_state = await self._graph.ainvoke(initial_state)
        return {
            "answer": final_state["final_answer"],
            "citations": final_state["research_citations"],
            "quality_score": final_state["quality_score"],
            "iterations": final_state["iteration"],
            "trace": trace.to_dict(),
            "tool_calls": final_state["tool_calls_made"],
        }

    def run(self, query: str, trace_id: str | None = None) -> dict:
        """Sync execution wrapper."""
        import asyncio
        return asyncio.run(self.arun(query, trace_id))
