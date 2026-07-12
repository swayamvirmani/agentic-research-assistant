

from __future__ import annotations

import structlog

from models.llm import Message, get_llm

logger = structlog.get_logger()

QUERY_EXPANSION_PROMPT = """You are a search query optimizer. Given a user query, generate
3 alternative search queries that would retrieve complementary information.
Return ONLY a JSON object with key "queries" containing a list of 3 strings.

Original query: {query}
Prior feedback (if any): {feedback}"""


class ResearcherAgent:
    """
    Retrieves and ranks relevant documents for a query.
    Uses query expansion to improve recall on the first iteration.
    On retry iterations, uses critic feedback to reformulate queries.
    """

    def __init__(self, rag_pipeline=None):
        self._rag = rag_pipeline
        self._llm = get_llm(temperature=0.2)

    def research(
        self,
        query: str,
        query_type: str = "factual",
        iteration: int = 0,
        prior_feedback: str = "",
    ) -> tuple[str, list[dict]]:
        """
        Returns (context_string, citations_list).
        """
        if self._rag is None or self._rag.retriever.chunk_count == 0:
            logger.warning("rag_empty_or_unavailable")
            return self._web_fallback(query), []

        # Expand queries for first iteration
        if iteration == 0:
            queries = self._expand_query(query, prior_feedback)
        else:
            # On retry, add feedback-informed query
            queries = [query, f"{query} {prior_feedback}"]

        # Retrieve for all queries and merge
        all_results = []
        seen_chunk_ids: set[str] = set()

        for q in queries:
            results = self._rag.retrieve(query=q, rerank=True)
            for r in results:
                if r.chunk.chunk_id not in seen_chunk_ids:
                    all_results.append(r)
                    seen_chunk_ids.add(r.chunk.chunk_id)

        # Re-sort by score and take top
        all_results.sort(key=lambda r: r.rrf_score, reverse=True)
        top_results = all_results[:8]

        context, citations = self._rag.build_context(top_results)
        return context, citations

    def _expand_query(self, query: str, feedback: str = "") -> list[str]:
        """Generate expanded queries using LLM."""
        try:
            result = self._llm.json_chat(
                messages=[
                    Message(
                        "user",
                        QUERY_EXPANSION_PROMPT.format(query=query, feedback=feedback),
                    )
                ]
            )
            expanded = result.get("queries", [])
            queries = [query] + [q for q in expanded if q and q != query]
            logger.debug("query_expanded", original=query, expanded=expanded)
            return queries[:4]
        except Exception as e:
            logger.warning("query_expansion_failed", error=str(e))
            return [query]

    def _web_fallback(self, query: str) -> str:
        """Return a placeholder context when no documents are indexed."""
        return (
            f"No documents are currently indexed. "
            f"The query was: '{query}'. "
            f"Please ingest documents using the /ingest endpoint before querying."
        )
