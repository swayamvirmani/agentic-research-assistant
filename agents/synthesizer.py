

from __future__ import annotations

import structlog

from models.llm import Message, get_llm

logger = structlog.get_logger()

SYNTHESIZER_SYSTEM = """You are an expert research writer. Synthesize the provided 
context and analysis into a clear, well-structured answer.

Guidelines:
- Open with a direct answer to the question
- Support claims with inline citations like [1], [2], [3]
- Use markdown formatting (headers, bullets) for complex answers
- End with a brief conclusion if the answer is long
- Be comprehensive but concise
- Do NOT hallucinate — only use facts from the provided context
- If the context is insufficient, say so explicitly"""

DRAFT_PROMPT = """Query: {query}
Query Type: {query_type}

Research Context:
---
{context}
---

Analysis Findings:
---
{analysis}
---

Write a comprehensive draft answer that directly addresses the query.
Use [1], [2] etc. to cite sources from the context."""

FINAL_PROMPT = """Query: {query}

Research Context:
---
{context}
---

Analysis:
---
{analysis}
---

Quality Feedback to Address:
{feedback}

Write the FINAL, polished answer. Address all feedback points.
Be thorough, accurate, and well-cited. Use [1], [2] etc. for citations."""


class SynthesizerAgent:
    """
    Produces draft and final answers from research context and analysis.
    """

    def __init__(self):
        self._llm = get_llm(temperature=0.3)

    def synthesize(
        self,
        query: str,
        context: str,
        analysis: str,
        citations: list[dict],
        is_final: bool = False,
        quality_feedback: str = "",
        query_type: str = "factual",
    ) -> str:
        """Generate draft or final answer."""
        if is_final:
            prompt = FINAL_PROMPT.format(
                query=query,
                context=context[:4000],
                analysis=analysis[:1500],
                feedback=quality_feedback or "Ensure completeness and accuracy.",
            )
        else:
            prompt = DRAFT_PROMPT.format(
                query=query,
                query_type=query_type,
                context=context[:4000],
                analysis=analysis[:1500],
            )

        response = self._llm.chat(
            messages=[Message("user", prompt)],
            system=SYNTHESIZER_SYSTEM,
            max_tokens=2048,
        )

        answer = response.content.strip()

        # Append formatted citations at the end
        if citations and is_final:
            answer = self._append_citations(answer, citations)

        return answer

    @staticmethod
    def _append_citations(answer: str, citations: list[dict]) -> str:
        """Append a formatted citations section."""
        if not citations:
            return answer

        refs = ["\n\n---\n**References**\n"]
        for c in citations:
            filename = c.get("filename", "unknown")
            source = c.get("source", "")
            score = c.get("relevance_score", 0)
            refs.append(f"[{c['citation_id']}] {filename} (relevance: {score:.2f})")

        return answer + "\n".join(refs)
