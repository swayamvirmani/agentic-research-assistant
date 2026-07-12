

from __future__ import annotations

import json
import re

import structlog

from models.llm import Message, get_llm

logger = structlog.get_logger()

CRITIC_SYSTEM = """You are a rigorous AI answer evaluator. Score the given answer 
on these criteria, each from 0 to 1:

1. faithfulness: Is every claim supported by the provided context? (0=hallucinated, 1=fully grounded)
2. completeness: Does it answer ALL aspects of the question? (0=partial, 1=complete)
3. coherence: Is it clear, well-structured, and logically sound? (0=incoherent, 1=excellent)
4. citation_quality: Are sources properly cited and relevant? (0=no citations, 1=well-cited)

Respond ONLY with valid JSON:
{
  "scores": {
    "faithfulness": 0.0,
    "completeness": 0.0,
    "coherence": 0.0,
    "citation_quality": 0.0
  },
  "overall": 0.0,
  "feedback": "Specific, actionable feedback for improvement",
  "issues": ["list", "of", "specific", "issues"]
}"""

CRITIC_PROMPT = """Question: {query}

Retrieved Context (ground truth):
---
{context}
---

Answer to evaluate:
---
{answer}
---

Evaluate this answer strictly and provide your JSON assessment."""


class CriticAgent:
    """
    Evaluates answer quality and provides structured feedback.
    Used to drive the self-reflection retry loop.
    """

    SCORE_WEIGHTS = {
        "faithfulness": 0.40,    # Most important — avoid hallucination
        "completeness": 0.30,
        "coherence": 0.20,
        "citation_quality": 0.10,
    }

    def __init__(self):
        self._llm = get_llm(temperature=0.0)  # deterministic scoring

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        citations: list[dict],
    ) -> tuple[float, str]:
        """
        Returns (overall_score, feedback_string).
        Score range: [0.0, 1.0]
        """
        if not answer.strip():
            return 0.0, "Answer is empty"

        prompt = CRITIC_PROMPT.format(
            query=query,
            context=context[:3000],
            answer=answer[:2000],
        )

        try:
            result = self._llm.json_chat(
                messages=[Message("user", prompt)],
                system=CRITIC_SYSTEM,
            )

            scores = result.get("scores", {})
            overall = self._compute_weighted_score(scores)
            feedback = result.get("feedback", "No feedback provided")
            issues = result.get("issues", [])

            if issues:
                feedback = f"{feedback}\n\nIssues: {'; '.join(issues)}"

            logger.info(
                "critic_scores",
                faithfulness=scores.get("faithfulness"),
                completeness=scores.get("completeness"),
                coherence=scores.get("coherence"),
                citation_quality=scores.get("citation_quality"),
                overall=overall,
            )

            return overall, feedback

        except Exception as e:
            logger.warning("critic_failed", error=str(e))
            # Default to passable score on error (don't block the pipeline)
            return settings.reflection_threshold, f"Evaluation failed: {e}"

    def _compute_weighted_score(self, scores: dict[str, float]) -> float:
        """Compute weighted average of dimension scores."""
        total = 0.0
        weight_sum = 0.0
        for dim, weight in self.SCORE_WEIGHTS.items():
            score = scores.get(dim, 0.5)
            score = max(0.0, min(1.0, float(score)))
            total += score * weight
            weight_sum += weight
        return round(total / weight_sum if weight_sum > 0 else 0.5, 3)


# Fix missing import
from utils.config import settings
