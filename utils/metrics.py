"""
RAGAS-based evaluation pipeline.

Metrics computed:
- Faithfulness: Are claims grounded in context?
- Answer Relevancy: Does the answer address the question?
- Context Recall: Did retrieval get the relevant chunks?
- Context Precision: Are retrieved chunks precise?

Usage:
    evaluator = RAGASEvaluator()
    results = evaluator.evaluate(dataset)
    evaluator.report(results)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class EvalSample:
    """A single evaluation data point."""
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


@dataclass
class EvalResult:
    """Evaluation results for a dataset."""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    answer_correctness: float = 0.0
    n_samples: int = 0
    per_sample: list[dict] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        scores = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_recall,
            self.context_precision,
        ]
        return sum(scores) / len(scores)

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_recall": round(self.context_recall, 4),
            "context_precision": round(self.context_precision, 4),
            "answer_correctness": round(self.answer_correctness, 4),
            "overall_score": round(self.overall_score, 4),
            "n_samples": self.n_samples,
        }


class RAGASEvaluator:
    """
    Evaluates RAG pipeline quality using the RAGAS framework.
    Falls back to LLM-based evaluation if RAGAS is unavailable.
    """

    def __init__(self, use_ragas: bool = True):
        self.use_ragas = use_ragas
        self._check_ragas()

    def _check_ragas(self):
        try:
            import ragas  # noqa: F401
            self._ragas_available = True
        except ImportError:
            logger.warning("ragas_not_installed", fallback="LLM-based eval")
            self._ragas_available = False

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        """Run evaluation on a list of samples."""
        if self._ragas_available and self.use_ragas:
            return self._ragas_evaluate(samples)
        return self._llm_evaluate(samples)

    def _ragas_evaluate(self, samples: list[EvalSample]) -> EvalResult:
        """Use RAGAS library for evaluation."""
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from utils.config import settings

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
        }

        dataset = Dataset.from_dict(data)
        llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
        embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)

        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ],
            llm=llm,
            embeddings=embeddings,
        )

        df = result.to_pandas()
        return EvalResult(
            faithfulness=float(df["faithfulness"].mean()),
            answer_relevancy=float(df["answer_relevancy"].mean()),
            context_recall=float(df["context_recall"].mean()),
            context_precision=float(df["context_precision"].mean()),
            n_samples=len(samples),
            per_sample=df.to_dict("records"),
        )

    def _llm_evaluate(self, samples: list[EvalSample]) -> EvalResult:
        """Fallback: LLM-based metric estimation."""
        from models.llm import Message, get_llm

        llm = get_llm(temperature=0.0)
        per_sample = []
        totals = {k: 0.0 for k in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]}

        for sample in samples:
            prompt = f"""Evaluate this RAG answer. Return JSON only.

Question: {sample.question}
Context: {" | ".join(sample.contexts[:3])[:2000]}
Answer: {sample.answer}
Ground Truth: {sample.ground_truth}

JSON with scores 0-1:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0, "context_precision": 0.0}}"""

            result = llm.json_chat([Message("user", prompt)])
            per_sample.append({**result, "question": sample.question})

            for k in totals:
                totals[k] += float(result.get(k, 0.5))

        n = max(len(samples), 1)
        return EvalResult(
            faithfulness=totals["faithfulness"] / n,
            answer_relevancy=totals["answer_relevancy"] / n,
            context_recall=totals["context_recall"] / n,
            context_precision=totals["context_precision"] / n,
            n_samples=n,
            per_sample=per_sample,
        )

    def load_dataset(self, path: str) -> list[EvalSample]:
        """Load evaluation dataset from JSON file."""
        data = json.loads(Path(path).read_text())
        return [
            EvalSample(
                question=item["question"],
                answer=item.get("answer", ""),
                contexts=item.get("contexts", []),
                ground_truth=item.get("ground_truth", ""),
            )
            for item in data
        ]

    def save_results(self, result: EvalResult, path: str) -> None:
        """Save evaluation results to JSON."""
        output = {
            **result.to_dict(),
            "per_sample": result.per_sample,
        }
        Path(path).write_text(json.dumps(output, indent=2))
        logger.info("eval_results_saved", path=path, overall=result.overall_score)

    def report(self, result: EvalResult) -> str:
        """Generate a human-readable evaluation report."""
        lines = [
            "=" * 50,
            "  RAGAS EVALUATION REPORT",
            "=" * 50,
            f"  Samples evaluated:    {result.n_samples}",
            f"  Overall Score:        {result.overall_score:.4f}",
            "-" * 50,
            f"  Faithfulness:         {result.faithfulness:.4f}",
            f"  Answer Relevancy:     {result.answer_relevancy:.4f}",
            f"  Context Recall:       {result.context_recall:.4f}",
            f"  Context Precision:    {result.context_precision:.4f}",
            "=" * 50,
        ]
        report = "\n".join(lines)
        print(report)
        return report
