"""RAG Evaluation Runner: execute full evaluation pipeline and compute metrics."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.core.database import SessionLocal
from app.core.logging import get_logger
from app.evaluation.llm_judge import get_llm_judge
from app.evaluation.metrics import (
    analyze_retrieval,
    batch_precision_at_k,
    batch_recall_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.models.models import EvaluationSample, EvaluationTask
from app.rag.pipeline import get_rag_pipeline

logger = get_logger(__name__)

DEFAULT_EVAL_PATH = Path("./data/eval/rag_eval.json")


class EvaluationRunner:
    """Execute RAG evaluation across a dataset and compute all metrics.

    Metrics:
    - Retrieval: Recall@K, Precision@K, MRR
    - Generation: Accuracy, Faithfulness, Answer Relevancy
    - Performance: Latency
    """

    def __init__(self) -> None:
        self.rag = get_rag_pipeline()
        self.judge = get_llm_judge()

    def load_dataset(self, dataset_name: str = "rag_eval") -> list[dict[str, Any]]:
        """Load evaluation dataset from JSON file."""
        path = Path(f"./data/eval/{dataset_name}.json")
        if not path.exists():
            path = DEFAULT_EVAL_PATH
        if not path.exists():
            logger.error("Evaluation dataset not found: %s", path)
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        return data.get("questions", data.get("samples", []))

    async def run_evaluation(
        self,
        task_id: str,
        kb_id: str,
        dataset_name: str = "rag_eval",
        rag_version: str = "hybrid_rerank_rewrite",
        max_samples: int | None = None,
    ) -> dict[str, Any]:
        """Run full evaluation for a task.

        Args:
            task_id: EvaluationTask ID.
            kb_id: Knowledge base ID to evaluate against.
            dataset_name: Name of eval dataset file.
            rag_version: RAG pipeline version to evaluate.
            max_samples: Optional limit on number of samples.

        Returns:
            Aggregated metrics dict.
        """
        dataset = self.load_dataset(dataset_name)
        if max_samples:
            dataset = dataset[:max_samples]

        logger.info(
            "Starting evaluation: task=%s, version=%s, samples=%d",
            task_id, rag_version, len(dataset),
        )

        # Update task status
        db = SessionLocal()
        task = db.query(EvaluationTask).filter(EvaluationTask.id == task_id).first()
        if task:
            task.status = "RUNNING"
            task.total_samples = len(dataset)
            db.commit()
        db.close()

        all_samples: list[dict[str, Any]] = []
        retrieval_pairs: list[tuple[list[str], list[str]]] = []
        latencies: list[int] = []

        for i, sample in enumerate(dataset):
            result = await self._evaluate_sample(kb_id, sample, rag_version)
            all_samples.append(result)
            retrieval_pairs.append(
                (sample.get("expected_sources", []), result.get("retrieved_sources", []))
            )
            latencies.append(result.get("latency_ms", 0))

            # Save sample to DB
            self._save_sample(task_id, sample, result)

            # Update progress
            db = SessionLocal()
            task = db.query(EvaluationTask).filter(EvaluationTask.id == task_id).first()
            if task:
                task.completed_samples = i + 1
                db.commit()
            db.close()

            logger.info("Evaluated %d/%d", i + 1, len(dataset))

        # Compute aggregated metrics
        metrics = self._compute_aggregated_metrics(all_samples, retrieval_pairs, latencies)

        # Save final metrics
        db = SessionLocal()
        task = db.query(EvaluationTask).filter(EvaluationTask.id == task_id).first()
        if task:
            task.status = "COMPLETED"
            task.metrics = metrics
            db.commit()
        db.close()

        logger.info("Evaluation complete: %s", metrics)
        return metrics

    async def _evaluate_sample(
        self, kb_id: str, sample: dict[str, Any], rag_version: str
    ) -> dict[str, Any]:
        """Evaluate a single question."""
        question = sample.get("question", "")
        expected_sources = sample.get("expected_sources", [])
        ground_truth = sample.get("ground_truth", "")

        start = time.time()

        # Step 1: Retrieval
        retrieval_result = await self.rag.retrieve_only(
            kb_id=kb_id,
            query=question,
            rag_version=rag_version,
            top_k=10,
        )
        retrieved_docs = retrieval_result.get("results", [])
        retrieved_sources = [
            d.get("source", "") for d in retrieved_docs if d.get("source")
        ]
        retrieved_scores = [d.get("score", 0) for d in retrieved_docs]

        # Step 2: Generation (RAG pipeline)
        try:
            rag_result = await self.rag.query(
                kb_id=kb_id,
                question=question,
                rag_version=rag_version,
                top_k=5,
            )
            generated_answer = rag_result.get("answer", "")
            context = "\n".join(
                d.get("content", "") for d in rag_result.get("retrieved_docs", [])
            )
        except Exception as exc:
            logger.error("RAG generation failed for sample: %s", exc)
            generated_answer = ""
            context = ""

        latency_ms = int((time.time() - start) * 1000)

        # Step 3: Compute retrieval metrics
        recall_1 = recall_at_k(expected_sources, retrieved_sources, 1)
        recall_3 = recall_at_k(expected_sources, retrieved_sources, 3)
        recall_5 = recall_at_k(expected_sources, retrieved_sources, 5)
        recall_10 = recall_at_k(expected_sources, retrieved_sources, 10)
        precision_1 = precision_at_k(expected_sources, retrieved_sources, 1)
        precision_3 = precision_at_k(expected_sources, retrieved_sources, 3)
        precision_5 = precision_at_k(expected_sources, retrieved_sources, 5)
        precision_10 = precision_at_k(expected_sources, retrieved_sources, 10)
        mrr = reciprocal_rank(expected_sources, retrieved_sources)

        # Step 4: Compute generation metrics (LLM judge)
        accuracy_result = await self.judge.evaluate_accuracy(
            question, ground_truth, generated_answer
        )
        faithfulness_result = await self.judge.evaluate_faithfulness(
            question, context, generated_answer
        )
        relevancy_result = await self.judge.evaluate_relevancy(
            question, generated_answer
        )

        # Step 5: Analysis
        analysis = analyze_retrieval(
            expected_sources, retrieved_sources, recall_5, accuracy_result["score"]
        )

        return {
            "question": question,
            "ground_truth": ground_truth,
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            "retrieved_scores": retrieved_scores,
            "generated_answer": generated_answer,
            "latency_ms": latency_ms,
            "metrics": {
                "recall_at_1": recall_1,
                "recall_at_3": recall_3,
                "recall_at_5": recall_5,
                "recall_at_10": recall_10,
                "precision_at_1": precision_1,
                "precision_at_3": precision_3,
                "precision_at_5": precision_5,
                "precision_at_10": precision_10,
                "mrr": mrr,
                "accuracy": accuracy_result["score"],
                "accuracy_reason": accuracy_result["reason"],
                "faithfulness": faithfulness_result["score"],
                "faithfulness_reason": faithfulness_result["reason"],
                "answer_relevancy": relevancy_result["score"],
                "answer_relevancy_reason": relevancy_result["reason"],
            },
            "analysis": analysis,
        }

    def _save_sample(
        self, task_id: str, sample: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """Save evaluation sample to database."""
        try:
            db = SessionLocal()
            db.add(EvaluationSample(
                task_id=task_id,
                question=result["question"],
                ground_truth=result.get("ground_truth"),
                expected_sources=result.get("expected_sources", []),
                category=sample.get("category"),
                difficulty=sample.get("difficulty"),
                retrieved_sources=result.get("retrieved_sources", []),
                retrieved_scores=result.get("retrieved_scores", []),
                generated_answer=result.get("generated_answer"),
                metrics=result.get("metrics", {}),
                latency_ms=result.get("latency_ms", 0),
                analysis=result.get("analysis"),
            ))
            db.commit()
            db.close()
        except Exception as exc:
            logger.error("Failed to save evaluation sample: %s", exc)

    @staticmethod
    def _compute_aggregated_metrics(
        samples: list[dict[str, Any]],
        retrieval_pairs: list[tuple[list[str], list[str]]],
        latencies: list[int],
    ) -> dict[str, Any]:
        """Compute aggregated metrics from all samples."""
        if not samples:
            return {}

        n = len(samples)

        def avg(key: str) -> float:
            vals = [s["metrics"].get(key, 0) for s in samples]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "total_samples": n,
            "recall_at_1": round(avg("recall_at_1"), 4),
            "recall_at_3": round(avg("recall_at_3"), 4),
            "recall_at_5": round(avg("recall_at_5"), 4),
            "recall_at_10": round(avg("recall_at_10"), 4),
            "precision_at_1": round(avg("precision_at_1"), 4),
            "precision_at_3": round(avg("precision_at_3"), 4),
            "precision_at_5": round(avg("precision_at_5"), 4),
            "precision_at_10": round(avg("precision_at_10"), 4),
            "mrr": round(mean_reciprocal_rank(retrieval_pairs), 4),
            "accuracy": round(avg("accuracy"), 4),
            "faithfulness": round(avg("faithfulness"), 4),
            "answer_relevancy": round(avg("answer_relevancy"), 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        }


# Singleton
_runner: EvaluationRunner | None = None


def get_evaluation_runner() -> EvaluationRunner:
    global _runner
    if _runner is None:
        _runner = EvaluationRunner()
    return _runner
