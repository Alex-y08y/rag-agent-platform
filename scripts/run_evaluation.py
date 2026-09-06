"""Run RAG evaluation from command line."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.database import SessionLocal
from app.models.models import EvaluationTask, KnowledgeBase
from app.evaluation.runner import get_evaluation_runner


async def main():
    db = SessionLocal()
    kb = db.query(KnowledgeBase).first()
    if not kb:
        print("No knowledge base found. Run seed_demo_data.py first.")
        db.close()
        return

    import uuid
    task_id = uuid.uuid4().hex[:16]
    task = EvaluationTask(
        id=task_id,
        name="CLI Evaluation",
        dataset_name="rag_eval",
        rag_version="hybrid_rerank_rewrite",
        status="PENDING",
        total_samples=0,
        completed_samples=0,
    )
    db.add(task)
    db.commit()
    db.close()

    print(f"Starting evaluation: task={task_id}, kb={kb.id}")
    runner = get_evaluation_runner()
    metrics = await runner.run_evaluation(
        task_id=task_id,
        kb_id=kb.id,
        dataset_name="rag_eval",
        rag_version="hybrid_rerank_rewrite",
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key:25s}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
