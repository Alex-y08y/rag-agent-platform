"""Evaluation API routes: run evaluation, list tasks, get results."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import NotFoundError, to_http_error
from app.core.logging import get_logger, new_request_id
from app.evaluation.runner import get_evaluation_runner
from app.models.models import EvaluationSample, EvaluationTask, KnowledgeBase
from app.schemas.schemas import (
    EvaluationRunRequest,
    EvaluationSampleOut,
    EvaluationTaskOut,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationTaskOut)
async def run_evaluation(
    request: EvaluationRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Start a new RAG evaluation task (runs in background)."""
    new_request_id()
    try:
        # Find first knowledge base (or use specified)
        kb = db.query(KnowledgeBase).first()
        if not kb:
            return EvaluationTaskOut(
                id="error", name="No KB", dataset_name=request.dataset_name,
                rag_version=request.rag_version, status="FAILED",
                total_samples=0, completed_samples=0, metrics={},
                created_at=__import__("datetime").datetime.utcnow(),
            )

        task = EvaluationTask(
            id=uuid.uuid4().hex[:16],
            name=request.name or f"Eval-{request.rag_version}-{request.dataset_name}",
            dataset_name=request.dataset_name,
            rag_version=request.rag_version,
            status="PENDING",
            total_samples=0,
            completed_samples=0,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = task.id
        kb_id = kb.id

        # Run in background
        async def _run():
            runner = get_evaluation_runner()
            await runner.run_evaluation(
                task_id=task_id,
                kb_id=kb_id,
                dataset_name=request.dataset_name,
                rag_version=request.rag_version,
                max_samples=request.max_samples,
            )

        background_tasks.add_task(_run)
        logger.info("Evaluation task started: %s", task_id)
        return task

    except Exception as exc:
        logger.error("Evaluation run failed: %s", exc)
        raise to_http_error(exc) if hasattr(exc, "code") else exc


@router.get("/tasks", response_model=list[EvaluationTaskOut])
def list_evaluation_tasks(db: Session = Depends(get_db)):
    """List all evaluation tasks."""
    return (
        db.query(EvaluationTask)
        .order_by(EvaluationTask.created_at.desc())
        .all()
    )


@router.get("/tasks/{task_id}", response_model=EvaluationTaskOut)
def get_evaluation_task(task_id: str, db: Session = Depends(get_db)):
    """Get evaluation task details."""
    task = db.query(EvaluationTask).filter(EvaluationTask.id == task_id).first()
    if not task:
        raise NotFoundError("EvaluationTask", task_id)
    return task


@router.get("/results", response_model=list[EvaluationSampleOut])
def get_evaluation_results(
    task_id: str | None = None,
    db: Session = Depends(get_db),
):
    """Get evaluation sample results, optionally filtered by task."""
    query = db.query(EvaluationSample)
    if task_id:
        query = query.filter(EvaluationSample.task_id == task_id)
    return query.order_by(EvaluationSample.created_at.desc()).limit(200).all()
