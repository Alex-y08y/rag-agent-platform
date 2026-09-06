"""Retrieval API route for direct search testing."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.exceptions import to_http_error
from app.core.logging import get_logger, new_request_id
from app.rag.pipeline import get_rag_pipeline
from app.schemas.schemas import RetrievalRequest, RetrievalResponse, RetrievedDoc

logger = get_logger(__name__)
router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=RetrievalResponse)
async def search(request: RetrievalRequest):
    """Execute retrieval (without generation) for testing and debugging."""
    new_request_id()
    try:
        pipeline = get_rag_pipeline()
        kb_id = request.knowledge_base_id
        if not kb_id:
            return RetrievalResponse(
                query=request.query, results=[], total=0, latency_ms=0
            )

        result = await pipeline.retrieve_only(
            kb_id=kb_id,
            query=request.query,
            rag_version=request.rag_version,
            top_k=request.top_k,
            filters=request.filters or None,
        )

        docs = [
            RetrievedDoc(
                chunk_id=d.get("chunk_id", ""),
                document_id=d.get("document_id", ""),
                content=d.get("content", ""),
                score=d.get("score", 0),
                source=d.get("source"),
                page=d.get("page"),
                section=d.get("section"),
                title=d.get("title"),
                metadata=d.get("metadata", {}),
            )
            for d in result.get("results", [])
        ]

        return RetrievalResponse(
            query=request.query,
            rewritten_query=result.get("rewritten_query"),
            results=docs,
            total=len(docs),
            latency_ms=result.get("latency_ms", 0),
        )
    except Exception as exc:
        logger.error("Retrieval search failed: %s", exc)
        raise to_http_error(exc) if hasattr(exc, "code") else exc
