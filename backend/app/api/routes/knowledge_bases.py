"""Knowledge Base API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger, new_request_id
from app.models.models import KnowledgeBase
from app.retrieval.es_store import get_es_store
from app.retrieval.milvus_store import get_milvus_store
from app.schemas.schemas import KnowledgeBaseCreate, KnowledgeBaseOut

logger = get_logger(__name__)
router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


@router.get("", response_model=list[KnowledgeBaseOut])
def list_knowledge_bases(db: Session = Depends(get_db)):
    """List all knowledge bases."""
    return db.query(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()).all()


@router.post("", response_model=KnowledgeBaseOut)
def create_knowledge_base(
    request: KnowledgeBaseCreate, db: Session = Depends(get_db)
):
    """Create a new knowledge base."""
    new_request_id()
    kb = KnowledgeBase(
        name=request.name,
        description=request.description,
        status="active",
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)

    # Initialize vector collection and ES index
    try:
        get_milvus_store().create_collection(kb.id)
    except Exception as exc:
        logger.warning("Milvus collection init failed: %s", exc)
    try:
        get_es_store().create_index(kb.id)
    except Exception as exc:
        logger.warning("ES index init failed: %s", exc)

    logger.info("Knowledge base created: %s (%s)", kb.name, kb.id)
    return kb


@router.delete("/{kb_id}")
def delete_knowledge_base(kb_id: str, db: Session = Depends(get_db)):
    """Delete a knowledge base and all associated data."""
    new_request_id()
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise NotFoundError("KnowledgeBase", kb_id)

    # Delete vector collection and ES index
    try:
        get_milvus_store().delete_collection(kb_id)
    except Exception as exc:
        logger.warning("Milvus collection delete failed: %s", exc)
    try:
        get_es_store().delete_index(kb_id)
    except Exception as exc:
        logger.warning("ES index delete failed: %s", exc)

    db.delete(kb)
    db.commit()
    logger.info("Knowledge base deleted: %s", kb_id)
    return {"success": True, "message": f"Knowledge base {kb_id} deleted"}
