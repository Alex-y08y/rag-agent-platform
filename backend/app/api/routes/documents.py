"""Document API routes: upload, list, detail, delete, reindex."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import NotFoundError, to_http_error
from app.core.logging import get_logger, new_request_id
from app.models.models import Document, DocumentChunk
from app.schemas.schemas import ChunkOut, DocumentDetailOut, DocumentOut
from app.services.document_service import get_document_service

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    knowledge_base_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload and ingest a document into a knowledge base."""
    new_request_id()
    try:
        content = await file.read()
        service = get_document_service()
        doc = await service.ingest_document(
            kb_id=knowledge_base_id,
            filename=file.filename or "unknown",
            file_content=content,
        )
        return doc
    except Exception as exc:
        logger.error("Document upload failed: %s", exc)
        raise to_http_error(exc) if hasattr(exc, "code") else exc


@router.get("", response_model=list[DocumentOut])
def list_documents(
    knowledge_base_id: str | None = None,
    db: Session = Depends(get_db),
):
    """List documents, optionally filtered by knowledge base."""
    query = db.query(Document)
    if knowledge_base_id:
        query = query.filter(Document.knowledge_base_id == knowledge_base_id)
    return query.order_by(Document.created_at.desc()).all()


@router.get("/{doc_id}", response_model=DocumentDetailOut)
def get_document(doc_id: str, db: Session = Depends(get_db)):
    """Get document detail with chunks."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise NotFoundError("Document", doc_id)
    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == doc_id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )
    return DocumentDetailOut(
        id=doc.id,
        knowledge_base_id=doc.knowledge_base_id,
        filename=doc.filename,
        file_type=doc.file_type,
        file_size=doc.file_size,
        status=doc.status,
        chunk_count=doc.chunk_count,
        created_at=doc.created_at,
        error_message=doc.error_message,
        chunks=[ChunkOut.model_validate(c) for c in chunks],
    )


@router.delete("/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """Delete a document and all its data."""
    new_request_id()
    try:
        service = get_document_service()
        service.delete_document(doc_id)
        return {"success": True, "message": f"Document {doc_id} deleted"}
    except Exception as exc:
        logger.error("Document delete failed: %s", exc, exc_info=True)
        if hasattr(exc, "code") and hasattr(exc, "message"):
            raise to_http_error(exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{doc_id}/reindex")
async def reindex_document(doc_id: str, db: Session = Depends(get_db)):
    """Re-index a document."""
    new_request_id()
    try:
        service = get_document_service()
        await service.reindex_document(doc_id)
        return {"success": True, "message": f"Document {doc_id} reindexed"}
    except Exception as exc:
        logger.error("Document reindex failed: %s", exc, exc_info=True)
        if hasattr(exc, "code") and hasattr(exc, "message"):
            raise to_http_error(exc)
        raise HTTPException(status_code=500, detail=str(exc))
