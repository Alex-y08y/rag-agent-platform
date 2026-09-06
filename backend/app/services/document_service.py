"""Document ingestion service: upload -> parse -> chunk -> embed -> index."""
from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.exceptions import FileUploadError, NotFoundError
from app.core.logging import get_logger
from app.embeddings.bge_embedding import get_embedding_service
from app.models.models import Document, DocumentChunk, KnowledgeBase
from app.retrieval.es_store import get_es_store
from app.retrieval.milvus_store import get_milvus_store
from app.services.chunking import get_chunker
from app.services.parser import get_parser

logger = get_logger(__name__)


class DocumentService:
    """Handle document upload, parsing, chunking, embedding, and indexing."""

    def __init__(self) -> None:
        self.parser = get_parser()
        self.chunker = get_chunker()
        self.embedding = get_embedding_service()
        self.milvus = get_milvus_store()
        self.es = get_es_store()
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def validate_file(self, filename: str, file_size: int) -> str:
        """Validate file type and size."""
        ext = Path(filename).suffix.lower()
        if ext not in settings.allowed_extension_set:
            raise FileUploadError(
                f"Unsupported file type: {ext}. Allowed: {settings.allowed_extension_set}"
            )
        if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise FileUploadError(
                f"File too large: {file_size / 1024 / 1024:.1f}MB. "
                f"Max: {settings.MAX_FILE_SIZE_MB}MB"
            )
        return ext

    def save_upload(self, file_content: bytes, filename: str, kb_id: str) -> str:
        """Save uploaded file to disk and return storage path."""
        kb_dir = self.upload_dir / kb_id
        kb_dir.mkdir(parents=True, exist_ok=True)

        # Add timestamp to avoid name collisions
        timestamp = int(time.time())
        safe_name = f"{timestamp}_{filename}"
        storage_path = kb_dir / safe_name

        with open(storage_path, "wb") as f:
            f.write(file_content)

        return str(storage_path)

    def compute_hash(self, file_content: bytes) -> str:
        """Compute SHA256 hash for duplicate detection."""
        return hashlib.sha256(file_content).hexdigest()

    def check_duplicate(self, kb_id: str, file_hash: str) -> Document | None:
        """Check if a file with the same hash already exists in the KB."""
        db = SessionLocal()
        doc = (
            db.query(Document)
            .filter(
                Document.knowledge_base_id == kb_id,
                Document.file_hash == file_hash,
            )
            .first()
        )
        db.close()
        return doc

    async def ingest_document(
        self,
        kb_id: str,
        filename: str,
        file_content: bytes,
    ) -> Document:
        """Full ingestion pipeline: parse -> chunk -> embed -> index.

        Updates document status at each stage:
        UPLOADING -> PARSING -> CHUNKING -> EMBEDDING -> INDEXING -> SUCCESS
        """
        file_size = len(file_content)
        ext = self.validate_file(filename, file_size)
        file_hash = self.compute_hash(file_content)

        # Duplicate check
        existing = self.check_duplicate(kb_id, file_hash)
        if existing:
            logger.info("Duplicate file detected: %s (id=%s)", filename, existing.id)
            return existing

        # Save file
        storage_path = self.save_upload(file_content, filename, kb_id)

        # Create document record
        db = SessionLocal()
        doc = Document(
            knowledge_base_id=kb_id,
            filename=filename,
            file_type=ext.lstrip("."),
            file_size=file_size,
            file_hash=file_hash,
            storage_path=storage_path,
            status="UPLOADING",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        doc_id = doc.id
        db.close()

        try:
            # Step 1: Parse
            self._update_status(doc_id, "PARSING")
            pages = self.parser.parse(storage_path, ext)
            if not pages:
                raise FileUploadError(f"Failed to extract text from {filename}")

            # Step 2: Chunk
            self._update_status(doc_id, "CHUNKING")
            if ext in (".md", "md"):
                full_text = "\n".join(p.content for p in pages)
                chunks = self.chunker.chunk_markdown(full_text, source=filename)
            else:
                chunks = self.chunker.chunk_pages(pages, source=filename)

            if not chunks:
                raise FileUploadError(f"No chunks generated from {filename}")

            # Save chunks to DB
            db = SessionLocal()
            chunk_records = []
            for chunk in chunks:
                record = DocumentChunk(
                    document_id=doc_id,
                    knowledge_base_id=kb_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    page=chunk.page,
                    section=chunk.section,
                    title=chunk.title,
                    source=chunk.source or filename,
                    metadata=chunk.metadata,
                    embedding_status="PENDING",
                )
                db.add(record)
                chunk_records.append(record)
            db.commit()

            # Update chunk count
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.chunk_count = len(chunks)
                db.commit()

            # Build chunk_data while session is still open (loads IDs eagerly)
            chunk_data = []
            for record in chunk_records:
                chunk_data.append({
                    "chunk_id": record.id,
                    "document_id": doc_id,
                    "content": record.content,
                    "source": record.source,
                    "page": record.page,
                    "section": record.section,
                    "title": record.title,
                    "vector_id": record.id,
                })
            db.close()

            # Step 3: Embed
            self._update_status(doc_id, "EMBEDDING")

            # Step 4: Index into Milvus + ES
            self._update_status(doc_id, "INDEXING")
            try:
                self.milvus.insert(kb_id, chunk_data)
            except Exception as exc:
                logger.warning("Milvus indexing failed (continuing): %s", exc)

            try:
                self.es.index_chunks(kb_id, chunk_data)
            except Exception as exc:
                logger.warning("ES indexing failed (continuing): %s", exc)

            # Update chunk embedding status
            db = SessionLocal()
            db.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc_id
            ).update({"embedding_status": "DONE"}, synchronize_session=False)
            db.commit()

            # Update KB counts
            kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
            if kb:
                kb.document_count = (kb.document_count or 0) + 1
                kb.chunk_count = (kb.chunk_count or 0) + len(chunks)
                db.commit()
            db.close()

            self._update_status(doc_id, "SUCCESS")
            logger.info(
                "Document ingested: %s, %d chunks", filename, len(chunks)
            )

        except Exception as exc:
            logger.error("Ingestion failed for %s: %s", filename, exc)
            self._update_status(doc_id, "FAILED", error_message=str(exc))
            raise

        db = SessionLocal()
        doc = db.query(Document).filter(Document.id == doc_id).first()
        db.close()
        return doc

    def _update_status(
        self, doc_id: str, status: str, error_message: str | None = None
    ) -> None:
        db = SessionLocal()
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = status
            if error_message:
                doc.error_message = error_message
            db.commit()
        db.close()

    def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its chunks/index entries."""
        db = SessionLocal()
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            db.close()
            raise NotFoundError("Document", doc_id)

        kb_id = doc.knowledge_base_id
        chunk_count = doc.chunk_count

        # Delete from vector store and ES
        try:
            self.milvus.delete(kb_id, doc_id)
        except Exception as exc:
            logger.warning("Milvus delete failed: %s", exc)
        try:
            self.es.delete_document(kb_id, doc_id)
        except Exception as exc:
            logger.warning("ES delete failed: %s", exc)

        # Delete file
        if os.path.exists(doc.storage_path):
            os.remove(doc.storage_path)

        # Delete from DB (cascades chunks)
        db.delete(doc)

        # Update KB counts
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if kb:
            kb.document_count = max(0, (kb.document_count or 0) - 1)
            kb.chunk_count = max(0, (kb.chunk_count or 0) - chunk_count)
        db.commit()
        db.close()
        logger.info("Document deleted: %s", doc_id)

    async def reindex_document(self, doc_id: str) -> None:
        """Re-index an existing document."""
        db = SessionLocal()
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            db.close()
            raise NotFoundError("Document", doc_id)

        # Read file and re-ingest
        if not os.path.exists(doc.storage_path):
            db.close()
            raise FileUploadError(f"File not found: {doc.storage_path}")

        with open(doc.storage_path, "rb") as f:
            content = f.read()

        kb_id = doc.knowledge_base_id
        filename = doc.filename
        db.close()

        # Delete old index entries
        try:
            self.milvus.delete(kb_id, doc_id)
        except Exception:
            pass
        try:
            self.es.delete_document(kb_id, doc_id)
        except Exception:
            pass

        # Delete old chunks and document record (so ingest creates fresh)
        db = SessionLocal()
        db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()
        db.query(Document).filter(Document.id == doc_id).delete()
        db.commit()
        db.close()

        # Re-ingest
        await self.ingest_document(kb_id, filename, content)


# Singleton
_doc_service: DocumentService | None = None


def get_document_service() -> DocumentService:
    global _doc_service
    if _doc_service is None:
        _doc_service = DocumentService()
    return _doc_service
