"""Document Analysis Tool: analyze uploaded documents."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.tools.base import BaseTool

logger = get_logger(__name__)


class DocumentAnalysisInput(BaseModel):
    document_id: str = Field(..., description="文档ID")
    analysis_type: str = Field(
        default="summary",
        description="分析类型：summary(摘要), keywords(关键词), structure(结构)",
    )


class DocumentAnalysisTool(BaseTool[DocumentAnalysisInput, dict[str, Any]]):
    """Analyze documents in the knowledge base."""

    name = "document_analysis"
    description = (
        "分析知识库中的文档，生成摘要、提取关键词或分析文档结构。"
        "当用户需要了解某份文档的内容概要、关键信息或结构时使用。"
    )
    input_schema = DocumentAnalysisInput

    async def execute(self, input_data: DocumentAnalysisInput) -> dict[str, Any]:
        from app.models.models import Document, DocumentChunk
        from app.core.database import SessionLocal

        try:
            db = SessionLocal()
            doc = db.query(Document).filter(Document.id == input_data.document_id).first()
            if not doc:
                db.close()
                return {"error": f"Document {input_data.document_id} not found"}

            chunks = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.document_id == doc.id)
                .order_by(DocumentChunk.chunk_index)
                .all()
            )
            db.close()

            full_text = "\n".join(c.content for c in chunks[:10])  # First 10 chunks

            if input_data.analysis_type == "summary":
                # Use LLM to summarize
                from app.services.llm_service import get_llm_service
                llm = get_llm_service()
                messages = [
                    {"role": "system", "content": "你是一个文档分析助手。请用简洁的中文总结以下文档内容，不超过200字。"},
                    {"role": "user", "content": full_text[:3000]},
                ]
                summary, _ = await llm.chat(messages)
                return {
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "analysis_type": "summary",
                    "summary": summary,
                    "chunk_count": len(chunks),
                }
            elif input_data.analysis_type == "keywords":
                # Simple keyword extraction
                words = full_text.replace("\n", " ").split()
                from collections import Counter
                # Filter meaningful words (length > 1, not common stop words)
                stop = {"的", "了", "是", "在", "和", "与", "或", "等", "及", "对", "为", "以", "有"}
                filtered = [w for w in words if len(w) > 1 and w not in stop]
                keywords = [w for w, _ in Counter(filtered).most_common(10)]
                return {
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "analysis_type": "keywords",
                    "keywords": keywords,
                    "chunk_count": len(chunks),
                }
            else:  # structure
                sections = list({c.section for c in chunks if c.section})
                return {
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "analysis_type": "structure",
                    "sections": sections,
                    "chunk_count": len(chunks),
                    "pages": sorted({c.page for c in chunks if c.page}),
                }

        except Exception as exc:
            logger.error("Document analysis failed: %s", exc)
            return {"error": str(exc)}
