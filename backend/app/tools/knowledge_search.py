"""Knowledge Search Tool: hybrid RAG retrieval over knowledge bases."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.retrieval.hybrid_retriever import get_hybrid_retriever
from app.reranker.bge_reranker import get_reranker_service
from app.tools.base import BaseTool

logger = get_logger(__name__)


class KnowledgeSearchInput(BaseModel):
    query: str = Field(..., description="搜索查询")
    knowledge_base_id: str | None = Field(None, description="知识库ID")
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] = Field(default_factory=dict)


class KnowledgeSearchTool(BaseTool[KnowledgeSearchInput, dict[str, Any]]):
    """Search the enterprise knowledge base using hybrid retrieval."""

    name = "knowledge_search"
    description = (
        "搜索企业知识库，返回相关文档片段。支持混合检索（向量+BM25）和重排序。"
        "当用户询问公司政策、制度、产品文档、流程规范等知识时使用此工具。"
    )
    input_schema = KnowledgeSearchInput

    async def execute(self, input_data: KnowledgeSearchInput) -> dict[str, Any]:
        kb_id = input_data.knowledge_base_id
        if not kb_id:
            return {"documents": [], "error": "No knowledge base specified"}

        try:
            retriever = get_hybrid_retriever()
            docs, meta = retriever.retrieve(
                kb_id=kb_id,
                query=input_data.query,
                top_k=input_data.top_k,
                filters=input_data.filters or None,
            )

            # Rerank
            if docs:
                reranker = get_reranker_service()
                docs = reranker.rerank(input_data.query, docs, top_n=input_data.top_k)

            # Format output
            formatted = []
            for doc in docs:
                formatted.append({
                    "content": doc.get("content", ""),
                    "source": doc.get("source", ""),
                    "page": doc.get("page"),
                    "section": doc.get("section"),
                    "title": doc.get("title"),
                    "score": doc.get("score", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                })

            logger.info(
                "KnowledgeSearch: query='%s', found=%d",
                input_data.query[:50], len(formatted),
            )
            return {"documents": formatted, "total": len(formatted), "meta": meta}

        except Exception as exc:
            logger.error("KnowledgeSearch failed: %s", exc)
            return {"documents": [], "error": str(exc)}
