"""RAG Pipeline: Query Rewrite -> Hybrid Search -> Rerank -> Compress -> Generate."""
from __future__ import annotations

import time
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.citation import get_citation_manager
from app.rag.context_compression import get_context_compressor
from app.rag.query_rewrite import get_query_rewriter
from app.reranker.bge_reranker import get_reranker_service
from app.retrieval.hybrid_retriever import get_hybrid_retriever
from app.schemas.schemas import Citation
from app.services.llm_service import get_llm_service

logger = get_logger(__name__)

RAG_SYSTEM_PROMPT = """你是一个企业级智能知识库助手。请根据提供的参考资料回答用户问题。

规则：
1. 只使用参考资料中的信息回答问题，不要编造信息
2. 在回答中使用 [数字] 标注引用来源，例如 [1] [2]
3. 如果参考资料中没有相关信息，明确说明"根据现有资料无法回答该问题"
4. 回答要准确、简洁、专业
5. 涉及数字、日期、政策条款时必须精确引用
6. 最后列出参考来源列表

参考资料格式：
[编号] 《文档名》 第X页 [章节]
内容...
"""


class RAGPipeline:
    """Full RAG pipeline with configurable retrieval strategy.

    Supports 4 versions:
    - baseline: Vector search only
    - hybrid: Vector + BM25 + RRF
    - hybrid_rerank: Hybrid + BGE Reranker
    - hybrid_rerank_rewrite: Hybrid + Reranker + Query Rewrite
    """

    def __init__(self) -> None:
        self.retriever = get_hybrid_retriever()
        self.reranker = get_reranker_service()
        self.rewriter = get_query_rewriter()
        self.compressor = get_context_compressor()
        self.citation_mgr = get_citation_manager()
        self.llm = get_llm_service()

    async def query(
        self,
        kb_id: str,
        question: str,
        rag_version: str = "hybrid_rerank_rewrite",
        top_k: int = 10,
        history: list[dict[str, str]] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute full RAG pipeline.

        Returns:
            {
                "answer": str,
                "citations": list[Citation],
                "retrieved_docs": list,
                "rewritten_query": str | None,
                "latency": {total, retrieval, reranker, llm},
                "token_usage": dict,
            }
        """
        timings: dict[str, int] = {}
        total_start = time.time()

        # Step 1: Query Rewrite (only for hybrid_rerank_rewrite)
        rewritten_query = question
        if rag_version == "hybrid_rerank_rewrite" and history:
            rewrite_start = time.time()
            rewritten_query = await self.rewriter.rewrite(question, history)
            timings["rewrite_ms"] = int((time.time() - rewrite_start) * 1000)

        # Step 2: Retrieval
        retrieval_start = time.time()
        use_bm25 = rag_version in ("hybrid", "hybrid_rerank", "hybrid_rerank_rewrite")
        use_vector = True  # All versions use vector

        docs, retrieval_meta = self.retriever.retrieve(
            kb_id=kb_id,
            query=rewritten_query,
            top_k=top_k,
            filters=filters,
            use_vector=use_vector,
            use_bm25=use_bm25,
        )
        timings["retrieval_ms"] = int((time.time() - retrieval_start) * 1000)

        # Step 3: Reranker (for hybrid_rerank and hybrid_rerank_rewrite)
        if rag_version in ("hybrid_rerank", "hybrid_rerank_rewrite") and docs:
            rerank_start = time.time()
            docs = self.reranker.rerank(rewritten_query, docs, top_n=min(5, len(docs)))
            timings["reranker_ms"] = int((time.time() - rerank_start) * 1000)

        # Step 4: Context Compression
        if docs:
            docs = self.compressor.compress(rewritten_query, docs)

        # Step 5: Build context and generate answer
        llm_start = time.time()
        context = self.citation_mgr.format_context(docs)

        if context:
            user_prompt = f"参考资料：\n{context}\n\n用户问题：{question}\n\n请回答："
            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            answer, token_usage = await self.llm.chat(messages)
        else:
            answer = "根据现有知识库资料，无法找到与该问题相关的信息。请尝试换一种问法，或确认知识库中是否已上传相关文档。"
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        timings["llm_ms"] = int((time.time() - llm_start) * 1000)
        timings["total_ms"] = int((time.time() - total_start) * 1000)

        citations = self.citation_mgr.build_citations(docs)

        logger.info(
            "RAG query complete: version=%s, docs=%d, citations=%d, total=%dms",
            rag_version, len(docs), len(citations), timings["total_ms"],
        )

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_docs": docs,
            "rewritten_query": rewritten_query if rewritten_query != question else None,
            "latency": timings,
            "token_usage": token_usage,
        }

    async def retrieve_only(
        self,
        kb_id: str,
        query: str,
        rag_version: str = "hybrid_rerank_rewrite",
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieval-only mode (for evaluation)."""
        rewritten = query
        if rag_version == "hybrid_rerank_rewrite":
            rewritten = await self.rewriter.rewrite(query, [])

        use_bm25 = rag_version != "baseline"
        docs, meta = self.retriever.retrieve(
            kb_id, rewritten, top_k=top_k, filters=filters,
            use_vector=True, use_bm25=use_bm25,
        )

        if rag_version in ("hybrid_rerank", "hybrid_rerank_rewrite") and docs:
            docs = self.reranker.rerank(rewritten, docs, top_n=min(top_k, len(docs)))

        return {
            "query": query,
            "rewritten_query": rewritten,
            "results": docs,
            "total": len(docs),
            "latency_ms": meta.get("total_latency_ms", 0),
        }


# Singleton
_rag_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
