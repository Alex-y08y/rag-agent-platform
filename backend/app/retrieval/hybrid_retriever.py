"""Hybrid Retriever: Vector (Milvus) + BM25 (Elasticsearch) -> RRF Fusion -> Top-K."""
from __future__ import annotations

import time
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.es_store import get_es_store
from app.retrieval.milvus_store import get_milvus_store
from app.retrieval.rrf import rrf_fuse

logger = get_logger(__name__)


class HybridRetriever:
    """Combines vector search and BM25 keyword search with RRF fusion.

    Pipeline:
        query -> Vector Search (Milvus) + BM25 Search (ES)
              -> RRF Fusion
              -> Top-K
    """

    def __init__(self) -> None:
        self.milvus = get_milvus_store()
        self.es = get_es_store()

    def retrieve(
        self,
        kb_id: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        use_vector: bool = True,
        use_bm25: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute hybrid retrieval.

        Returns:
            (fused_documents, metadata)
            metadata includes latency breakdown and per-method result counts.
        """
        metadata: dict[str, Any] = {
            "query": query,
            "vector_results": 0,
            "bm25_results": 0,
            "fused_results": 0,
            "vector_latency_ms": 0,
            "bm25_latency_ms": 0,
            "fusion_latency_ms": 0,
            "total_latency_ms": 0,
        }

        result_lists: list[list[dict[str, Any]]] = []
        total_start = time.time()

        # Vector search
        if use_vector:
            try:
                v_start = time.time()
                vector_docs = self.milvus.search(kb_id, query, top_k=top_k, filters=filters)
                metadata["vector_latency_ms"] = int((time.time() - v_start) * 1000)
                metadata["vector_results"] = len(vector_docs)
                result_lists.append(vector_docs)
            except Exception as exc:
                logger.warning("Vector search failed: %s", exc)
                metadata["vector_error"] = str(exc)

        # BM25 search
        if use_bm25:
            try:
                b_start = time.time()
                bm25_docs = self.es.search(kb_id, query, top_k=top_k, filters=filters)
                metadata["bm25_latency_ms"] = int((time.time() - b_start) * 1000)
                metadata["bm25_results"] = len(bm25_docs)
                result_lists.append(bm25_docs)
            except Exception as exc:
                logger.warning("BM25 search failed: %s", exc)
                metadata["bm25_error"] = str(exc)

        # RRF Fusion
        f_start = time.time()
        if len(result_lists) == 0:
            fused: list[dict[str, Any]] = []
        elif len(result_lists) == 1:
            fused = result_lists[0][:top_k]
        else:
            fused = rrf_fuse(result_lists, top_k=top_k)

        metadata["fusion_latency_ms"] = int((time.time() - f_start) * 1000)
        metadata["fused_results"] = len(fused)
        metadata["total_latency_ms"] = int((time.time() - total_start) * 1000)

        logger.info(
            "Hybrid retrieval: vector=%d, bm25=%d, fused=%d in %dms",
            metadata["vector_results"],
            metadata["bm25_results"],
            metadata["fused_results"],
            metadata["total_latency_ms"],
        )

        return fused, metadata

    def vector_only(
        self,
        kb_id: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Baseline: vector search only."""
        return self.retrieve(
            kb_id, query, top_k, filters, use_vector=True, use_bm25=False
        )


# Singleton
_hybrid_retriever: HybridRetriever | None = None


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever
