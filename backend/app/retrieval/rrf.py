"""Reciprocal Rank Fusion (RRF) for combining retrieval results."""
from __future__ import annotations

from typing import Any

from app.core.config import settings


def rrf_fuse(
    result_lists: list[list[dict[str, Any]]],
    k: int | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    RRF(d) = Σ 1 / (k + rank(d))

    Args:
        result_lists: List of ranked result lists. Each result must have
            a 'chunk_id' key for deduplication.
        k: RRF constant (default: settings.RRF_K = 60).
        top_k: Number of final results to return.

    Returns:
        Fused and re-ranked list of documents with 'rrf_score'.
    """
    k = k or settings.RRF_K

    # Accumulate RRF scores by chunk_id
    scores: dict[str, float] = {}
    doc_map: dict[str, dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            cid = doc.get("chunk_id", "")
            if not cid:
                continue
            # RRF: 1 / (k + rank), rank is 1-based
            rrf_score = 1.0 / (k + rank + 1)
            scores[cid] = scores.get(cid, 0.0) + rrf_score

            # Keep the doc with the highest individual score
            if cid not in doc_map or doc.get("score", 0) > doc_map[cid].get("score", 0):
                doc_map[cid] = doc.copy()

    # Sort by RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused: list[dict[str, Any]] = []
    for cid in sorted_ids[:top_k]:
        doc = doc_map[cid].copy()
        doc["rrf_score"] = scores[cid]
        doc["score"] = scores[cid]  # Use RRF score as the combined score
        doc["retrieval_method"] = "hybrid_rrf"
        fused.append(doc)

    return fused


def weighted_fuse(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    vector_weight: float | None = None,
    bm25_weight: float | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Alternative: weighted score fusion (normalized).

    Normalizes scores from each retriever to [0,1] then combines with weights.
    """
    vector_weight = vector_weight or settings.HYBRID_WEIGHT_VECTOR
    bm25_weight = bm25_weight or settings.HYBRID_WEIGHT_BM25

    def _normalize(docs: list[dict[str, Any]]) -> None:
        if not docs:
            return
        max_score = max(d.get("score", 0) for d in docs)
        min_score = min(d.get("score", 0) for d in docs)
        range_ = max_score - min_score
        for d in docs:
            if range_ > 0:
                d["norm_score"] = (d["score"] - min_score) / range_
            else:
                d["norm_score"] = 1.0

    _normalize(vector_results)
    _normalize(bm25_results)

    combined: dict[str, dict[str, Any]] = {}

    for doc in vector_results:
        cid = doc["chunk_id"]
        combined[cid] = doc.copy()
        combined[cid]["fused_score"] = vector_weight * doc.get("norm_score", 0)

    for doc in bm25_results:
        cid = doc["chunk_id"]
        if cid in combined:
            combined[cid]["fused_score"] += bm25_weight * doc.get("norm_score", 0)
        else:
            combined[cid] = doc.copy()
            combined[cid]["fused_score"] = bm25_weight * doc.get("norm_score", 0)

    sorted_docs = sorted(
        combined.values(), key=lambda d: d.get("fused_score", 0), reverse=True
    )
    for doc in sorted_docs:
        doc["score"] = doc.pop("fused_score", 0)
        doc["retrieval_method"] = "hybrid_weighted"

    return sorted_docs[:top_k]
