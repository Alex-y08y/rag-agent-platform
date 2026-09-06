"""RAG Evaluation metrics: Recall@K, Precision@K, MRR."""
from __future__ import annotations

from typing import Any


def recall_at_k(
    expected_sources: list[str],
    retrieved_sources: list[str],
    k: int,
) -> float:
    """Recall@K = relevant docs retrieved in top-K / total relevant docs.

    Args:
        expected_sources: Ground truth source identifiers (filenames).
        retrieved_sources: Retrieved source identifiers in rank order.
        k: Cutoff rank.

    Returns:
        Recall score in [0, 1].
    """
    if not expected_sources:
        return 1.0  # No expected sources -> vacuously true
    if not retrieved_sources:
        return 0.0

    top_k = retrieved_sources[:k]
    relevant_retrieved = sum(
        1 for src in expected_sources if any(_match_source(src, r) for r in top_k)
    )
    return relevant_retrieved / len(expected_sources)


def precision_at_k(
    expected_sources: list[str],
    retrieved_sources: list[str],
    k: int,
) -> float:
    """Precision@K = relevant docs in top-K / K.

    Args:
        expected_sources: Ground truth source identifiers.
        retrieved_sources: Retrieved source identifiers in rank order.
        k: Cutoff rank.

    Returns:
        Precision score in [0, 1].
    """
    if not retrieved_sources or k == 0:
        return 0.0

    top_k = retrieved_sources[:k]
    if not expected_sources:
        return 0.0

    relevant_in_topk = sum(
        1 for r in top_k if any(_match_source(src, r) for src in expected_sources)
    )
    return relevant_in_topk / k


def reciprocal_rank(
    expected_sources: list[str],
    retrieved_sources: list[str],
) -> float:
    """Reciprocal Rank of the first relevant document.

    RR = 1 / rank_of_first_relevant (1-based).
    Returns 0 if no relevant document found.
    """
    if not expected_sources or not retrieved_sources:
        return 0.0

    for rank, src in enumerate(retrieved_sources, 1):
        if any(_match_source(exp, src) for exp in expected_sources):
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    samples: list[tuple[list[str], list[str]]],
) -> float:
    """Mean Reciprocal Rank across multiple samples.

    Args:
        samples: List of (expected_sources, retrieved_sources) tuples.

    Returns:
        MRR score in [0, 1].
    """
    if not samples:
        return 0.0
    rrs = [reciprocal_rank(exp, ret) for exp, ret in samples]
    return sum(rrs) / len(rrs)


def batch_recall_at_k(
    samples: list[tuple[list[str], list[str]]],
    k: int,
) -> float:
    """Average Recall@K across samples."""
    if not samples:
        return 0.0
    scores = [recall_at_k(exp, ret, k) for exp, ret in samples]
    return sum(scores) / len(scores)


def batch_precision_at_k(
    samples: list[tuple[list[str], list[str]]],
    k: int,
) -> float:
    """Average Precision@K across samples."""
    if not samples:
        return 0.0
    scores = [precision_at_k(exp, ret, k) for exp, ret in samples]
    return sum(scores) / len(scores)


def _match_source(expected: str, retrieved: str) -> bool:
    """Check if a retrieved source matches an expected source.

    Matches by filename (case-insensitive, ignoring path).
    """
    expected_norm = expected.lower().split("/")[-1].split("\\")[-1]
    retrieved_norm = retrieved.lower().split("/")[-1].split("\\")[-1]
    return expected_norm == retrieved_norm or expected_norm in retrieved_norm or retrieved_norm in expected_norm


def analyze_retrieval(
    expected_sources: list[str],
    retrieved_sources: list[str],
    recall_5: float,
    accuracy: float,
) -> str:
    """Generate human-readable analysis of retrieval vs generation quality.

    Distinguishes:
    - Retrieval Problem: correct docs not recalled
    - Generation Problem: correct docs recalled but answer wrong
    - Both / Neither
    """
    if recall_5 < 1.0 and expected_sources:
        missing = [
            s for s in expected_sources
            if not any(_match_source(s, r) for r in retrieved_sources[:5])
        ]
        return (
            f"Retrieval Problem: 正确文档未被召回。"
            f"期望来源: {expected_sources}, 实际召回: {retrieved_sources[:5]}. "
            f"缺失: {missing}. 需要优化检索策略（Hybrid Search / Reranker / Query Rewrite）。"
        )
    elif recall_5 >= 1.0 and accuracy < 0.6:
        return (
            f"Generation Problem: 正确文档已召回（Recall@5=1.0），但生成答案不准确。"
            f"可能原因：LLM未充分利用上下文、上下文压缩丢失关键信息、或Prompt不够精确。"
            f"建议优化Context Compression或System Prompt。"
        )
    elif recall_5 >= 1.0 and accuracy >= 0.6:
        return "检索和生成都表现良好。"
    else:
        return "检索和生成均有改进空间。"
