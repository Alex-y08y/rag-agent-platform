"""BGE Reranker service — BAAI/bge-reranker-v2-m3, free and local."""
from __future__ import annotations

import time
from typing import Any

from app.core.config import settings
from app.core.exceptions import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _select_device(device_str: str) -> str:
    if device_str != "auto":
        return device_str
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class RerankerService:
    """Cross-encoder reranker using BGE-reranker-v2-m3.

    Re-ranks candidate documents against the query to improve precision.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL
        self.device = _select_device(device or settings.RERANKER_DEVICE)
        self._model = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        logger.info("Loading reranker model %s on %s", self.model_name, self.device)
        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                self.model_name,
                use_fp16=(self.device == "cuda"),
            )
            logger.info("Reranker model loaded")
            return self._model
        except Exception as exc:
            logger.error("Failed to load reranker: %s", exc)
            # Fallback: use sentence-transformers cross-encoder
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name, device=self.device
                )
                logger.info("Reranker loaded via CrossEncoder fallback")
                return self._model
            except Exception as exc2:
                raise AppError(f"Failed to load reranker: {exc2}") from exc2

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Re-rank documents by query relevance.

        Args:
            query: The search query.
            documents: List of dicts with at least 'content' key.
            top_n: Number of top results to return (default: settings.RERANKER_TOP_N).

        Returns:
            Documents sorted by reranker score descending, with 'rerank_score' added.
        """
        if not documents:
            return []

        top_n = top_n or settings.RERANKER_TOP_N
        if len(documents) <= top_n:
            # Still compute scores but return all
            pass

        model = self._load_model()
        start = time.time()

        pairs = [[query, doc.get("content", "")] for doc in documents]

        try:
            if hasattr(model, "compute_score"):
                # FlagEmbedding API
                scores = model.compute_score(pairs, normalize=True)
            else:
                # CrossEncoder API
                scores = model.predict(pairs)

            # Ensure scores is a list
            if not isinstance(scores, list):
                scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)

            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)

            # Sort by rerank score descending
            documents.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)

            elapsed = (time.time() - start) * 1000
            logger.info(
                "Reranked %d docs -> top %d in %.0fms",
                len(documents), min(top_n, len(documents)), elapsed,
            )

            return documents[:top_n]

        except Exception as exc:
            logger.error("Reranker failed: %s, returning original order", exc)
            for doc in documents:
                doc.setdefault("rerank_score", 0.0)
            return documents[:top_n]


# Singleton
_reranker_service: RerankerService | None = None


def get_reranker_service() -> RerankerService:
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
