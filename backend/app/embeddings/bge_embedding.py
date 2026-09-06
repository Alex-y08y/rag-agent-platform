"""BGE-M3 embedding service — free, open-source, local inference."""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _select_device(device_str: str) -> str:
    """Select compute device: auto -> cuda if available else cpu."""
    if device_str != "auto":
        return device_str
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class BGEEmbeddingService:
    """Local embedding using BAAI/bge-m3 via sentence-transformers.

    Supports:
    - CPU / CUDA auto-detection
    - Model caching
    - Batch embedding
    - Dynamic dimension detection (not hardcoded)
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = _select_device(device or settings.EMBEDDING_DEVICE)
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._model = None
        self._dimension: int | None = None
        self._cache_dir = settings.EMBEDDING_CACHE_DIR
        os.makedirs(self._cache_dir, exist_ok=True)

    def _load_model(self) -> Any:
        """Lazy-load the embedding model."""
        if self._model is not None:
            return self._model

        logger.info(
            "Loading embedding model %s on %s (batch=%d)",
            self.model_name, self.device, self.batch_size,
        )
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self._cache_dir,
            )
            # Detect dimension dynamically
            test_emb = self._model.encode(["test"], normalize_embeddings=True)
            self._dimension = test_emb.shape[1]
            logger.info("Embedding model loaded. Dimension=%d", self._dimension)
            return self._model
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise EmbeddingError(f"Failed to load embedding model: {exc}") from exc

    @property
    def dimension(self) -> int:
        """Get embedding dimension (loads model if needed)."""
        if self._dimension is None:
            self._load_model()
        return self._dimension or 1024  # bge-m3 default

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents.

        Returns list of normalized embedding vectors.
        """
        if not texts:
            return []

        model = self._load_model()
        try:
            start = time.time()
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            elapsed = (time.time() - start) * 1000
            logger.info(
                "Embedded %d documents in %.0fms (dim=%d)",
                len(texts), elapsed, embeddings.shape[1],
            )
            return embeddings.tolist()
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    def embed_documents_generator(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Alias for embed_documents (for LangChain compatibility)."""
        return self.embed_documents(texts)


# Singleton
_embedding_service: BGEEmbeddingService | None = None


def get_embedding_service() -> BGEEmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = BGEEmbeddingService()
    return _embedding_service
