"""In-memory vector store for LITE_MODE (no Milvus required).

Uses numpy cosine similarity. Thread-safe for basic usage.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LiteVectorStore:
    """In-memory vector store using numpy cosine similarity.

    Drop-in replacement for MilvusStore when LITE_MODE=true or Milvus unavailable.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._collections: dict[str, dict[str, dict[str, Any]]] = {}
        logger.info("LiteVectorStore initialized (in-memory, numpy cosine)")

    def create_collection(self, kb_id: str, dimension: int) -> None:
        with self._lock:
            if kb_id not in self._collections:
                self._collections[kb_id] = {}
                logger.info("Lite collection created: %s (dim=%d)", kb_id, dimension)

    def insert(
        self,
        kb_id: str,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        metadatas = metadatas or [{} for _ in chunk_ids]
        with self._lock:
            coll = self._collections.setdefault(kb_id, {})
            for cid, emb, doc, meta in zip(chunk_ids, embeddings, documents, metadatas):
                coll[cid] = {
                    "embedding": np.array(emb, dtype=np.float32),
                    "content": doc,
                    "metadata": meta,
                }
        logger.info("Lite insert: %d chunks into %s", len(chunk_ids), kb_id)

    def search(
        self,
        kb_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            coll = self._collections.get(kb_id, {})
            if not coll:
                return []

            query_vec = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return []

            results = []
            for cid, data in coll.items():
                # Metadata filter
                if filters:
                    skip = False
                    for k, v in filters.items():
                        if data["metadata"].get(k) != v:
                            skip = True
                            break
                    if skip:
                        continue

                emb = data["embedding"]
                emb_norm = np.linalg.norm(emb)
                if emb_norm == 0:
                    continue
                score = float(np.dot(query_vec, emb) / (query_norm * emb_norm))
                results.append({
                    "chunk_id": cid,
                    "content": data["content"],
                    "score": score,
                    "metadata": data["metadata"],
                    "source": data["metadata"].get("source", ""),
                    "page": data["metadata"].get("page"),
                    "section": data["metadata"].get("section"),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete(self, kb_id: str, chunk_ids: list[str] | None = None) -> None:
        with self._lock:
            if chunk_ids is None:
                self._collections.pop(kb_id, None)
            else:
                coll = self._collections.get(kb_id, {})
                for cid in chunk_ids:
                    coll.pop(cid, None)

    def upsert(self, kb_id: str, chunk_id: str, embedding: list[float],
               document: str, metadata: dict[str, Any] | None = None) -> None:
        self.insert(kb_id, [chunk_id], [embedding], [document], [metadata or {}])

    def count(self, kb_id: str) -> int:
        with self._lock:
            return len(self._collections.get(kb_id, {}))


# Singleton
_lite_vector_store: LiteVectorStore | None = None


def get_lite_vector_store() -> LiteVectorStore:
    global _lite_vector_store
    if _lite_vector_store is None:
        _lite_vector_store = LiteVectorStore()
    return _lite_vector_store
