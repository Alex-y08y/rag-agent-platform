"""Milvus vector store with dynamic collection creation and metadata filtering."""
from __future__ import annotations

import time
from typing import Any

from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.embeddings.bge_embedding import get_embedding_service

logger = get_logger(__name__)


class MilvusStore:
    """Vector store backed by Milvus.

    Collections are created per knowledge base with dynamic dimension
    detected from the embedding model.
    """

    def __init__(self) -> None:
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self._client = None
        self._collections: dict[str, Any] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from pymilvus import connections, utility

                connections.connect(
                    alias="default", host=self.host, port=self.port
                )
                self._client = utility
                logger.info("Connected to Milvus at %s:%d", self.host, self.port)
            except Exception as exc:
                logger.error("Failed to connect to Milvus: %s", exc)
                raise RetrievalError(f"Milvus connection failed: {exc}") from exc
        return self._client

    def _collection_name(self, kb_id: str) -> str:
        import re
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", kb_id)
        return f"{settings.MILVUS_COLLECTION_PREFIX}{safe}"

    def create_collection(self, kb_id: str, dimension: int | None = None) -> Any:
        """Create a Milvus collection for a knowledge base."""
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        name = self._collection_name(kb_id)
        self._get_client()

        if utility.has_collection(name):
            logger.info("Collection %s already exists", name)
            coll = Collection(name)
            coll.load()
            return coll

        dim = dimension or get_embedding_service().dimension
        logger.info("Creating Milvus collection %s (dim=%d)", name, dim)

        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="knowledge_base_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields=fields, description=f"RAG KB: {kb_id}")
        coll = Collection(name=name, schema=schema)

        # Create IVF_FLAT index
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        }
        coll.create_index(field_name="embedding", index_params=index_params)
        coll.load()
        logger.info("Collection %s created and indexed", name)
        return coll

    def insert(self, kb_id: str, chunks: list[dict[str, Any]]) -> int:
        """Insert chunks with embeddings into Milvus."""
        from pymilvus import Collection

        coll = self.create_collection(kb_id)
        if not chunks:
            return 0

        # Embed all chunk contents
        texts = [c["content"] for c in chunks]
        embeddings = get_embedding_service().embed_documents(texts)

        data = [
            [c.get("vector_id") or c["chunk_id"] for c in chunks],
            [c["chunk_id"] for c in chunks],
            [c["document_id"] for c in chunks],
            [kb_id for _ in chunks],
            [c["content"] for c in chunks],
            [c.get("source", "") for c in chunks],
            [c.get("page", 0) or 0 for c in chunks],
            [c.get("section", "") or "" for c in chunks],
            [c.get("title", "") or "" for c in chunks],
            embeddings,
        ]

        result = coll.insert(data)
        coll.flush()
        logger.info("Inserted %d vectors into %s", len(chunks), self._collection_name(kb_id))
        return result.insert_count if hasattr(result, "insert_count") else len(chunks)

    def search(
        self,
        kb_id: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search with optional metadata filtering."""
        from pymilvus import Collection, utility

        name = self._collection_name(kb_id)
        self._get_client()

        if not utility.has_collection(name):
            logger.warning("Collection %s does not exist", name)
            return []

        coll = Collection(name)
        coll.load()

        query_emb = get_embedding_service().embed_query(query)

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16},
        }

        # Build filter expression
        expr = ""
        if filters:
            parts = []
            for key, value in filters.items():
                if isinstance(value, str):
                    parts.append(f'{key} == "{value}"')
                else:
                    parts.append(f"{key} == {value}")
            expr = " and ".join(parts)

        start = time.time()
        results = coll.search(
            data=[query_emb],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr or None,
            output_fields=["chunk_id", "document_id", "content", "source", "page", "section", "title"],
        )
        elapsed = (time.time() - start) * 1000

        docs: list[dict[str, Any]] = []
        for hits in results:
            for hit in hits:
                entity = hit.entity
                def _get(field: str, default: Any = None) -> Any:
                    try:
                        val = entity.get(field)
                        return val if val is not None else default
                    except Exception:
                        return default
                docs.append({
                    "chunk_id": _get("chunk_id", ""),
                    "document_id": _get("document_id", ""),
                    "content": _get("content", ""),
                    "source": _get("source", ""),
                    "page": _get("page"),
                    "section": _get("section"),
                    "title": _get("title"),
                    "score": float(hit.score),
                    "retrieval_method": "vector",
                })

        logger.info("Milvus search: %d results in %.0fms", len(docs), elapsed)
        return docs

    def delete(self, kb_id: str, document_id: str) -> None:
        """Delete all vectors for a document."""
        from pymilvus import Collection, utility

        name = self._collection_name(kb_id)
        self._get_client()
        if not utility.has_collection(name):
            return

        coll = Collection(name)
        coll.delete(expr=f'document_id == "{document_id}"')
        coll.flush()
        logger.info("Deleted vectors for document %s", document_id)

    def delete_collection(self, kb_id: str) -> None:
        """Drop an entire collection."""
        from pymilvus import utility

        name = self._collection_name(kb_id)
        self._get_client()
        if utility.has_collection(name):
            utility.drop_collection(name)
            logger.info("Dropped collection %s", name)


# Singleton
_milvus_store: MilvusStore | None = None


def get_milvus_store() -> MilvusStore:
    global _milvus_store
    if _milvus_store is None:
        _milvus_store = MilvusStore()
    return _milvus_store
