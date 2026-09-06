"""Elasticsearch BM25 full-text search store."""
from __future__ import annotations

import time
from typing import Any

from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger

logger = get_logger(__name__)


class ElasticsearchStore:
    """BM25 keyword search using Elasticsearch.

    Excels at:
    - Professional terminology
    - Numbers and IDs
    - Policy names, product model numbers
    - Exact phrase matching
    """

    def __init__(self) -> None:
        self.url = settings.ELASTICSEARCH_URL
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch

                self._client = Elasticsearch(self.url)
                if not self._client.ping():
                    logger.warning("Elasticsearch ping failed at %s", self.url)
                else:
                    logger.info("Connected to Elasticsearch at %s", self.url)
            except Exception as exc:
                logger.error("Failed to connect to Elasticsearch: %s", exc)
                raise RetrievalError(f"ES connection failed: {exc}") from exc
        return self._client

    def _index_name(self, kb_id: str) -> str:
        import re
        safe = re.sub(r"[^a-z0-9_]", "_", kb_id.lower())
        return f"{settings.ELASTICSEARCH_INDEX_PREFIX}{safe}"

    def create_index(self, kb_id: str) -> None:
        """Create an ES index with BM25 similarity and Chinese-friendly analyzer."""
        from elasticsearch import Elasticsearch

        client = self._get_client()
        name = self._index_name(kb_id)

        if client.indices.exists(index=name):
            return

        mappings = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "knowledge_base_id": {"type": "keyword"},
                    "content": {"type": "text", "analyzer": "cjk"},
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "section": {"type": "text", "analyzer": "cjk"},
                    "title": {"type": "text", "analyzer": "cjk"},
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "similarity": {"default": {"type": "BM25"}},
            },
        }
        client.indices.create(index=name, body=mappings)
        logger.info("Created ES index %s", name)

    def index_chunks(self, kb_id: str, chunks: list[dict[str, Any]]) -> int:
        """Index chunks into Elasticsearch."""
        from elasticsearch import Elasticsearch
        from elasticsearch.helpers import bulk

        client = self._get_client()
        self.create_index(kb_id)
        name = self._index_name(kb_id)

        actions = []
        for c in chunks:
            actions.append({
                "_index": name,
                "_id": c.get("vector_id") or c["chunk_id"],
                "_source": {
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "knowledge_base_id": kb_id,
                    "content": c["content"],
                    "source": c.get("source", ""),
                    "page": c.get("page"),
                    "section": c.get("section", ""),
                    "title": c.get("title", ""),
                },
            })

        if actions:
            success, _ = bulk(client, actions, refresh=True)
            logger.info("Indexed %d chunks into ES %s", success, name)
            return success
        return 0

    def search(
        self,
        kb_id: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 search with optional filters."""
        from elasticsearch import Elasticsearch

        client = self._get_client()
        name = self._index_name(kb_id)

        if not client.indices.exists(index=name):
            logger.warning("ES index %s does not exist", name)
            return []

        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["content^3", "title^2", "section"],
                    "type": "best_fields",
                }
            }
        ]

        filter_clauses = []
        if filters:
            for key, value in filters.items():
                filter_clauses.append({"term": {key: value}})

        body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses,
                }
            },
            "size": top_k,
            "_source": ["chunk_id", "document_id", "content", "source", "page", "section", "title"],
        }

        start = time.time()
        resp = client.search(index=name, body=body)
        elapsed = (time.time() - start) * 1000

        docs: list[dict[str, Any]] = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            docs.append({
                "chunk_id": src.get("chunk_id", ""),
                "document_id": src.get("document_id", ""),
                "content": src.get("content", ""),
                "source": src.get("source", ""),
                "page": src.get("page"),
                "section": src.get("section"),
                "title": src.get("title"),
                "score": float(hit["_score"]),
                "retrieval_method": "bm25",
            })

        logger.info("ES BM25 search: %d results in %.0fms", len(docs), elapsed)
        return docs

    def delete_document(self, kb_id: str, document_id: str) -> None:
        """Delete all ES docs for a document."""
        from elasticsearch import Elasticsearch

        client = self._get_client()
        name = self._index_name(kb_id)
        if not client.indices.exists(index=name):
            return
        client.delete_by_query(
            index=name,
            body={"query": {"term": {"document_id": document_id}}},
            refresh=True,
        )
        logger.info("Deleted ES docs for document %s", document_id)

    def delete_index(self, kb_id: str) -> None:
        """Delete an ES index."""
        from elasticsearch import Elasticsearch

        client = self._get_client()
        name = self._index_name(kb_id)
        if client.indices.exists(index=name):
            client.indices.delete(index=name)
            logger.info("Deleted ES index %s", name)


# Singleton
_es_store: ElasticsearchStore | None = None


def get_es_store() -> ElasticsearchStore:
    global _es_store
    if _es_store is None:
        _es_store = ElasticsearchStore()
    return _es_store
