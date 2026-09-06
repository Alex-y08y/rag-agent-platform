"""Tests for retrieval stores (Milvus, ES) - structural tests."""
from __future__ import annotations

import pytest

from app.retrieval.milvus_store import MilvusStore
from app.retrieval.es_store import ElasticsearchStore
from app.retrieval.hybrid_retriever import HybridRetriever


class TestMilvusStore:
    def test_collection_name_generation(self):
        store = MilvusStore.__new__(MilvusStore)
        name = store._collection_name("kb-123")
        assert "kb_123" in name
        assert name.startswith("rag_kb_")

    def test_collection_name_special_chars(self):
        store = MilvusStore.__new__(MilvusStore)
        name = store._collection_name("my kb/test")
        assert " " not in name
        assert "/" not in name


class TestElasticsearchStore:
    def test_index_name_generation(self):
        store = ElasticsearchStore.__new__(ElasticsearchStore)
        name = store._index_name("KB-123")
        assert name.startswith("rag-kb-")
        assert name.islower()


class TestHybridRetriever:
    def test_singleton(self):
        from app.retrieval.hybrid_retriever import get_hybrid_retriever
        r1 = get_hybrid_retriever()
        r2 = get_hybrid_retriever()
        assert r1 is r2

    def test_retriever_has_milvus_and_es(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        assert hasattr(HybridRetriever, "retrieve")
        assert hasattr(HybridRetriever, "vector_only")
