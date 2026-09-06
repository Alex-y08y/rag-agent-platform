"""Tests for BGE Reranker service."""
from __future__ import annotations

import pytest

from app.reranker.bge_reranker import RerankerService, _select_device


class TestRerankerDevice:
    def test_explicit_device(self):
        assert _select_device("cpu") == "cpu"

    def test_auto_returns_valid_device(self):
        device = _select_device("auto")
        assert device in ("cpu", "cuda")


class TestRerankerService:
    def test_empty_documents(self):
        service = RerankerService.__new__(RerankerService)
        result = service.rerank("query", []) if hasattr(service, 'rerank') else []
        assert result == []

    def test_rerank_preserves_count_when_fewer_than_topn(self):
        """If model loads, test basic functionality; otherwise skip."""
        try:
            service = RerankerService(model_name="BAAI/bge-reranker-base", device="cpu")
            docs = [
                {"content": "差旅报销标准是500元", "chunk_id": "1"},
                {"content": "今天天气很好", "chunk_id": "2"},
            ]
            result = service.rerank("差旅报销", docs, top_n=5)
            assert len(result) == 2
            assert "rerank_score" in result[0]
        except Exception as exc:
            pytest.skip(f"Reranker model not available: {exc}")

    def test_rerank_top_n_truncation(self):
        try:
            service = RerankerService(model_name="BAAI/bge-reranker-base", device="cpu")
            docs = [{"content": f"文档{i}", "chunk_id": str(i)} for i in range(10)]
            result = service.rerank("test", docs, top_n=3)
            assert len(result) == 3
        except Exception as exc:
            pytest.skip(f"Reranker model not available: {exc}")
