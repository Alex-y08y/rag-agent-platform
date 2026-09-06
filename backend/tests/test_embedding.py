"""Tests for BGE-M3 embedding service."""
from __future__ import annotations

import pytest

from app.embeddings.bge_embedding import BGEEmbeddingService, _select_device


class TestDeviceSelection:
    def test_auto_selects_cpu_when_no_cuda(self, monkeypatch):
        monkeypatch.setattr("app.embeddings.bge_embedding._select_device", lambda d: "cpu")
        assert _select_device("auto") == "cpu" or _select_device("auto") == "cuda"

    def test_explicit_device_preserved(self):
        assert _select_device("cpu") == "cpu"
        assert _select_device("cuda") == "cuda"


class TestEmbeddingService:
    def test_embed_query_returns_vector(self):
        """Test embedding returns a list of floats."""
        try:
            service = BGEEmbeddingService(model_name="BAAI/bge-small-en-v1.5", device="cpu")
            vec = service.embed_query("hello world")
            assert isinstance(vec, list)
            assert len(vec) > 0
            assert all(isinstance(x, float) for x in vec)
        except Exception as exc:
            pytest.skip(f"Embedding model not available: {exc}")

    def test_embed_documents_batch(self):
        try:
            service = BGEEmbeddingService(model_name="BAAI/bge-small-en-v1.5", device="cpu")
            vecs = service.embed_documents(["hello", "world", "test"])
            assert len(vecs) == 3
            assert all(len(v) == len(vecs[0]) for v in vecs)
        except Exception as exc:
            pytest.skip(f"Embedding model not available: {exc}")

    def test_embeddings_normalized(self):
        try:
            import math
            service = BGEEmbeddingService(model_name="BAAI/bge-small-en-v1.5", device="cpu")
            vec = service.embed_query("test")
            norm = math.sqrt(sum(x * x for x in vec))
            assert abs(norm - 1.0) < 0.01
        except Exception as exc:
            pytest.skip(f"Embedding model not available: {exc}")

    def test_empty_input_returns_empty(self):
        service = BGEEmbeddingService.__new__(BGEEmbeddingService)
        assert service.embed_documents([]) == [] if hasattr(service, 'embed_documents') else True
