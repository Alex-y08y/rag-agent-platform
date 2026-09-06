"""Tests for retrieval metrics and RRF fusion."""
from __future__ import annotations

import pytest

from app.retrieval.rrf import rrf_fuse, weighted_fuse


class TestRRFFusion:
    def test_basic_fusion(self):
        list1 = [
            {"chunk_id": "a", "score": 0.9, "content": "A"},
            {"chunk_id": "b", "score": 0.8, "content": "B"},
        ]
        list2 = [
            {"chunk_id": "b", "score": 0.95, "content": "B"},
            {"chunk_id": "c", "score": 0.7, "content": "C"},
        ]
        result = rrf_fuse([list1, list2], top_k=10)
        assert len(result) == 3
        # b appears in both -> highest RRF score
        assert result[0]["chunk_id"] == "b"

    def test_single_list(self):
        docs = [{"chunk_id": "a", "score": 0.9}, {"chunk_id": "b", "score": 0.8}]
        result = rrf_fuse([docs], top_k=10)
        assert len(result) == 2

    def test_empty_lists(self):
        assert rrf_fuse([], top_k=10) == []
        assert rrf_fuse([[], []], top_k=10) == []

    def test_top_k_truncation(self):
        docs = [{"chunk_id": str(i), "score": 1.0 - i * 0.1} for i in range(10)]
        result = rrf_fuse([docs], top_k=3)
        assert len(result) == 3

    def test_rrf_score_positive(self):
        list1 = [{"chunk_id": "a", "score": 0.9}]
        list2 = [{"chunk_id": "a", "score": 0.8}]
        result = rrf_fuse([list1, list2], top_k=1)
        assert result[0]["rrf_score"] > 0

    def test_duplicate_chunk_ids_deduped(self):
        list1 = [{"chunk_id": "a", "score": 0.9, "content": "A1"}]
        list2 = [{"chunk_id": "a", "score": 0.8, "content": "A2"}]
        result = rrf_fuse([list1, list2], top_k=10)
        assert len(result) == 1


class TestWeightedFusion:
    def test_weighted_fusion_basic(self):
        v = [{"chunk_id": "a", "score": 0.9}, {"chunk_id": "b", "score": 0.5}]
        b = [{"chunk_id": "a", "score": 0.8}, {"chunk_id": "c", "score": 0.7}]
        result = weighted_fuse(v, b, top_k=10)
        assert len(result) == 3
