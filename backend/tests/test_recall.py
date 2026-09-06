"""Tests for Recall@K metric."""
from __future__ import annotations

import pytest

from app.evaluation.metrics import recall_at_k, batch_recall_at_k


class TestRecallAtK:
    def test_perfect_recall(self):
        expected = ["doc_a.pdf"]
        retrieved = ["doc_a.pdf", "doc_b.pdf"]
        assert recall_at_k(expected, retrieved, 5) == 1.0

    def test_zero_recall(self):
        expected = ["doc_a.pdf"]
        retrieved = ["doc_b.pdf", "doc_c.pdf"]
        assert recall_at_k(expected, retrieved, 5) == 0.0

    def test_partial_recall(self):
        expected = ["doc_a.pdf", "doc_b.pdf"]
        retrieved = ["doc_a.pdf", "doc_c.pdf"]
        assert recall_at_k(expected, retrieved, 5) == 0.5

    def test_recall_respects_k(self):
        expected = ["doc_c.pdf"]
        retrieved = ["doc_a.pdf", "doc_b.pdf", "doc_c.pdf"]
        assert recall_at_k(expected, retrieved, 2) == 0.0
        assert recall_at_k(expected, retrieved, 3) == 1.0

    def test_empty_expected(self):
        assert recall_at_k([], ["a.pdf"], 5) == 1.0

    def test_empty_retrieved(self):
        assert recall_at_k(["a.pdf"], [], 5) == 0.0

    def test_case_insensitive_match(self):
        expected = ["Travel_Policy.PDF"]
        retrieved = ["travel_policy.pdf"]
        assert recall_at_k(expected, retrieved, 5) == 1.0

    def test_path_ignored(self):
        expected = ["policy.pdf"]
        retrieved = ["/data/uploads/policy.pdf"]
        assert recall_at_k(expected, retrieved, 5) == 1.0

    def test_recall_at_1(self):
        expected = ["b.pdf"]
        retrieved = ["a.pdf", "b.pdf"]
        assert recall_at_k(expected, retrieved, 1) == 0.0
        assert recall_at_k(expected, retrieved, 2) == 1.0


class TestBatchRecall:
    def test_batch_average(self):
        samples = [
            (["a.pdf"], ["a.pdf", "b.pdf"]),
            (["c.pdf"], ["d.pdf", "e.pdf"]),
        ]
        assert batch_recall_at_k(samples, 5) == 0.5

    def test_empty_batch(self):
        assert batch_recall_at_k([], 5) == 0.0
