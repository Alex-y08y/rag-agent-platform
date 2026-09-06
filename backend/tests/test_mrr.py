"""Tests for MRR (Mean Reciprocal Rank) metric."""
from __future__ import annotations

import pytest

from app.evaluation.metrics import reciprocal_rank, mean_reciprocal_rank


class TestReciprocalRank:
    def test_first_rank(self):
        expected = ["a.pdf"]
        retrieved = ["a.pdf", "b.pdf"]
        assert reciprocal_rank(expected, retrieved) == 1.0

    def test_second_rank(self):
        expected = ["b.pdf"]
        retrieved = ["a.pdf", "b.pdf"]
        assert reciprocal_rank(expected, retrieved) == 0.5

    def test_fifth_rank(self):
        expected = ["e.pdf"]
        retrieved = ["a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"]
        assert reciprocal_rank(expected, retrieved) == 0.2

    def test_not_found(self):
        expected = ["z.pdf"]
        retrieved = ["a.pdf", "b.pdf"]
        assert reciprocal_rank(expected, retrieved) == 0.0

    def test_empty_expected(self):
        assert reciprocal_rank([], ["a.pdf"]) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank(["a.pdf"], []) == 0.0


class TestMRR:
    def test_perfect_mrr(self):
        samples = [
            (["a.pdf"], ["a.pdf"]),
            (["b.pdf"], ["b.pdf"]),
        ]
        assert mean_reciprocal_rank(samples) == 1.0

    def test_mixed_mrr(self):
        samples = [
            (["a.pdf"], ["a.pdf"]),       # RR=1
            (["b.pdf"], ["c.pdf", "b.pdf"]),  # RR=0.5
        ]
        assert mean_reciprocal_rank(samples) == 0.75

    def test_all_not_found(self):
        samples = [
            (["a.pdf"], ["b.pdf"]),
            (["c.pdf"], ["d.pdf"]),
        ]
        assert mean_reciprocal_rank(samples) == 0.0

    def test_empty_samples(self):
        assert mean_reciprocal_rank([]) == 0.0
