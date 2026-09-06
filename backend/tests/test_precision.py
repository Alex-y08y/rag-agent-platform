"""Tests for Precision@K metric."""
from __future__ import annotations

import pytest

from app.evaluation.metrics import precision_at_k, batch_precision_at_k


class TestPrecisionAtK:
    def test_perfect_precision(self):
        expected = ["a.pdf"]
        retrieved = ["a.pdf"]
        assert precision_at_k(expected, retrieved, 1) == 1.0

    def test_half_precision(self):
        expected = ["a.pdf"]
        retrieved = ["a.pdf", "b.pdf"]
        assert precision_at_k(expected, retrieved, 2) == 0.5

    def test_zero_precision(self):
        expected = ["a.pdf"]
        retrieved = ["b.pdf", "c.pdf"]
        assert precision_at_k(expected, retrieved, 2) == 0.0

    def test_precision_respects_k(self):
        expected = ["a.pdf"]
        retrieved = ["b.pdf", "a.pdf"]
        assert precision_at_k(expected, retrieved, 1) == 0.0
        assert precision_at_k(expected, retrieved, 2) == 0.5

    def test_k_larger_than_retrieved(self):
        expected = ["a.pdf"]
        retrieved = ["a.pdf"]
        # K=5 but only 1 retrieved -> precision = 1/5
        assert precision_at_k(expected, retrieved, 5) == 0.2

    def test_empty_retrieved(self):
        assert precision_at_k(["a.pdf"], [], 5) == 0.0

    def test_empty_expected(self):
        assert precision_at_k([], ["a.pdf"], 5) == 0.0


class TestBatchPrecision:
    def test_batch_average(self):
        samples = [
            (["a.pdf"], ["a.pdf", "b.pdf"]),
            (["c.pdf"], ["c.pdf", "d.pdf"]),
        ]
        assert batch_precision_at_k(samples, 2) == 0.5

    def test_empty_batch(self):
        assert batch_precision_at_k([], 5) == 0.0
