"""Tests for answer accuracy, faithfulness, and relevancy evaluation."""
from __future__ import annotations

import pytest

from app.evaluation.llm_judge import LLMJudge


class TestRuleBasedAccuracy:
    def test_numeric_exact_match(self):
        judge = LLMJudge()
        score = judge._rule_based_accuracy("500元", "标准是500元")
        assert score == 1.0

    def test_numeric_partial_match(self):
        judge = LLMJudge()
        score = judge._rule_based_accuracy("500元 3天", "标准是500元")
        assert score == 0.5

    def test_numeric_no_match(self):
        judge = LLMJudge()
        score = judge._rule_based_accuracy("500元", "标准是300元")
        assert score == 0.0

    def test_non_numeric_returns_none(self):
        judge = LLMJudge()
        assert judge._rule_based_accuracy("公司政策规定", "根据制度执行") is None

    def test_empty_answer(self):
        judge = LLMJudge()
        score = judge._rule_based_accuracy("500元", "")
        assert score == 0.0


class TestLLMJudge:
    @pytest.mark.asyncio
    async def test_evaluate_accuracy_no_ground_truth(self):
        judge = LLMJudge()
        result = await judge.evaluate_accuracy("test", "", "answer")
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_faithfulness_missing_context(self):
        judge = LLMJudge()
        result = await judge.evaluate_faithfulness("q", "", "answer")
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_relevancy_empty_answer(self):
        judge = LLMJudge()
        result = await judge.evaluate_relevancy("q", "")
        assert result["score"] == 0.0
