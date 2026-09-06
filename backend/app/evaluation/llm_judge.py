"""LLM-as-a-Judge for Answer Accuracy, Faithfulness, and Answer Relevancy."""
from __future__ import annotations

import re
from typing import Any

from app.core.logging import get_logger
from app.services.llm_service import get_llm_service

logger = get_logger(__name__)

ACCURACY_PROMPT = """你是一个答案准确性评判专家。比较生成答案和标准答案，给出准确性评分。

评分标准（0-1分）：
- 1.0: 完全正确，所有关键信息一致
- 0.7: 基本正确，有微小差异
- 0.4: 部分正确，有明显错误或遗漏
- 0.0: 完全错误或不相关

输出 JSON：{"score": 0.0-1.0, "reason": "评分理由"}
"""

FAITHFULNESS_PROMPT = """你是一个答案忠实度评判专家。检查生成答案是否被检索上下文支撑。

评分标准（0-1分）：
- 1.0: 答案完全由上下文支撑，无编造
- 0.7: 大部分有支撑，少量无法验证
- 0.4: 有明显编造或与上下文矛盾
- 0.0: 答案与上下文无关或完全编造

输出 JSON：{"score": 0.0-1.0, "reason": "评分理由"}
"""

RELEVANCY_PROMPT = """你是一个答案相关性评判专家。检查生成答案是否回答了用户问题。

评分标准（0-1分）：
- 1.0: 完全回答了问题
- 0.7: 基本回答，有少量偏离
- 0.4: 部分相关，未充分回答
- 0.0: 完全不相关

输出 JSON：{"score": 0.0-1.0, "reason": "评分理由"}
"""


class LLMJudge:
    """LLM-as-a-Judge evaluator for generation quality."""

    def __init__(self) -> None:
        self.llm = get_llm_service()

    async def evaluate_accuracy(
        self, question: str, ground_truth: str, generated_answer: str
    ) -> dict[str, Any]:
        """Evaluate answer accuracy against ground truth."""
        if not ground_truth:
            return {"score": 0.0, "reason": "No ground truth provided"}

        # Try rule-based for numeric/date answers first
        rule_score = self._rule_based_accuracy(ground_truth, generated_answer)
        if rule_score is not None:
            return {"score": rule_score, "reason": "Rule-based numeric match"}

        user_prompt = (
            f"用户问题：{question}\n"
            f"标准答案：{ground_truth}\n"
            f"生成答案：{generated_answer}\n"
        )
        try:
            result = await self.llm.generate_json(ACCURACY_PROMPT, user_prompt, temperature=0.0)
            return {"score": float(result.get("score", 0)), "reason": result.get("reason", "")}
        except Exception as exc:
            logger.error("Accuracy judge failed: %s", exc)
            return {"score": 0.0, "reason": f"Judge failed: {exc}"}

    async def evaluate_faithfulness(
        self, question: str, context: str, answer: str
    ) -> dict[str, Any]:
        """Evaluate whether answer is supported by context."""
        if not context or not answer:
            return {"score": 0.0, "reason": "Missing context or answer"}

        user_prompt = (
            f"用户问题：{question}\n"
            f"检索上下文：{context[:3000]}\n"
            f"生成答案：{answer}\n"
        )
        try:
            result = await self.llm.generate_json(FAITHFULNESS_PROMPT, user_prompt, temperature=0.0)
            return {"score": float(result.get("score", 0)), "reason": result.get("reason", "")}
        except Exception as exc:
            logger.error("Faithfulness judge failed: %s", exc)
            return {"score": 0.0, "reason": f"Judge failed: {exc}"}

    async def evaluate_relevancy(
        self, question: str, answer: str
    ) -> dict[str, Any]:
        """Evaluate whether answer addresses the question."""
        if not answer:
            return {"score": 0.0, "reason": "Empty answer"}

        user_prompt = f"用户问题：{question}\n生成答案：{answer}\n"
        try:
            result = await self.llm.generate_json(RELEVANCY_PROMPT, user_prompt, temperature=0.0)
            return {"score": float(result.get("score", 0)), "reason": result.get("reason", "")}
        except Exception as exc:
            logger.error("Relevancy judge failed: %s", exc)
            return {"score": 0.0, "reason": f"Judge failed: {exc}"}

    @staticmethod
    def _rule_based_accuracy(ground_truth: str, answer: str) -> float | None:
        """Rule-based accuracy for numeric/date/fixed answers.

        Returns a score if rule-based applies, None otherwise.
        """
        # Extract numbers from both
        gt_numbers = set(re.findall(r"\d+\.?\d*", ground_truth))
        ans_numbers = set(re.findall(r"\d+\.?\d*", answer))

        # If ground truth is primarily numeric
        gt_words = len(re.findall(r"[\u4e00-\u9fff]", ground_truth))
        if gt_numbers and gt_words < 10:
            if not ans_numbers:
                return 0.0
            overlap = len(gt_numbers & ans_numbers)
            return overlap / len(gt_numbers)

        return None


# Singleton
_judge: LLMJudge | None = None


def get_llm_judge() -> LLMJudge:
    global _judge
    if _judge is None:
        _judge = LLMJudge()
    return _judge
