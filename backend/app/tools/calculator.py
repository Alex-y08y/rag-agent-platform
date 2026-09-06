"""Calculator Tool: mathematical and statistical calculations."""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.tools.base import BaseTool

logger = get_logger(__name__)

# Safe operator mapping
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe functions
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "pow": math.pow,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
    "ceil": math.ceil,
    "floor": math.floor,
    "mean": lambda x: sum(x) / len(x) if x else 0,
    "percent": lambda part, total: (part / total * 100) if total else 0,
}


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="数学表达式，例如 '23 * 1.15 + 100' 或 'percent(25, 200)'")


class CalculatorTool(BaseTool[CalculatorInput, dict[str, Any]]):
    """Perform safe mathematical calculations."""

    name = "calculator"
    description = (
        "执行数学计算，包括基本运算、百分比、统计计算。"
        "支持函数：abs, round, min, max, sum, sqrt, pow, log, exp, percent(part, total), mean。"
        "当需要精确计算数字、百分比、增长率、统计量时使用。"
    )
    input_schema = CalculatorInput

    @staticmethod
    def _eval_node(node: ast.AST) -> Any:
        """Safely evaluate an AST node."""
        if isinstance(node, ast.Expression):
            return CalculatorTool._eval_node(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = CalculatorTool._eval_node(node.left)
            right = CalculatorTool._eval_node(node.right)
            op_type = type(node.op)
            if op_type in _OPERATORS:
                return _OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = CalculatorTool._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in _OPERATORS:
                return _OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in _FUNCTIONS:
                    args = [CalculatorTool._eval_node(a) for a in node.args]
                    return _FUNCTIONS[func_name](*args)
            raise ValueError(f"Unsupported function: {node.func}")
        elif isinstance(node, ast.Name):
            if node.id in _FUNCTIONS:
                return _FUNCTIONS[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.List):
            return [CalculatorTool._eval_node(e) for e in node.elts]
        else:
            raise ValueError(f"Unsupported expression: {type(node)}")

    async def execute(self, input_data: CalculatorInput) -> dict[str, Any]:
        expression = input_data.expression.strip()
        try:
            tree = ast.parse(expression, mode="eval")
            result = self._eval_node(tree)
            # Round to reasonable precision
            if isinstance(result, float):
                result = round(result, 6)
            logger.info("Calculator: '%s' = %s", expression, result)
            return {
                "expression": expression,
                "result": result,
                "formatted": f"{expression} = {result}",
            }
        except Exception as exc:
            logger.error("Calculator failed: %s", exc)
            return {"expression": expression, "error": str(exc), "result": None}
