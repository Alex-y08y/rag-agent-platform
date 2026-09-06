"""Tests for Agent tools."""
from __future__ import annotations

import pytest

from app.tools.calculator import CalculatorTool, CalculatorInput
from app.tools.knowledge_search import KnowledgeSearchTool, KnowledgeSearchInput
from app.tools.base import BaseTool


class TestCalculatorTool:
    @pytest.mark.asyncio
    async def test_basic_addition(self):
        tool = CalculatorTool()
        result = await tool(expression="2 + 3")
        assert result["result"] == 5

    @pytest.mark.asyncio
    async def test_multiplication(self):
        tool = CalculatorTool()
        result = await tool(expression="4 * 5")
        assert result["result"] == 20

    @pytest.mark.asyncio
    async def test_percentage_function(self):
        tool = CalculatorTool()
        result = await tool(expression="percent(25, 200)")
        assert result["result"] == 12.5

    @pytest.mark.asyncio
    async def test_complex_expression(self):
        tool = CalculatorTool()
        result = await tool(expression="(100 + 50) * 0.85")
        assert abs(result["result"] - 127.5) < 0.01

    @pytest.mark.asyncio
    async def test_sqrt(self):
        tool = CalculatorTool()
        result = await tool(expression="sqrt(144)")
        assert result["result"] == 12.0

    @pytest.mark.asyncio
    async def test_invalid_expression(self):
        tool = CalculatorTool()
        result = await tool(expression="__import__('os').system('ls')")
        assert result.get("error") is not None

    def test_tool_metadata(self):
        tool = CalculatorTool()
        assert tool.name == "calculator"
        assert "数学计算" in tool.description
        assert tool.input_schema == CalculatorInput


class TestKnowledgeSearchTool:
    def test_tool_metadata(self):
        tool = KnowledgeSearchTool()
        assert tool.name == "knowledge_search"
        assert tool.input_schema == KnowledgeSearchInput

    @pytest.mark.asyncio
    async def test_no_kb_specified(self):
        tool = KnowledgeSearchTool()
        result = await tool(query="test", knowledge_base_id=None)
        assert "error" in result or result.get("total", 0) == 0


class TestBaseTool:
    def test_to_langchain_tool(self):
        tool = CalculatorTool()
        spec = tool.to_langchain_tool()
        assert spec["name"] == "calculator"
        assert "description" in spec
        assert "parameters" in spec
