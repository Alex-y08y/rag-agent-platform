"""Tool registry and factory."""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool
from app.tools.calculator import CalculatorTool
from app.tools.document_analysis import DocumentAnalysisTool
from app.tools.knowledge_search import KnowledgeSearchTool
from app.tools.sql_query import SQLQueryTool
from app.tools.web_search import WebSearchTool


def get_all_tools() -> dict[str, BaseTool]:
    """Get all available tools."""
    return {
        "knowledge_search": KnowledgeSearchTool(),
        "sql_query": SQLQueryTool(),
        "calculator": CalculatorTool(),
        "document_analysis": DocumentAnalysisTool(),
        "web_search": WebSearchTool(),
    }


def get_tool(name: str) -> BaseTool | None:
    """Get a tool by name."""
    return get_all_tools().get(name)


def get_tool_specs() -> list[dict[str, Any]]:
    """Get LangChain-compatible tool specifications."""
    return [t.to_langchain_tool() for t in get_all_tools().values()]
