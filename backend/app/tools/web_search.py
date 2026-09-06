"""Web Search Tool: search the web for current information."""
from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger
from app.tools.base import BaseTool

logger = get_logger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(..., description="搜索查询")
    num_results: int = Field(default=5, ge=1, le=10)


class WebSearchTool(BaseTool[WebSearchInput, dict[str, Any]]):
    """Search the web using DuckDuckGo HTML (no API key required).

    Falls back gracefully if search is unavailable.
    """

    name = "web_search"
    description = (
        "搜索互联网获取最新信息。当知识库中没有相关信息，或用户询问时事、"
        "最新动态、外部信息时使用此工具。"
    )
    input_schema = WebSearchInput

    async def execute(self, input_data: WebSearchInput) -> dict[str, Any]:
        try:
            # Use DuckDuckGo HTML endpoint (no API key needed)
            url = "https://html.duckduckgo.com/html/"
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    url,
                    data={"q": input_data.query},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()

                # Parse HTML results (simple regex-based extraction)
                import re
                html = resp.text
                results = []

                # Extract result snippets
                snippets = re.findall(
                    r'class="result__snippet"[^>]*>(.*?)</a>',
                    html, re.DOTALL,
                )
                titles = re.findall(
                    r'class="result__a"[^>]*>(.*?)</a>',
                    html, re.DOTALL,
                )

                for i in range(min(len(snippets), input_data.num_results)):
                    title = re.sub(r"<[^>]+>", "", titles[i]).strip() if i < len(titles) else ""
                    snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
                    results.append({"title": title, "snippet": snippet})

                logger.info("Web search: '%s', %d results", input_data.query[:50], len(results))
                return {"results": results, "total": len(results), "query": input_data.query}

        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            return {
                "results": [],
                "total": 0,
                "query": input_data.query,
                "error": f"Web search unavailable: {exc}",
                "note": "网络搜索暂时不可用，建议基于知识库内容回答。",
            }
