"""Query Rewrite: resolve coreference and context gaps using conversation history."""
from __future__ import annotations

from app.core.logging import get_logger
from app.services.llm_service import get_llm_service

logger = get_logger(__name__)

SYSTEM_PROMPT = """你是一个查询重写专家。根据对话历史，将用户当前问题重写为一个完整、自包含的查询。

规则：
1. 解析代词（它、这个、那个、他们等），替换为具体实体
2. 补全省略的上下文信息
3. 保持原始问题的意图和关键信息
4. 如果问题已经完整自包含，直接返回原问题
5. 输出 JSON 格式：{"rewritten_query": "..."}

示例：
历史：公司差旅报销制度有什么变化？
当前：什么时候生效？
重写：公司2026年差旅报销制度什么时候生效？
"""


class QueryRewriter:
    """Rewrite user queries using conversation history and LLM."""

    def __init__(self) -> None:
        self.llm = get_llm_service()

    async def rewrite(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Rewrite a query given conversation history.

        Args:
            query: Current user query.
            history: List of {"role": "user"/"assistant", "content": "..."} messages.

        Returns:
            Rewritten query string.
        """
        if not history:
            return query

        # Only use last 6 turns for context
        recent = history[-6:]
        history_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in recent
        )

        user_prompt = f"""对话历史：
{history_text}

当前问题：{query}

请重写当前问题为完整自包含的查询。"""

        try:
            result = await self.llm.generate_json(SYSTEM_PROMPT, user_prompt, temperature=0.1)
            rewritten = result.get("rewritten_query", query)
            if rewritten and rewritten != query:
                logger.info("Query rewritten: '%s' -> '%s'", query, rewritten)
            return rewritten or query
        except Exception as exc:
            logger.warning("Query rewrite failed, using original: %s", exc)
            return query


# Singleton
_rewriter: QueryRewriter | None = None


def get_query_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter
