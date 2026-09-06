"""Redis-backed conversation memory with short-term and long-term storage."""
from __future__ import annotations

import json
import time
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RedisMemory:
    """Conversation memory backed by Redis.

    Features:
    - Short-term: recent messages (up to MEMORY_MAX_TURNS)
    - Long-term: conversation summary (auto-generated when history grows)
    - Automatic summarization when exceeding threshold
    """

    def __init__(self) -> None:
        self._client = None
        self.max_turns = settings.MEMORY_MAX_TURNS
        self.summary_threshold = settings.MEMORY_SUMMARY_THRESHOLD

    def _get_client(self) -> Any:
        if self._client is None:
            import redis
            self._client = redis.from_url(settings.redis_url, decode_responses=True)
            logger.info("Connected to Redis at %s", settings.redis_url)
        return self._client

    def _key(self, conversation_id: str) -> str:
        return f"rag:conv:{conversation_id}"

    def _summary_key(self, conversation_id: str) -> str:
        return f"rag:conv:{conversation_id}:summary"

    async def add_message(
        self, conversation_id: str, role: str, content: str, metadata: dict | None = None
    ) -> None:
        """Add a message to conversation history."""
        try:
            client = self._get_client()
            msg = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "metadata": metadata or {},
            }
            client.rpush(self._key(conversation_id), json.dumps(msg, ensure_ascii=False))
            # Trim to max turns
            client.ltrim(self._key(conversation_id), -self.max_turns, -1)
        except Exception as exc:
            logger.warning("Redis add_message failed: %s", exc)

    async def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Get conversation history as list of {role, content}."""
        try:
            client = self._get_client()
            raw = client.lrange(self._key(conversation_id), 0, -1)
            messages = []
            for item in raw:
                msg = json.loads(item)
                messages.append({"role": msg["role"], "content": msg["content"]})
            return messages
        except Exception as exc:
            logger.warning("Redis get_history failed: %s", exc)
            return []

    async def get_summary(self, conversation_id: str) -> str:
        """Get long-term conversation summary."""
        try:
            client = self._get_client()
            return client.get(self._summary_key(conversation_id)) or ""
        except Exception as exc:
            logger.warning("Redis get_summary failed: %s", exc)
            return ""

    async def set_summary(self, conversation_id: str, summary: str) -> None:
        """Set conversation summary."""
        try:
            client = self._get_client()
            client.set(self._summary_key(conversation_id), summary)
        except Exception as exc:
            logger.warning("Redis set_summary failed: %s", exc)

    async def maybe_summarize(self, conversation_id: str) -> str | None:
        """Auto-summarize if history exceeds threshold.

        Returns the new summary if summarization was performed, None otherwise.
        """
        history = await self.get_history(conversation_id)
        if len(history) < self.summary_threshold:
            return None

        try:
            from app.services.llm_service import get_llm_service
            llm = get_llm_service()
            history_text = "\n".join(
                f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
                for m in history
            )
            messages = [
                {"role": "system", "content": "请用简洁的中文总结以下对话的核心内容和关键信息，不超过300字。"},
                {"role": "user", "content": history_text[:4000]},
            ]
            summary, _ = await llm.chat(messages)
            await self.set_summary(conversation_id, summary)
            logger.info("Auto-summarized conversation %s", conversation_id)
            return summary
        except Exception as exc:
            logger.warning("Auto-summarize failed: %s", exc)
            return None

    async def clear(self, conversation_id: str) -> None:
        """Clear all conversation data."""
        try:
            client = self._get_client()
            client.delete(self._key(conversation_id), self._summary_key(conversation_id))
        except Exception as exc:
            logger.warning("Redis clear failed: %s", exc)


# Singleton
_memory: RedisMemory | None = None


def get_memory() -> RedisMemory:
    global _memory
    if _memory is None:
        _memory = RedisMemory()
    return _memory
