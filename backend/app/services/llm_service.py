"""LLM service using OpenAI-compatible API (supports Qwen via DashScope)."""
from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI

from app.core.config import settings
from app.core import runtime_config
from app.core.exceptions import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _cfg(key: str, default: Any = None) -> Any:
    """Get config value: runtime override first, then env var."""
    val = runtime_config.get(key)
    if val is not None and val != "":
        return val
    return getattr(settings, key, default)


class LLMService:
    """Wrapper around OpenAI-compatible chat completions API."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _effective_api_key(self) -> str:
        return _cfg("OPENAI_API_KEY", "") or _cfg("DASHSCOPE_API_KEY", "")

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = self._effective_api_key()
            if not api_key:
                logger.warning("No LLM API key configured. LLM calls will fail.")
            self._client = AsyncOpenAI(
                api_key=api_key or "sk-not-configured",
                base_url=_cfg("LLM_BASE_URL", settings.LLM_BASE_URL),
                timeout=int(_cfg("LLM_TIMEOUT", settings.LLM_TIMEOUT)),
            )
        return self._client

    def reconfigure(self) -> None:
        """Reset the OpenAI client to pick up new runtime config."""
        self._client = None
        logger.info("LLM client reconfigured with new settings")

    async def test_connection(self) -> dict[str, Any]:
        """Test LLM connection with a minimal prompt. Returns success/error."""
        try:
            api_key = self._effective_api_key()
            if not api_key:
                return {"ok": False, "error": "未配置 API Key"}
            # Force fresh client
            old = self._client
            self._client = None
            resp = await self.client.chat.completions.create(
                model=_cfg("LLM_MODEL", settings.LLM_MODEL),
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                temperature=0,
            )
            self._client = old  # restore
            return {
                "ok": True,
                "model": _cfg("LLM_MODEL", settings.LLM_MODEL),
                "base_url": _cfg("LLM_BASE_URL", settings.LLM_BASE_URL),
                "reply": (resp.choices[0].message.content or "")[:50],
            }
        except Exception as exc:
            self._client = None
            return {"ok": False, "error": str(exc)[:200]}

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Non-streaming chat completion.

        Returns (content, token_usage).
        """
        try:
            kwargs: dict[str, Any] = dict(
                model=_cfg("LLM_MODEL", settings.LLM_MODEL),
                messages=messages,
                temperature=temperature if temperature is not None else float(_cfg("LLM_TEMPERATURE", settings.LLM_TEMPERATURE)),
                max_tokens=max_tokens or int(_cfg("LLM_MAX_TOKENS", settings.LLM_MAX_TOKENS)),
            )
            if response_format:
                kwargs["response_format"] = response_format

            resp = await self.client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                "total_tokens": getattr(resp.usage, "total_tokens", 0),
            }
            return content, usage
        except Exception as exc:
            logger.error("LLM chat failed: %s", exc)
            raise LLMError(f"LLM call failed: {exc}") from exc

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion yielding text deltas."""
        try:
            resp = await self.client.chat.completions.create(
                model=_cfg("LLM_MODEL", settings.LLM_MODEL),
                messages=messages,
                temperature=temperature if temperature is not None else float(_cfg("LLM_TEMPERATURE", settings.LLM_TEMPERATURE)),
                max_tokens=int(_cfg("LLM_MAX_TOKENS", settings.LLM_MAX_TOKENS)),
                stream=True,
            )
            async for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            logger.error("LLM stream failed: %s", exc)
            raise LLMError(f"LLM stream failed: {exc}") from exc

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Generate a JSON response from the LLM."""
        messages = [
            {"role": "system", "content": system_prompt + "\n\n请严格输出 JSON 格式，不要输出任何其他内容。"},
            {"role": "user", "content": user_prompt},
        ]
        content, _ = await self.chat(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:]
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM JSON: %s", content[:200])
                return {"raw": content}


# Singleton
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
