"""Settings API routes — view and update runtime configuration (LLM API keys etc.)."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core import runtime_config
from app.core.config import settings
from app.core.logging import get_logger, new_request_id
from app.services.llm_service import get_llm_service

logger = get_logger(__name__)
router = APIRouter(prefix="/settings", tags=["settings"])


class SettingsOut(BaseModel):
    LLM_PROVIDER: str = ""
    LLM_MODEL: str = ""
    LLM_BASE_URL: str = ""
    OPENAI_API_KEY: str = ""
    DASHSCOPE_API_KEY: str = ""
    _OPENAI_API_KEY_configured: bool = False
    _DASHSCOPE_API_KEY_configured: bool = False
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: int = 60


class SettingsUpdate(BaseModel):
    LLM_PROVIDER: str | None = None
    LLM_MODEL: str | None = None
    LLM_BASE_URL: str | None = None
    OPENAI_API_KEY: str | None = None
    DASHSCOPE_API_KEY: str | None = None
    LLM_TEMPERATURE: float | None = None
    LLM_MAX_TOKENS: int | None = None
    LLM_TIMEOUT: int | None = None


class TestResult(BaseModel):
    ok: bool
    error: str | None = None
    model: str | None = None
    base_url: str | None = None
    reply: str | None = None


def _merge_with_defaults() -> dict:
    """Merge runtime config with env var defaults for display."""
    runtime = runtime_config.get_all()
    merged = {}
    for field in SettingsOut.model_fields:
        if field.startswith("_"):
            merged[field] = runtime.get(field, False)
        else:
            val = runtime.get(field)
            if val is None or val == "":
                val = getattr(settings, field, "")
            merged[field] = val
    return merged


@router.get("", response_model=SettingsOut)
def get_settings():
    """Get current settings (API keys masked)."""
    new_request_id()
    return SettingsOut(**_merge_with_defaults())


@router.post("", response_model=SettingsOut)
def update_settings(body: SettingsUpdate):
    """Update runtime settings. Persists to data/config/settings.json.

    API keys containing '*' are treated as 'unchanged' (masked display value).
    """
    new_request_id()
    values = body.model_dump(exclude_none=True)
    result = runtime_config.update(values)
    # Reconfigure LLM client to pick up new settings
    try:
        get_llm_service().reconfigure()
    except Exception as exc:
        logger.warning("LLM reconfigure failed: %s", exc)
    merged = _merge_with_defaults()
    return SettingsOut(**merged)


@router.post("/test", response_model=TestResult)
async def test_llm_connection():
    """Test LLM API connection with current settings."""
    new_request_id()
    svc = get_llm_service()
    result = await svc.test_connection()
    return TestResult(**result)


@router.post("/reset")
def reset_settings():
    """Clear all runtime settings (revert to .env defaults)."""
    new_request_id()
    runtime_config.reset()
    try:
        get_llm_service().reconfigure()
    except Exception as exc:
        logger.warning("LLM reconfigure failed: %s", exc)
    return {"success": True, "message": "Runtime settings cleared, using .env defaults"}
