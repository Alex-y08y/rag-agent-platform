"""Runtime configuration manager — persists UI-configured settings to JSON.

Allows users to configure LLM API keys etc. from the frontend without
editing .env or restarting the backend. Runtime values override env vars.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

_CONFIG_DIR = Path(os.getenv("RUNTIME_CONFIG_DIR", "./data/config"))
_CONFIG_FILE = _CONFIG_DIR / "settings.json"

# Fields that can be configured at runtime
_RUNTIME_FIELDS = {
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
    "LLM_TIMEOUT",
}

_lock = threading.RLock()
_cache: dict[str, Any] = {}


def _load() -> dict[str, Any]:
    """Load runtime config from disk (cached in memory)."""
    global _cache
    if _cache:
        return _cache
    try:
        if _CONFIG_FILE.exists():
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
            logger.info("Loaded runtime config from %s", _CONFIG_FILE)
        else:
            _cache = {}
    except Exception as exc:
        logger.warning("Failed to load runtime config: %s", exc)
        _cache = {}
    return _cache


def _save(data: dict[str, Any]) -> None:
    """Persist runtime config to disk."""
    global _cache
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    _cache = data
    logger.info("Saved runtime config to %s", _CONFIG_FILE)


def get(key: str, default: Any = None) -> Any:
    """Get a runtime config value (falls back to default)."""
    with _lock:
        data = _load()
        return data.get(key, default)


def get_all() -> dict[str, Any]:
    """Get all runtime config values (with API keys masked)."""
    with _lock:
        data = dict(_load())
    # Mask sensitive fields
    for key in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY"):
        if data.get(key):
            val = data[key]
            data[key] = val[:4] + "*" * (len(val) - 8) + val[-4:] if len(val) > 8 else "****"
            data[f"_{key}_configured"] = True
        else:
            data[f"_{key}_configured"] = False
    return data


def update(values: dict[str, Any]) -> dict[str, Any]:
    """Update runtime config values. Only known fields are accepted.

    If an API key field is set to a masked value (contains '*'), it is
    left unchanged (user didn't want to change it).
    """
    with _lock:
        data = dict(_load())
        for key, value in values.items():
            if key not in _RUNTIME_FIELDS:
                continue
            # Skip masked API keys (user didn't change them)
            if key in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY"):
                if value and "*" in str(value):
                    continue
                if value == "":
                    data.pop(key, None)
                    continue
            data[key] = value
        _save(data)
        return get_all()


def reset() -> None:
    """Clear all runtime config (revert to env vars)."""
    with _lock:
        _save({})


def is_configured() -> bool:
    """Check if any LLM API key is configured (runtime or env)."""
    from app.core.config import settings
    runtime = _load()
    return bool(
        runtime.get("OPENAI_API_KEY")
        or runtime.get("DASHSCOPE_API_KEY")
        or settings.OPENAI_API_KEY
        or settings.DASHSCOPE_API_KEY
    )
