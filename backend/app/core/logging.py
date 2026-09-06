"""Structured logging with request / conversation / agent context."""
from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings

# Context variables for distributed tracing
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
conversation_id_var: ContextVar[str] = ContextVar("conversation_id", default="")
agent_run_id_var: ContextVar[str] = ContextVar("agent_run_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:16]
    request_id_var.set(rid)
    return rid


def new_agent_run_id() -> str:
    aid = uuid.uuid4().hex[:16]
    agent_run_id_var.set(aid)
    return aid


class JsonFormatter(logging.Formatter):
    """JSON log formatter with context fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
            "conversation_id": conversation_id_var.get(),
            "agent_run_id": agent_run_id_var.get(),
            "user_id": user_id_var.get(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Extra fields
        for key in ("latency", "token_usage", "tool_call", "retrieval", "status"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging() -> None:
    """Initialize logging for the application."""
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(JsonFormatter())
    root.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(
        log_dir / "app.log", encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    # Reduce noise from third-party
    for noisy in ("uvicorn.access", "httpx", "httpcore", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
