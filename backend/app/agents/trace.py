"""Agent Trace: record execution steps for debugging and visualization."""
from __future__ import annotations

import time
from typing import Any

from app.core.database import SessionLocal
from app.core.logging import get_logger
from app.models.models import AgentTrace as AgentTraceModel

logger = get_logger(__name__)


class TraceRecorder:
    """Record agent execution traces to PostgreSQL.

    Each node execution is recorded with input/output, latency, status,
    and token usage. Does NOT record hidden chain-of-thought.
    """

    def __init__(self, conversation_id: str, agent_run_id: str) -> None:
        self.conversation_id = conversation_id
        self.agent_run_id = agent_run_id
        self.sequence = 0
        self.traces: list[dict[str, Any]] = []

    def record(
        self,
        node: str,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        tool_name: str | None = None,
        latency_ms: int = 0,
        status: str = "success",
        token_usage: dict[str, int] | None = None,
        error_message: str | None = None,
    ) -> None:
        """Record a trace entry."""
        self.sequence += 1
        trace = {
            "conversation_id": self.conversation_id,
            "agent_run_id": self.agent_run_id,
            "node": node,
            "input_data": input_data or {},
            "output_data": output_data or {},
            "tool_name": tool_name,
            "latency_ms": latency_ms,
            "status": status,
            "token_usage": token_usage or {},
            "error_message": error_message,
            "sequence": self.sequence,
        }
        self.traces.append(trace)
        logger.info(
            "Trace: node=%s, status=%s, latency=%dms, seq=%d",
            node, status, latency_ms, self.sequence,
        )

    def flush(self) -> None:
        """Persist all traces to database."""
        if not self.traces:
            return
        try:
            db = SessionLocal()
            for t in self.traces:
                db.add(AgentTraceModel(**t))
            db.commit()
            db.close()
            logger.info("Flushed %d traces for run %s", len(self.traces), self.agent_run_id)
        except Exception as exc:
            logger.error("Failed to flush traces: %s", exc)

    def get_traces(self) -> list[dict[str, Any]]:
        """Get in-memory traces (for streaming response)."""
        return self.traces.copy()


def get_traces_by_conversation(conversation_id: str) -> list[dict[str, Any]]:
    """Retrieve all traces for a conversation, grouped by agent_run_id."""
    try:
        db = SessionLocal()
        traces = (
            db.query(AgentTraceModel)
            .filter(AgentTraceModel.conversation_id == conversation_id)
            .order_by(AgentTraceModel.agent_run_id, AgentTraceModel.sequence)
            .all()
        )
        db.close()
        return [
            {
                "id": t.id,
                "conversation_id": t.conversation_id,
                "agent_run_id": t.agent_run_id,
                "node": t.node,
                "input_data": t.input_data,
                "output_data": t.output_data,
                "tool_name": t.tool_name,
                "latency_ms": t.latency_ms,
                "status": t.status,
                "token_usage": t.token_usage,
                "error_message": t.error_message,
                "sequence": t.sequence,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in traces
        ]
    except Exception as exc:
        logger.error("Failed to get traces: %s", exc)
        return []
