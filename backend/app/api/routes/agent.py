"""Agent Trace API route."""
from __future__ import annotations

from fastapi import APIRouter

from app.agents.trace import get_traces_by_conversation
from app.core.logging import get_logger
from app.schemas.schemas import AgentTraceOut

logger = get_logger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/traces/{conversation_id}", response_model=list[AgentTraceOut])
def get_agent_traces(conversation_id: str):
    """Get agent execution traces for a conversation."""
    traces = get_traces_by_conversation(conversation_id)
    return [AgentTraceOut(**t) for t in traces]
