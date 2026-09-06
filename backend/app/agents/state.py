"""Agent state definition for LangGraph."""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    # Input
    query: str
    conversation_id: str
    knowledge_base_id: str | None
    history: list[dict[str, str]]

    # Processing
    intent: str
    rewritten_query: str
    plan: list[dict[str, Any]]
    current_step_index: int

    # Tool execution
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    retrieved_docs: list[dict[str, Any]]
    sql_results: dict[str, Any]

    # Generation
    context: str
    answer: str
    citations: list[dict[str, Any]]

    # Verification
    verification_passed: bool
    verification_reason: str
    retry_count: int

    # Metadata
    agent_run_id: str
    token_usage: dict[str, int]
    latencies: dict[str, int]
    error: str | None
