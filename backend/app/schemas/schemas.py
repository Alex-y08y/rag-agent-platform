"""Pydantic schemas for API request/response validation."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Generic ───────────────────────────────────────────────
class ApiResponse(BaseModel):
    success: bool = True
    message: str = ""
    data: Any = None


class PaginatedResponse(BaseModel):
    items: list[Any]
    total: int
    page: int = 1
    page_size: int = 20


# ── Knowledge Base ────────────────────────────────────────
class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=1024)


class KnowledgeBaseOut(BaseModel):
    id: str
    name: str
    description: str
    document_count: int
    chunk_count: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Document ──────────────────────────────────────────────
class DocumentOut(BaseModel):
    id: str
    knowledge_base_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    created_at: datetime
    error_message: str | None = None

    model_config = {"from_attributes": True}


class DocumentDetailOut(DocumentOut):
    chunks: list["ChunkOut"] = []


class ChunkOut(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str
    page: int | None = None
    section: str | None = None
    title: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = {}
    embedding_status: str = "PENDING"

    model_config = {"from_attributes": True}


DocumentDetailOut.model_rebuild()


# ── Chat ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    conversation_id: str | None = None
    knowledge_base_id: str | None = None
    query: str = Field(..., min_length=1)
    stream: bool = False
    rag_version: Literal[
        "baseline", "hybrid", "hybrid_rerank", "hybrid_rerank_rewrite"
    ] = "hybrid_rerank_rewrite"
    top_k: int = Field(default=10, ge=1, le=50)


class Citation(BaseModel):
    document_id: str
    filename: str
    page: int | None = None
    section: str | None = None
    title: str | None = None
    chunk_id: str
    content: str
    score: float = 0.0


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    citations: list[Citation] = []
    retrieved_docs: int = 0
    latency_ms: int = 0
    token_usage: dict[str, int] = {}
    tool_calls: list[dict[str, Any]] = []


class ChatStreamEvent(BaseModel):
    """SSE event for streaming chat."""
    type: Literal[
        "status", "tool_call", "retrieval", "token", "citation", "done", "error"
    ]
    data: Any = None


# ── Retrieval ─────────────────────────────────────────────
class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    knowledge_base_id: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    filters: dict[str, Any] = {}
    rag_version: Literal[
        "baseline", "hybrid", "hybrid_rerank", "hybrid_rerank_rewrite"
    ] = "hybrid_rerank_rewrite"


class RetrievedDoc(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    source: str | None = None
    page: int | None = None
    section: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = {}


class RetrievalResponse(BaseModel):
    query: str
    rewritten_query: str | None = None
    results: list[RetrievedDoc]
    total: int
    latency_ms: int = 0


# ── Evaluation ────────────────────────────────────────────
class EvaluationRunRequest(BaseModel):
    dataset_name: str = "rag_eval"
    rag_version: Literal[
        "baseline", "hybrid", "hybrid_rerank", "hybrid_rerank_rewrite"
    ] = "hybrid_rerank_rewrite"
    name: str | None = None
    max_samples: int | None = None


class EvaluationMetrics(BaseModel):
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0
    accuracy: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    avg_latency_ms: float = 0.0
    total_samples: int = 0


class EvaluationTaskOut(BaseModel):
    id: str
    name: str
    dataset_name: str
    rag_version: str
    status: str
    total_samples: int
    completed_samples: int
    metrics: dict[str, Any] = {}
    created_at: datetime

    model_config = {"from_attributes": True}


class EvaluationSampleOut(BaseModel):
    id: str
    task_id: str
    question: str
    ground_truth: str | None = None
    expected_sources: list[str] = []
    category: str | None = None
    difficulty: str | None = None
    retrieved_sources: list[str] = []
    generated_answer: str | None = None
    metrics: dict[str, Any] = {}
    latency_ms: int = 0
    analysis: str | None = None

    model_config = {"from_attributes": True}


# ── Agent Trace ───────────────────────────────────────────
class AgentTraceOut(BaseModel):
    id: str
    conversation_id: str
    agent_run_id: str
    node: str
    input_data: dict[str, Any] = {}
    output_data: dict[str, Any] = {}
    tool_name: str | None = None
    latency_ms: int = 0
    status: str
    token_usage: dict[str, Any] = {}
    error_message: str | None = None
    sequence: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Conversation ──────────────────────────────────────────
class ConversationOut(BaseModel):
    id: str
    title: str
    knowledge_base_id: str | None = None
    message_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    citations: list[Any] = []
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Health ────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    services: dict[str, str] = {}
