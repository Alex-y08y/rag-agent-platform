"""SQLAlchemy ORM models for the RAG Agent Platform."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


# ── Timestamps mixin ──────────────────────────────────────
class TimestampMixin:
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


# ── Users ─────────────────────────────────────────────────
class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(String(32), primary_key=True, default=_uuid)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    is_active = Column(Boolean, default=True)
    role = Column(String(32), default="user")

    conversations = relationship("Conversation", back_populates="user")


# ── Knowledge Bases ───────────────────────────────────────
class KnowledgeBase(Base, TimestampMixin):
    __tablename__ = "knowledge_bases"

    id = Column(String(32), primary_key=True, default=_uuid)
    name = Column(String(128), nullable=False, index=True)
    description = Column(Text, default="")
    user_id = Column(String(32), ForeignKey("users.id"), nullable=True)
    is_public = Column(Boolean, default=False)
    document_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    status = Column(String(32), default="active")  # active | indexing | error

    documents = relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")


# ── Documents ─────────────────────────────────────────────
class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id = Column(String(32), primary_key=True, default=_uuid)
    knowledge_base_id = Column(
        String(32), ForeignKey("knowledge_bases.id"), nullable=False, index=True
    )
    filename = Column(String(256), nullable=False)
    file_type = Column(String(16), nullable=False)  # pdf | docx | txt | md | csv | xlsx
    file_size = Column(Integer, default=0)  # bytes
    file_hash = Column(String(64), index=True)
    storage_path = Column(String(512), nullable=False)
    status = Column(String(32), default="UPLOADING", index=True)
    # UPLOADING | PARSING | CHUNKING | EMBEDDING | INDEXING | SUCCESS | FAILED
    chunk_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSON, default=dict)

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_doc_kb_status", "knowledge_base_id", "status"),
    )


# ── Document Chunks ───────────────────────────────────────
class DocumentChunk(Base, TimestampMixin):
    __tablename__ = "document_chunks"

    id = Column(String(32), primary_key=True, default=_uuid)
    document_id = Column(String(32), ForeignKey("documents.id"), nullable=False, index=True)
    knowledge_base_id = Column(String(32), index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page = Column(Integer, nullable=True)
    section = Column(String(256), nullable=True)
    title = Column(String(256), nullable=True)
    source = Column(String(256), nullable=True)  # filename
    metadata_ = Column("metadata", JSON, default=dict)
    embedding_status = Column(String(32), default="PENDING")  # PENDING | DONE | FAILED
    vector_id = Column(String(64), nullable=True, index=True)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunk_doc_index", "document_id", "chunk_index"),
    )


# ── Conversations & Messages ──────────────────────────────
class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"

    id = Column(String(32), primary_key=True, default=_uuid)
    user_id = Column(String(32), ForeignKey("users.id"), nullable=True)
    title = Column(String(256), default="新对话")
    knowledge_base_id = Column(String(32), nullable=True, index=True)
    message_count = Column(Integer, default=0)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    traces = relationship("AgentTrace", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base, TimestampMixin):
    __tablename__ = "messages"

    id = Column(String(32), primary_key=True, default=_uuid)
    conversation_id = Column(
        String(32), ForeignKey("conversations.id"), nullable=False, index=True
    )
    role = Column(String(16), nullable=False)  # user | assistant | system
    content = Column(Text, nullable=False)
    citations = Column(JSON, default=list)
    metadata_ = Column("metadata", JSON, default=dict)
    token_usage = Column(JSON, default=dict)
    latency_ms = Column(Integer, nullable=True)

    conversation = relationship("Conversation", back_populates="messages")


# ── Tool Calls ────────────────────────────────────────────
class ToolCall(Base, TimestampMixin):
    __tablename__ = "tool_calls"

    id = Column(String(32), primary_key=True, default=_uuid)
    conversation_id = Column(String(32), index=True)
    agent_run_id = Column(String(32), index=True)
    tool_name = Column(String(64), nullable=False)
    input_args = Column(JSON, default=dict)
    output = Column(JSON, default=dict)
    status = Column(String(32), default="success")  # success | error
    latency_ms = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)


# ── Evaluation ────────────────────────────────────────────
class EvaluationTask(Base, TimestampMixin):
    __tablename__ = "evaluation_tasks"

    id = Column(String(32), primary_key=True, default=_uuid)
    name = Column(String(128), nullable=False)
    dataset_name = Column(String(128), nullable=False)
    rag_version = Column(String(64), nullable=False)
    # baseline | hybrid | hybrid_rerank | hybrid_rerank_rewrite
    status = Column(String(32), default="PENDING", index=True)
    # PENDING | RUNNING | COMPLETED | FAILED
    total_samples = Column(Integer, default=0)
    completed_samples = Column(Integer, default=0)
    metrics = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)

    samples = relationship(
        "EvaluationSample", back_populates="task", cascade="all, delete-orphan"
    )


class EvaluationSample(Base, TimestampMixin):
    __tablename__ = "evaluation_samples"

    id = Column(String(32), primary_key=True, default=_uuid)
    task_id = Column(String(32), ForeignKey("evaluation_tasks.id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=True)
    expected_sources = Column(JSON, default=list)
    category = Column(String(64), nullable=True)
    difficulty = Column(String(32), nullable=True)
    retrieved_sources = Column(JSON, default=list)
    retrieved_scores = Column(JSON, default=list)
    generated_answer = Column(Text, nullable=True)
    metrics = Column(JSON, default=dict)
    # recall@k, precision@k, mrr, accuracy, faithfulness, relevancy
    latency_ms = Column(Integer, default=0)
    analysis = Column(Text, nullable=True)

    task = relationship("EvaluationTask", back_populates="samples")


# ── Agent Traces ──────────────────────────────────────────
class AgentTrace(Base, TimestampMixin):
    __tablename__ = "agent_traces"

    id = Column(String(32), primary_key=True, default=_uuid)
    conversation_id = Column(
        String(32), ForeignKey("conversations.id"), nullable=False, index=True
    )
    agent_run_id = Column(String(32), nullable=False, index=True)
    node = Column(String(64), nullable=False)
    # analyze_intent | rewrite_query | plan | retrieve | tool_call | observe | verify | final_answer
    input_data = Column(JSON, default=dict)
    output_data = Column(JSON, default=dict)
    tool_name = Column(String(64), nullable=True)
    latency_ms = Column(Integer, default=0)
    status = Column(String(32), default="success")  # success | error | skipped
    token_usage = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)
    sequence = Column(Integer, default=0)

    conversation = relationship("Conversation", back_populates="traces")

    __table_args__ = (
        Index("ix_trace_conv_run", "conversation_id", "agent_run_id"),
    )


# ── Demo SQL Tables (for SQLQueryTool) ────────────────────
class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256), nullable=False)
    category = Column(String(64))
    price = Column(Float, default=0.0)
    stock = Column(Integer, default=0)
    description = Column(Text)


class SalesOrder(Base):
    __tablename__ = "sales_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    customer_id = Column(Integer, ForeignKey("customers.id"))
    quantity = Column(Integer, default=1)
    amount = Column(Float, default=0.0)
    region = Column(String(64))
    order_date = Column(DateTime, default=datetime.utcnow)


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    email = Column(String(128))
    region = Column(String(64))
    tier = Column(String(32), default="standard")


class MarketingCampaign(Base):
    __tablename__ = "marketing_campaigns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    budget = Column(Float, default=0.0)
    description = Column(Text)
