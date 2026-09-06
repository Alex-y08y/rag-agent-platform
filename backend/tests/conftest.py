"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables before importing app
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_DB", "rag_test")
os.environ.setdefault("POSTGRES_USER", "rag_user")
os.environ.setdefault("POSTGRES_PASSWORD", "rag_password")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("MILVUS_HOST", "localhost")
os.environ.setdefault("ELASTICSEARCH_URL", "http://localhost:9200")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("RERANKER_DEVICE", "cpu")


@pytest.fixture
def sample_text():
    return "这是一段测试文本。用于测试文本分块功能。包含多个句子。每个句子都有意义。"


@pytest.fixture
def sample_pages():
    from app.services.parser import ParsedPage
    return [
        ParsedPage(content="第一页内容。这是关于公司制度的介绍。", page=1, section="第一章", title="总则"),
        ParsedPage(content="第二页内容。这是关于考勤制度的详细说明。", page=2, section="第二章", title="考勤"),
    ]


@pytest.fixture
def sample_chunks():
    return [
        {"chunk_id": "c1", "document_id": "d1", "content": "差旅报销标准是500元", "source": "travel_policy.md", "page": 3, "score": 0.9},
        {"chunk_id": "c2", "document_id": "d1", "content": "住宿标准一线城市500元", "source": "travel_policy.md", "page": 3, "score": 0.85},
        {"chunk_id": "c3", "document_id": "d2", "content": "员工手册规定工作时间9点到18点", "source": "employee_handbook.md", "page": 1, "score": 0.7},
    ]
