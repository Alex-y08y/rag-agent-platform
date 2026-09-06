"""Tests for FastAPI endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_status(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data


class TestChatEndpoint:
    def test_chat_requires_query(self, client):
        response = client.post("/api/chat", json={"query": ""})
        assert response.status_code == 422  # Pydantic validation

    def test_chat_validation(self, client):
        response = client.post("/api/chat", json={})
        assert response.status_code == 422


class TestKnowledgeBaseEndpoint:
    def test_list_kb(self, client):
        response = client.get("/api/knowledge-bases")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_kb_validation(self, client):
        response = client.post("/api/knowledge-bases", json={"name": ""})
        assert response.status_code == 422


class TestDocumentsEndpoint:
    def test_list_documents(self, client):
        response = client.get("/api/documents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestRetrievalEndpoint:
    def test_search_validation(self, client):
        response = client.post("/api/retrieval/search", json={"query": ""})
        assert response.status_code == 422


class TestEvaluationEndpoint:
    def test_list_tasks(self, client):
        response = client.get("/api/evaluation/tasks")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestAgentTraceEndpoint:
    def test_get_traces(self, client):
        response = client.get("/api/agent/traces/test-conv-id")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
