"""Tests for Agent graph and state."""
from __future__ import annotations

import pytest

from app.agents.state import AgentState
from app.agents.trace import TraceRecorder


class TestAgentState:
    def test_state_creation(self):
        state: AgentState = {
            "query": "test query",
            "conversation_id": "conv1",
            "history": [],
            "retry_count": 0,
        }
        assert state["query"] == "test query"
        assert state["retry_count"] == 0

    def test_state_optional_fields(self):
        state: AgentState = {"query": "test"}
        assert "intent" not in state
        assert "answer" not in state


class TestTraceRecorder:
    def test_record_trace(self):
        recorder = TraceRecorder("conv1", "run1")
        recorder.record(
            node="analyze_intent",
            input_data={"query": "test"},
            output_data={"intent": "test intent"},
            latency_ms=100,
        )
        assert len(recorder.traces) == 1
        assert recorder.traces[0]["node"] == "analyze_intent"
        assert recorder.traces[0]["sequence"] == 1

    def test_trace_sequence_increments(self):
        recorder = TraceRecorder("conv1", "run1")
        recorder.record(node="step1")
        recorder.record(node="step2")
        recorder.record(node="step3")
        assert recorder.traces[0]["sequence"] == 1
        assert recorder.traces[1]["sequence"] == 2
        assert recorder.traces[2]["sequence"] == 3

    def test_get_traces_returns_copy(self):
        recorder = TraceRecorder("conv1", "run1")
        recorder.record(node="step1")
        traces = recorder.get_traces()
        traces.append({"node": "fake"})
        assert len(recorder.traces) == 1

    def test_trace_status_default(self):
        recorder = TraceRecorder("conv1", "run1")
        recorder.record(node="step1")
        assert recorder.traces[0]["status"] == "success"

    def test_trace_error_recording(self):
        recorder = TraceRecorder("conv1", "run1")
        recorder.record(node="step1", status="error", error_message="test error")
        assert recorder.traces[0]["status"] == "error"
        assert recorder.traces[0]["error_message"] == "test error"


class TestAgentRunner:
    def test_singleton(self):
        from app.agents.graph import get_agent_runner
        runner1 = get_agent_runner()
        runner2 = get_agent_runner()
        assert runner1 is runner2

    def test_max_iterations_config(self):
        from app.agents.graph import AgentRunner
        runner = AgentRunner.__new__(AgentRunner)
        # Just verify the class exists and has expected attributes
        assert hasattr(AgentRunner, "run")
