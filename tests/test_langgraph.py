"""Tests for the LangGraph integration (Brief 06).

Uses mocks since langgraph is an optional dependency.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest

from sylo.config import reset_config, set_config, SyloConfig
from sylo.integrations.langgraph import (
    LuroGraph,
    LuroTokenTracker,
    SyloGraph,
    SyloLangGraphError,
    LuroLangGraphError,
)


@pytest.fixture(autouse=True)
def reset_sylo():
    """Reset global config before each test."""
    reset_config()
    yield
    reset_config()


class TestLuroTokenTracker:
    """Tests for the LuroTokenTracker callback handler."""

    def test_initial_state(self):
        tracker = LuroTokenTracker()
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.total_tokens == 0
        assert tracker.model is None

    def test_on_llm_end_extracts_tokens(self):
        tracker = LuroTokenTracker()

        # Simulate a LangChain LLMResult
        response = MagicMock()
        response.llm_output = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }

        tracker.on_llm_end(response)
        assert tracker.prompt_tokens == 100
        assert tracker.completion_tokens == 50
        assert tracker.total_tokens == 150

    def test_on_llm_end_accumulates_tokens(self):
        tracker = LuroTokenTracker()

        for i in range(3):
            response = MagicMock()
            response.llm_output = {
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                }
            }
            tracker.on_llm_end(response)

        assert tracker.prompt_tokens == 300
        assert tracker.completion_tokens == 150
        assert tracker.total_tokens == 450

    def test_on_llm_start_extracts_model_name(self):
        tracker = LuroTokenTracker()

        serialized = {"kwargs": {"model_name": "gpt-4o"}}
        tracker.on_llm_start(serialized, ["test prompt"])
        assert tracker.model == "gpt-4o"

    def test_on_llm_end_handles_missing_usage(self):
        tracker = LuroTokenTracker()

        response = MagicMock()
        response.llm_output = {}

        tracker.on_llm_end(response)
        assert tracker.total_tokens == 0

    def test_on_llm_end_handles_none_output(self):
        tracker = LuroTokenTracker()

        response = MagicMock()
        response.llm_output = None

        tracker.on_llm_end(response)
        assert tracker.total_tokens == 0

    def test_reset(self):
        tracker = LuroTokenTracker()

        response = MagicMock()
        response.llm_output = {
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }
        tracker.on_llm_end(response)
        assert tracker.total_tokens == 150

        tracker.reset()
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.total_tokens == 0


class TestSyloGraphError:
    """Tests for the SyloLangGraphError exception."""

    def test_basic_error(self):
        err = SyloLangGraphError("Node failed")
        assert "Node failed" in str(err)

    def test_error_with_resume_instructions(self):
        err = SyloLangGraphError(
            message='Node "summarize" failed after 2 retries.',
            execution_id="abc123",
            step_name="summarize",
            original_error=ValueError("API timeout"),
        )
        msg = str(err)
        assert "summarize" in msg
        assert "abc123" in msg
        assert "Resume with:" in msg
        assert "API timeout" in msg

    def test_backwards_compat_alias(self):
        assert LuroLangGraphError is SyloLangGraphError


class TestSyloGraph:
    """Tests for the SyloGraph wrapper."""

    def test_creation(self):
        mock_graph = MagicMock()
        graph = SyloGraph(mock_graph, pipeline_name="test-pipe", version="2.0")
        assert graph.pipeline_name == "test-pipe"
        assert graph._version == "2.0"

    def test_backwards_compat_alias(self):
        assert LuroGraph is SyloGraph

    def test_add_node_wraps_function(self):
        mock_graph = MagicMock()
        graph = SyloGraph(mock_graph, pipeline_name="test-pipe")

        def my_node(state):
            return {"result": "done"}

        graph.add_node("my-node", my_node)

        # Check the underlying graph received a wrapped function
        mock_graph.add_node.assert_called_once()
        call_args = mock_graph.add_node.call_args
        assert call_args[0][0] == "my-node"
        # The second arg should be a wrapper, not the original function
        wrapped = call_args[0][1]
        assert wrapped.__name__ == "my_node"  # functools.wraps preserves name

    def test_add_edge_passthrough(self):
        mock_graph = MagicMock()
        graph = SyloGraph(mock_graph)

        graph.add_edge("node-a", "node-b")
        mock_graph.add_edge.assert_called_once_with("node-a", "node-b")

    def test_add_conditional_edges_passthrough(self):
        mock_graph = MagicMock()
        graph = SyloGraph(mock_graph)

        def router(state):
            return "branch-a"

        graph.add_conditional_edges("source", router, {"branch-a": "target"})
        mock_graph.add_conditional_edges.assert_called_once()

    def test_compile_passthrough(self):
        mock_graph = MagicMock()
        graph = SyloGraph(mock_graph)

        result = graph.compile(checkpointer="memory")
        mock_graph.compile.assert_called_once_with(checkpointer="memory")

    def test_getattr_proxy(self):
        mock_graph = MagicMock()
        mock_graph.custom_property = "test_value"
        graph = SyloGraph(mock_graph)

        assert graph.custom_property == "test_value"

    def test_sync_node_wrapper_calls_function(self):
        """Test that sync node wrappers call the original function."""
        mock_graph = MagicMock()
        graph = SyloGraph(mock_graph)

        call_count = 0

        def my_sync_node(state):
            nonlocal call_count
            call_count += 1
            return {"processed": True}

        graph.add_node("sync-node", my_sync_node)

        # Get the wrapped function
        wrapped = mock_graph.add_node.call_args[0][1]

        # Call it outside pipeline context — should just work
        result = wrapped({"input": "data"})
        assert call_count == 1
        assert result == {"processed": True}

    def test_sync_node_wrapper_raises_sylo_error_on_failure(self):
        """Test that sync node failures are wrapped in SyloLangGraphError."""
        mock_graph = MagicMock()
        graph = SyloGraph(mock_graph)

        def failing_node(state):
            raise RuntimeError("API timeout")

        graph.add_node("failing-node", failing_node)
        wrapped = mock_graph.add_node.call_args[0][1]

        with pytest.raises(SyloLangGraphError, match="failing-node"):
            wrapped({"input": "data"})

    @pytest.mark.asyncio
    async def test_async_node_wrapper_calls_function(self):
        """Test that async node wrappers call the original function."""
        mock_graph = MagicMock()
        graph = SyloGraph(mock_graph)

        async def my_async_node(state):
            return {"processed": True}

        graph.add_node("async-node", my_async_node)
        wrapped = mock_graph.add_node.call_args[0][1]

        result = await wrapped({"input": "data"})
        assert result == {"processed": True}

    @pytest.mark.asyncio
    async def test_async_node_wrapper_raises_sylo_error_on_failure(self):
        """Test that async node failures are wrapped in SyloLangGraphError."""
        mock_graph = MagicMock()
        graph = SyloGraph(mock_graph)

        async def failing_node(state):
            raise RuntimeError("Connection refused")

        graph.add_node("failing-async", failing_node)
        wrapped = mock_graph.add_node.call_args[0][1]

        with pytest.raises(SyloLangGraphError, match="failing-async"):
            await wrapped({"input": "data"})

    def test_token_tracker_default(self):
        mock_graph = MagicMock()
        graph = SyloGraph(mock_graph)
        assert isinstance(graph.token_tracker, LuroTokenTracker)

    def test_custom_token_tracker(self):
        mock_graph = MagicMock()
        tracker = LuroTokenTracker()
        graph = SyloGraph(mock_graph, token_tracker=tracker)
        assert graph.token_tracker is tracker
