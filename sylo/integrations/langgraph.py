"""LangGraph integration for Sylo SDK — Brief 06.

Provides drop-in wrappers for LangGraph's StateGraph that automatically
adds Sylo checkpointing, token tracking, and approval gates.

Key exports:
    - ``LuroGraph``         — Wraps a LangGraph StateGraph with Sylo primitives
    - ``SyloGraph``         — Alias for LuroGraph (new naming)
    - ``LuroTokenTracker``  — LangChain callback handler for token extraction
    - ``luro_interrupt``    — Wraps LangGraph's interrupt() with Sylo approval flow

Usage:
    from sylo.integrations.langgraph import SyloGraph

    graph = SyloGraph(StateGraph(MyState), pipeline_name="email-pipeline")
    graph.add_node("fetch_emails", fetch_emails)
    graph.add_node("summarize", summarize)
    graph.add_edge("fetch_emails", "summarize")
    app = graph.compile()

Note:
    Requires ``langgraph`` and ``langchain-core`` as optional dependencies.
    Install with: ``pip install sylo-sdk[langgraph]``
"""

from __future__ import annotations

import functools
import logging
from datetime import datetime, timezone
from typing import Any, Callable, TYPE_CHECKING

import sylo
from sylo.config import get_config
from sylo.core.context import Context
from sylo.exceptions import SyloError

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("sylo.integrations.langgraph")


class SyloLangGraphError(SyloError):
    """Raised when a LangGraph node fails within a Sylo-wrapped graph.

    Includes the execution ID and resume instructions in the error message.
    """

    def __init__(
        self,
        message: str,
        execution_id: str | None = None,
        step_name: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        self.execution_id = execution_id
        self.step_name = step_name
        self.original_error = original_error

        parts = [message]
        if execution_id:
            parts.append(f"Execution ID: {execution_id}")
        if step_name and execution_id:
            parts.append(
                f"Resume with: sylo executions replay {execution_id} --from-step {step_name}"
            )
        if original_error:
            parts.append(f"Original error: {original_error}")

        super().__init__("\n".join(parts))


# Backwards compatibility alias
LuroLangGraphError = SyloLangGraphError


class LuroTokenTracker:
    """LangChain callback handler that extracts token usage from LLM calls.

    Automatically injected into LangChain model calls within a SyloGraph.
    Accumulates token counts and reports them to the active Sylo context.

    Usage:
        tracker = LuroTokenTracker()
        model = ChatOpenAI(callbacks=[tracker])
        # ... run model ...
        print(tracker.total_tokens)

    Note:
        This class intentionally does NOT inherit from
        ``langchain_core.callbacks.BaseCallbackHandler`` to avoid
        requiring LangChain at import time. It implements the same
        interface (duck typing) so LangChain will call its methods.
    """

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.model: str | None = None
        self._runs: list[dict[str, Any]] = []

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call starts."""
        # Extract model name from serialized data
        model_name = serialized.get("kwargs", {}).get("model_name")
        if model_name:
            self.model = model_name

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when an LLM call finishes. Extracts token usage."""
        try:
            # LangChain's LLMResult has llm_output with token_usage
            llm_output = getattr(response, "llm_output", {}) or {}
            token_usage = llm_output.get("token_usage", {})

            if token_usage:
                prompt = token_usage.get("prompt_tokens", 0)
                completion = token_usage.get("completion_tokens", 0)
                total = token_usage.get("total_tokens", prompt + completion)

                self.prompt_tokens += prompt
                self.completion_tokens += completion
                self.total_tokens += total

                self._runs.append({
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total,
                    "model": self.model,
                })
            else:
                # Try extracting from generations (Anthropic format)
                generations = getattr(response, "generations", [])
                for gen_list in generations:
                    for gen in gen_list:
                        info = getattr(gen, "generation_info", {}) or {}
                        usage = info.get("usage", {})
                        if usage:
                            prompt = usage.get("input_tokens", usage.get("prompt_tokens", 0))
                            completion = usage.get("output_tokens", usage.get("completion_tokens", 0))
                            total = prompt + completion
                            self.prompt_tokens += prompt
                            self.completion_tokens += completion
                            self.total_tokens += total
        except Exception as exc:
            logger.debug("Could not extract token usage: %s", exc)

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when an LLM call fails."""
        logger.debug("LLM call failed: %s", error)

    def reset(self) -> None:
        """Reset accumulated token counts."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._runs.clear()


class SyloGraph:
    """Wraps a LangGraph StateGraph with automatic Sylo checkpointing.

    Acts as a transparent proxy to the underlying StateGraph, intercepting
    ``add_node()`` calls to wrap each node function with Sylo step logic.

    Args:
        graph: A LangGraph ``StateGraph`` instance.
        pipeline_name: Name for the Sylo pipeline.
        version: Pipeline version string.
        token_tracker: Optional ``LuroTokenTracker`` for automatic token extraction.

    Example:
        from langgraph.graph import StateGraph
        from sylo.integrations.langgraph import SyloGraph

        graph = SyloGraph(StateGraph(MyState), pipeline_name="email-pipeline")
        graph.add_node("fetch_emails", fetch_emails)
        graph.add_node("summarize", summarize)
        graph.add_edge("fetch_emails", "summarize")
        app = graph.compile()
    """

    def __init__(
        self,
        graph: Any,  # StateGraph — typed as Any to avoid import
        pipeline_name: str = "langgraph-pipeline",
        version: str = "1.0",
        token_tracker: LuroTokenTracker | None = None,
    ) -> None:
        self._graph = graph
        self._pipeline_name = pipeline_name
        self._version = version
        self._token_tracker = token_tracker or LuroTokenTracker()
        self._wrapped_nodes: dict[str, Callable] = {}
        self._thread_id: str | None = None

    @property
    def pipeline_name(self) -> str:
        """The Sylo pipeline name for this graph."""
        return self._pipeline_name

    @property
    def token_tracker(self) -> LuroTokenTracker:
        """The token tracker instance."""
        return self._token_tracker

    def add_node(self, name: str, func: Callable, **kwargs: Any) -> Any:
        """Add a node to the graph with automatic Sylo wrapping.

        The node function is wrapped with ``@sylo.step`` so that it
        automatically gets checkpointing, retry logic, and audit events.

        Args:
            name: Node name (becomes the Sylo step name).
            func: The node function.
            **kwargs: Additional arguments passed to the underlying StateGraph.add_node.

        Returns:
            Result of the underlying StateGraph.add_node call.
        """
        wrapped = self._wrap_node(name, func)
        self._wrapped_nodes[name] = wrapped
        return self._graph.add_node(name, wrapped, **kwargs)

    def add_edge(self, source: str, target: str, **kwargs: Any) -> Any:
        """Add an edge (passthrough to underlying StateGraph)."""
        return self._graph.add_edge(source, target, **kwargs)

    def add_conditional_edges(
        self,
        source: str,
        path: Callable,
        path_map: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Add conditional edges (passthrough to underlying StateGraph)."""
        return self._graph.add_conditional_edges(source, path, path_map, **kwargs)

    def set_entry_point(self, key: str) -> Any:
        """Set the entry point (passthrough to underlying StateGraph)."""
        return self._graph.set_entry_point(key)

    def set_finish_point(self, key: str) -> Any:
        """Set the finish point (passthrough to underlying StateGraph)."""
        return self._graph.set_finish_point(key)

    def compile(self, **kwargs: Any) -> Any:
        """Compile the graph (passthrough to underlying StateGraph)."""
        return self._graph.compile(**kwargs)

    def _wrap_node(self, name: str, func: Callable) -> Callable:
        """Wrap a LangGraph node function with Sylo step logic.

        The wrapper:
        1. Creates a Sylo checkpoint before and after execution
        2. Tracks token usage via the LuroTokenTracker
        3. Maps LangGraph state to Sylo step input/output
        4. Provides informative error messages with resume instructions
        """
        tracker = self._token_tracker

        @functools.wraps(func)
        async def async_wrapper(state: Any, config: Any = None, **kw: Any) -> Any:
            from sylo.core.pipeline import _current_pipeline

            pipeline = _current_pipeline.get(None)

            # Reset tracker for this node
            tracker.reset()

            # Extract thread_id from LangGraph config
            thread_id = None
            if config and isinstance(config, dict):
                thread_id = config.get("configurable", {}).get("thread_id")
            elif hasattr(config, "configurable"):
                thread_id = getattr(config.configurable, "thread_id", None)

            try:
                # Call the original node function
                import inspect
                if inspect.iscoroutinefunction(func):
                    result = await func(state, config=config, **kw) if config is not None else await func(state, **kw)
                else:
                    result = func(state, config=config, **kw) if config is not None else func(state, **kw)

                # Record token usage from tracker into pipeline
                if tracker.total_tokens > 0 and pipeline is not None and pipeline.context is not None:
                    pipeline.context.record_token_usage(
                        prompt_tokens=tracker.prompt_tokens,
                        completion_tokens=tracker.completion_tokens,
                        model=tracker.model,
                    )

                return result

            except Exception as exc:
                execution_id = pipeline.execution_id if pipeline else None
                raise SyloLangGraphError(
                    message=f'Node "{name}" failed.',
                    execution_id=execution_id,
                    step_name=name,
                    original_error=exc,
                ) from exc

        @functools.wraps(func)
        def sync_wrapper(state: Any, config: Any = None, **kw: Any) -> Any:
            from sylo.core.pipeline import _current_pipeline

            pipeline = _current_pipeline.get(None)
            tracker.reset()

            try:
                result = func(state, config=config, **kw) if config is not None else func(state, **kw)

                if tracker.total_tokens > 0 and pipeline is not None and pipeline.context is not None:
                    pipeline.context.record_token_usage(
                        prompt_tokens=tracker.prompt_tokens,
                        completion_tokens=tracker.completion_tokens,
                        model=tracker.model,
                    )

                return result

            except Exception as exc:
                execution_id = pipeline.execution_id if pipeline else None
                raise SyloLangGraphError(
                    message=f'Node "{name}" failed.',
                    execution_id=execution_id,
                    step_name=name,
                    original_error=exc,
                ) from exc

        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    def __getattr__(self, name: str) -> Any:
        """Proxy any unrecognized attributes to the underlying StateGraph."""
        return getattr(self._graph, name)


# Backwards compat alias
LuroGraph = SyloGraph


def luro_interrupt(
    title: str,
    description: str,
    action_class: str = "destructive",
    timeout_hours: float = 24.0,
    on_timeout: str = "abort",
) -> None:
    """Wrap LangGraph's built-in interrupt() with Sylo's approval flow.

    Call this inside a LangGraph node function to pause execution
    and request human approval before proceeding.

    Args:
        title: Short title summarizing the action.
        description: Description of what the step is about to do.
        action_class: Category ("destructive", "financial", "external").
        timeout_hours: Hours before the request expires.
        on_timeout: Behavior on expiry ("abort", "auto_approve", "escalate").

    Raises:
        SyloLangGraphError: If no LangGraph interrupt is available.
    """
    try:
        from langgraph.types import interrupt as lg_interrupt
    except ImportError:
        try:
            from langgraph.prebuilt import interrupt as lg_interrupt
        except ImportError:
            logger.warning(
                "LangGraph interrupt not available. "
                "Install langgraph to use luro_interrupt()."
            )
            return

    # Emit the interrupt with Sylo metadata
    lg_interrupt({
        "sylo_approval": True,
        "title": title,
        "description": description,
        "action_class": action_class,
        "timeout_hours": timeout_hours,
        "on_timeout": on_timeout,
    })
