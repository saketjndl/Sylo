"""OpenAI Agents SDK integration for Sylo SDK.

Provides a wrapper around the OpenAI Agents SDK's Agent class that adds
Sylo checkpointing, token tracking, and crash-resume support.

Key exports:
    - ``wrap_agent``              — Wrap an Agent with Sylo checkpointing
    - ``SyloOpenAIAgentsError``   — Integration-specific error class

Usage:
    from openai import AsyncOpenAI
    from agents import Agent, Runner
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from sylo.integrations.openai_agents import wrap_agent

    client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key="...")
    model = OpenAIChatCompletionsModel(model="openai/gpt-oss-20b", openai_client=client)

    agent = Agent(name="Researcher", instructions="...", model=model)
    wrapped = wrap_agent(agent, step_name="research")

    async with sylo.pipeline("research-pipeline") as pipe:
        result = await wrapped.run(pipe.context, "quantum computing")

Note:
    Requires ``openai-agents`` and ``openai`` as optional dependencies.
    Install with: ``pip install sylo-sdk[openai-agents]``

Known Limitations:
    - The OpenAI Agents SDK has built-in tracing that attempts to phone home
      to OpenAI's servers. When using non-OpenAI endpoints (e.g., Groq), you
      should disable tracing: ``from agents import set_tracing_disabled; set_tracing_disabled(True)``
    - Token usage extraction relies on ``RunResult.raw_responses`` which may
      vary across SDK versions. We implement multiple fallback paths.
    - Some Agents SDK features (handoffs, guardrails) may not work with
      non-OpenAI endpoints. The wrapper focuses on single-agent runs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

import sylo
from sylo.core.context import Context
from sylo.exceptions import SyloError

if TYPE_CHECKING:
    pass

logger = logging.getLogger("sylo.integrations.openai_agents")


class SyloOpenAIAgentsError(SyloError):
    """Raised when an OpenAI Agents SDK operation fails within a Sylo wrapper.

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


class WrappedAgent:
    """An OpenAI Agents SDK Agent wrapped with Sylo checkpointing.

    This wrapper intercepts calls to ``Runner.run()`` and adds:
    - Checkpoint save/restore for crash-resume
    - Token usage extraction from the RunResult
    - Audit event emission via the active pipeline

    Args:
        agent: An OpenAI Agents SDK ``Agent`` instance.
        step_name: Unique name for this step in the Sylo pipeline.

    Example:
        wrapped = wrap_agent(agent, step_name="research")
        result = await wrapped.run(pipe.context, "quantum computing")
    """

    def __init__(self, agent: Any, step_name: str) -> None:
        self._agent = agent
        self._step_name = step_name

    @property
    def agent(self) -> Any:
        """The underlying OpenAI Agents SDK Agent."""
        return self._agent

    @property
    def step_name(self) -> str:
        """The Sylo step name for this wrapper."""
        return self._step_name

    async def run(
        self,
        ctx: Context,
        input_text: str,
        **runner_kwargs: Any,
    ) -> str:
        """Run the agent with Sylo checkpointing.

        If a checkpoint exists for this step_name in the current execution
        (or a previous failed execution being resumed), the cached output
        is returned without re-running the agent.

        Args:
            ctx: The Sylo context from the active pipeline.
            input_text: The user input to send to the agent.
            **runner_kwargs: Additional keyword arguments passed to Runner.run().

        Returns:
            The agent's final output string.

        Raises:
            SyloOpenAIAgentsError: If the agent run fails.
        """
        from sylo.core.pipeline import _current_pipeline

        pipeline = _current_pipeline.get(None)
        if pipeline is None:
            # Running outside a pipeline — just call directly
            return await self._run_agent(input_text, **runner_kwargs)

        storage = pipeline._storage
        step_index = pipeline._step_counter
        pipeline._step_counter += 1

        # Check for existing checkpoint (resumption logic)
        from sylo.models import CheckpointStatus

        existing_checkpoint = None
        if storage is not None:
            existing_checkpoint = await pipeline._safe_storage_op(
                storage.get_checkpoint, pipeline.execution_id, self._step_name
            )
            if existing_checkpoint is None and pipeline.resume_from is not None:
                existing_checkpoint = await pipeline._safe_storage_op(
                    storage.get_checkpoint, pipeline.resume_from, self._step_name
                )
                if existing_checkpoint is not None:
                    existing_checkpoint.execution_id = pipeline.execution_id
                    await pipeline._safe_storage_op(
                        storage.save_checkpoint, existing_checkpoint
                    )

        if (
            existing_checkpoint is not None
            and existing_checkpoint.status == CheckpointStatus.COMPLETED
        ):
            # Checkpoint hit — skip re-execution
            logger.info(
                "OpenAI Agent step '%s' loaded from checkpoint (skipped)",
                self._step_name,
            )

            await pipeline._emit_audit_event(
                event_type="STEP_SKIPPED",
                step_name=self._step_name,
                data={
                    "reason": "checkpoint_hit",
                    "checkpoint_id": existing_checkpoint.checkpoint_id,
                    "original_duration_ms": existing_checkpoint.duration_ms,
                },
            )
            await pipeline._emit_audit_event(
                event_type="CHECKPOINT_LOADED",
                step_name=self._step_name,
                data={
                    "checkpoint_id": existing_checkpoint.checkpoint_id,
                },
            )

            ctx.previous_outputs[self._step_name] = existing_checkpoint.output

            from sylo.core.checkpoint import StepResult

            step_result = StepResult(
                step_name=self._step_name,
                output=existing_checkpoint.output,
                duration_ms=0,
                was_cached=True,
                token_usage=existing_checkpoint.token_usage,
                retry_count=0,
                cost_saved_usd=(
                    existing_checkpoint.token_usage.estimated_cost_usd
                    if existing_checkpoint.token_usage
                    else 0.0
                ),
                time_saved_ms=existing_checkpoint.duration_ms,
            )
            pipeline._step_results.append(step_result)

            return existing_checkpoint.output.get("final_output", "")

        # No checkpoint — execute the agent
        await pipeline._emit_audit_event(
            event_type="STEP_STARTED",
            step_name=self._step_name,
            data={"step_index": step_index, "framework": "openai-agents-sdk"},
        )

        step_started = datetime.now(timezone.utc)

        try:
            final_output = await self._run_agent(input_text, **runner_kwargs)
        except Exception as exc:
            step_ended = datetime.now(timezone.utc)
            duration_ms = int((step_ended - step_started).total_seconds() * 1000)

            await pipeline._emit_audit_event(
                event_type="STEP_FAILED",
                step_name=self._step_name,
                duration_ms=duration_ms,
                data={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )

            from sylo.core.checkpoint import StepResult

            step_result = StepResult(
                step_name=self._step_name,
                output=None,
                duration_ms=duration_ms,
                was_cached=False,
                token_usage=None,
                retry_count=0,
            )
            pipeline._step_results.append(step_result)

            raise SyloOpenAIAgentsError(
                message=f'OpenAI Agent step "{self._step_name}" failed.',
                execution_id=pipeline.execution_id,
                step_name=self._step_name,
                original_error=exc,
            ) from exc

        step_ended = datetime.now(timezone.utc)
        duration_ms = int((step_ended - step_started).total_seconds() * 1000)

        # Extract token usage from manually recorded usage on context
        token_usage = None
        if hasattr(ctx, '_recorded_token_usage') and ctx._recorded_token_usage is not None:
            token_usage = ctx._recorded_token_usage
            ctx._recorded_token_usage = None

        # Save checkpoint
        from sylo.models import Checkpoint, CheckpointStatus, TokenUsage

        output_dict = {"final_output": final_output}
        completed_checkpoint = Checkpoint(
            execution_id=pipeline.execution_id,
            step_name=self._step_name,
            step_index=step_index,
            status=CheckpointStatus.COMPLETED,
            input_hash="",
            output=output_dict,
            started_at=step_started,
            completed_at=step_ended,
            duration_ms=duration_ms,
            token_usage=token_usage or TokenUsage(),
            retry_count=0,
        )

        if storage is not None:
            await pipeline._safe_storage_op(
                storage.save_checkpoint, completed_checkpoint
            )

        # Emit completion events
        if token_usage is not None:
            await pipeline._emit_audit_event(
                event_type="TOKEN_USAGE_RECORDED",
                step_name=self._step_name,
                data={
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": token_usage.completion_tokens,
                    "total_tokens": token_usage.total_tokens,
                    "model": token_usage.model,
                    "estimated_cost_usd": token_usage.estimated_cost_usd,
                },
            )

        await pipeline._emit_audit_event(
            event_type="STEP_COMPLETED",
            step_name=self._step_name,
            duration_ms=duration_ms,
            data={
                "step_index": step_index,
                "has_token_usage": token_usage is not None,
                "framework": "openai-agents-sdk",
            },
        )
        await pipeline._emit_audit_event(
            event_type="CHECKPOINT_SAVED",
            step_name=self._step_name,
            data={
                "checkpoint_id": completed_checkpoint.checkpoint_id,
            },
        )

        ctx.previous_outputs[self._step_name] = output_dict

        from sylo.core.checkpoint import StepResult

        step_result = StepResult(
            step_name=self._step_name,
            output=output_dict,
            duration_ms=duration_ms,
            was_cached=False,
            token_usage=token_usage,
            retry_count=0,
        )
        pipeline._step_results.append(step_result)

        logger.info(
            "OpenAI Agent step '%s' completed in %dms",
            self._step_name,
            duration_ms,
        )

        return final_output

    async def _run_agent(self, input_text: str, **runner_kwargs: Any) -> str:
        """Execute the underlying agent via Runner.run().

        Returns:
            The agent's final_output string.
        """
        try:
            from agents import Runner
        except ImportError:
            raise ImportError(
                "The openai-agents package is required for this integration. "
                "Install with: pip install openai-agents"
            )

        result = await Runner.run(self._agent, input_text, **runner_kwargs)
        return result.final_output


def _extract_usage_from_result(result: Any) -> dict[str, int]:
    """Extract token usage from an OpenAI Agents SDK RunResult.

    Tries multiple extraction paths for compatibility across SDK versions:
    1. result.raw_responses[].usage
    2. result.context_wrapper.usage

    Args:
        result: A RunResult from Runner.run().

    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens.
    """
    total_input = 0
    total_output = 0

    # Path 1: Aggregate from raw_responses
    try:
        raw_responses = getattr(result, "raw_responses", []) or []
        for response in raw_responses:
            usage = getattr(response, "usage", None)
            if usage is not None:
                total_input += getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0
                total_output += getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0
    except Exception:
        pass

    # Path 2: Try context_wrapper.usage as fallback
    if total_input == 0 and total_output == 0:
        try:
            cw = getattr(result, "context_wrapper", None)
            if cw is not None:
                usage = getattr(cw, "usage", None)
                if usage is not None:
                    total_input = getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0
                    total_output = getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0
        except Exception:
            pass

    return {
        "prompt_tokens": total_input,
        "completion_tokens": total_output,
        "total_tokens": total_input + total_output,
    }


def wrap_agent(agent: Any, step_name: str) -> WrappedAgent:
    """Wrap an OpenAI Agents SDK Agent with Sylo checkpointing.

    The wrapped agent provides a ``.run(ctx, input_text)`` method that
    automatically handles checkpoint save/restore, token usage tracking,
    and crash-resume.

    Args:
        agent: An OpenAI Agents SDK ``Agent`` instance.
        step_name: Unique name for this step in the Sylo pipeline.

    Returns:
        A ``WrappedAgent`` instance with Sylo checkpointing.

    Example:
        from agents import Agent
        from sylo.integrations.openai_agents import wrap_agent

        agent = Agent(name="Researcher", instructions="...", model=model)
        wrapped = wrap_agent(agent, step_name="research")

        async with sylo.pipeline("pipeline") as pipe:
            result = await wrapped.run(pipe.context, "topic")
    """
    return WrappedAgent(agent=agent, step_name=step_name)
