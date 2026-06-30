"""CrewAI integration for Sylo SDK.

Provides a wrapper around CrewAI's Crew class that adds Sylo checkpointing
at the per-task level, token tracking, and crash-resume support.

Key exports:
    - ``SyloCrew``           — Wraps a CrewAI Crew with per-task Sylo checkpointing
    - ``SyloCrewAIError``    — Integration-specific error class

Usage:
    from crewai import Agent, Task, Crew
    from sylo.integrations.crewai import SyloCrew

    researcher = Agent(
        role="Researcher",
        goal="Research the topic thoroughly",
        backstory="Expert researcher",
        llm="groq/openai/gpt-oss-20b",
    )
    task1 = Task(description="Research quantum computing", agent=researcher,
                 expected_output="A research summary")

    crew = SyloCrew(
        agents=[researcher],
        tasks=[task1],
        pipeline_name="crew-research",
    )

    async with sylo.pipeline("crew-research") as pipe:
        result = await crew.run(pipe.context)

Note:
    Requires ``crewai`` as an optional dependency.
    Install with: ``pip install sylo-sdk[crewai]``

Known Limitations:
    - CrewAI's Crew.kickoff() is a monolithic execution. True per-task
      checkpointing is achieved using CrewAI's ``task_callback`` hook to
      record outputs after each task completes. However, on resume, we
      cannot skip individual tasks within a kickoff() call — instead we
      re-run the crew but inject cached context so the LLM can produce
      results faster. For full per-task skip, we use a task-by-task
      execution strategy where each task is run in its own mini-crew.
    - Token usage is available at the crew level via CrewOutput.token_usage,
      not per-task. We record aggregate usage.
    - CrewAI runs tasks synchronously by default. The wrapper uses
      asyncio.to_thread() for async compatibility with Sylo's pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import sylo
from sylo.core.context import Context
from sylo.exceptions import SyloError

logger = logging.getLogger("sylo.integrations.crewai")


def _patch_crewai_litellm_compatibility() -> None:
    """Ensure compatibility between CrewAI and non-native LiteLLM providers (e.g., Groq).

    CrewAI injects a 'cache_breakpoint' flag into message dicts. When passed to
    OpenAI-compatible endpoints like Groq via LiteLLM, this raises a 400 Bad Request
    unless stripped.
    """
    try:
        import crewai.llms.cache
        crewai.llms.cache.mark_cache_breakpoint = lambda message: message
    except Exception:
        pass

    try:
        import litellm
        litellm.drop_params = True
        orig_completion = litellm.completion
        if not getattr(orig_completion, "_sylo_patched", False):
            def patched_completion(*args: Any, **kwargs: Any) -> Any:
                messages = kwargs.get("messages")
                if messages and isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict):
                            msg.pop("cache_breakpoint", None)
                return orig_completion(*args, **kwargs)
            patched_completion._sylo_patched = True
            litellm.completion = patched_completion
    except Exception:
        pass


class SyloCrewAIError(SyloError):
    """Raised when a CrewAI operation fails within a Sylo wrapper.

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


class SyloCrew:
    """Wraps a CrewAI Crew with per-task Sylo checkpointing.

    Instead of running the entire crew as a single unit, SyloCrew executes
    each task individually, checkpointing after each one. On resume,
    already-completed tasks are skipped and their cached outputs are
    injected as context for subsequent tasks.

    Args:
        agents: List of CrewAI Agent instances.
        tasks: List of CrewAI Task instances (executed in order).
        pipeline_name: Name for the Sylo pipeline.
        crew_kwargs: Additional keyword arguments passed to CrewAI's Crew constructor.

    Example:
        crew = SyloCrew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            pipeline_name="content-pipeline",
        )

        async with sylo.pipeline("content-pipeline") as pipe:
            result = await crew.run(pipe.context)
    """

    def __init__(
        self,
        agents: list[Any],
        tasks: list[Any],
        pipeline_name: str = "crewai-pipeline",
        **crew_kwargs: Any,
    ) -> None:
        self._agents = agents
        self._tasks = tasks
        self._pipeline_name = pipeline_name
        self._crew_kwargs = crew_kwargs
        self._task_outputs: dict[str, Any] = {}

    @property
    def pipeline_name(self) -> str:
        """The Sylo pipeline name for this crew."""
        return self._pipeline_name

    async def run(self, ctx: Context, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the crew with per-task Sylo checkpointing.

        Each task is executed individually. If a checkpoint exists for a task
        from a previous failed execution, the cached output is loaded and
        the task is skipped.

        Args:
            ctx: The Sylo context from the active pipeline.
            inputs: Optional inputs dict passed to CrewAI's kickoff().

        Returns:
            Dict containing the final crew output and per-task results.

        Raises:
            SyloCrewAIError: If the crew execution fails.
        """
        from sylo.core.pipeline import _current_pipeline

        pipeline = _current_pipeline.get(None)
        if pipeline is None:
            # Running outside a pipeline — just kickoff directly
            return await self._run_full_crew(inputs)

        results: dict[str, Any] = {}
        accumulated_context = ""

        for i, task in enumerate(self._tasks):
            task_step_name = self._get_task_step_name(task, i)
            cached = await self._check_task_checkpoint(
                pipeline, ctx, task_step_name
            )

            if cached is not None:
                # Task was already completed — load from checkpoint
                results[task_step_name] = cached
                raw_output = cached.get("raw_output", "")
                accumulated_context += f"\n\nPrevious task result:\n{raw_output}"
                continue

            # Task needs execution
            await pipeline._emit_audit_event(
                event_type="STEP_STARTED",
                step_name=task_step_name,
                data={
                    "step_index": pipeline._step_counter,
                    "framework": "crewai",
                    "task_description": (
                        task.description[:100] if hasattr(task, "description") else ""
                    ),
                },
            )

            step_index = pipeline._step_counter
            pipeline._step_counter += 1
            step_started = datetime.now(timezone.utc)

            try:
                task_output = await self._run_single_task(
                    task, accumulated_context, inputs
                )
            except Exception as exc:
                step_ended = datetime.now(timezone.utc)
                duration_ms = int(
                    (step_ended - step_started).total_seconds() * 1000
                )

                # Save failed checkpoint
                from sylo.models import Checkpoint, CheckpointStatus

                failed_checkpoint = Checkpoint(
                    execution_id=pipeline.execution_id,
                    step_name=task_step_name,
                    step_index=step_index,
                    status=CheckpointStatus.FAILED,
                    input_hash="",
                    output={},
                    started_at=step_started,
                    completed_at=step_ended,
                    duration_ms=duration_ms,
                    retry_count=0,
                )

                storage = pipeline._storage
                if storage is not None:
                    await pipeline._safe_storage_op(
                        storage.save_checkpoint, failed_checkpoint
                    )

                await pipeline._emit_audit_event(
                    event_type="STEP_FAILED",
                    step_name=task_step_name,
                    duration_ms=duration_ms,
                    data={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )

                from sylo.core.checkpoint import StepResult

                step_result = StepResult(
                    step_name=task_step_name,
                    output=None,
                    duration_ms=duration_ms,
                    was_cached=False,
                    token_usage=None,
                    retry_count=0,
                )
                pipeline._step_results.append(step_result)

                raise SyloCrewAIError(
                    message=f'CrewAI task "{task_step_name}" failed.',
                    execution_id=pipeline.execution_id,
                    step_name=task_step_name,
                    original_error=exc,
                ) from exc

            step_ended = datetime.now(timezone.utc)
            duration_ms = int(
                (step_ended - step_started).total_seconds() * 1000
            )

            # Build output dict
            raw_output = str(task_output) if task_output is not None else ""
            output_dict = {
                "raw_output": raw_output,
                "task_description": (
                    task.description[:200] if hasattr(task, "description") else ""
                ),
            }

            # Check for manually recorded token usage
            token_usage = None
            if (
                hasattr(ctx, "_recorded_token_usage")
                and ctx._recorded_token_usage is not None
            ):
                token_usage = ctx._recorded_token_usage
                ctx._recorded_token_usage = None

            # Save completed checkpoint
            from sylo.models import Checkpoint, CheckpointStatus, TokenUsage

            completed_checkpoint = Checkpoint(
                execution_id=pipeline.execution_id,
                step_name=task_step_name,
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

            storage = pipeline._storage
            if storage is not None:
                await pipeline._safe_storage_op(
                    storage.save_checkpoint, completed_checkpoint
                )

            # Emit completion events
            if token_usage is not None:
                await pipeline._emit_audit_event(
                    event_type="TOKEN_USAGE_RECORDED",
                    step_name=task_step_name,
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
                step_name=task_step_name,
                duration_ms=duration_ms,
                data={
                    "step_index": step_index,
                    "has_token_usage": token_usage is not None,
                    "framework": "crewai",
                },
            )
            await pipeline._emit_audit_event(
                event_type="CHECKPOINT_SAVED",
                step_name=task_step_name,
                data={
                    "checkpoint_id": completed_checkpoint.checkpoint_id,
                },
            )

            ctx.previous_outputs[task_step_name] = output_dict
            results[task_step_name] = output_dict
            accumulated_context += f"\n\nPrevious task result:\n{raw_output}"

            from sylo.core.checkpoint import StepResult

            step_result = StepResult(
                step_name=task_step_name,
                output=output_dict,
                duration_ms=duration_ms,
                was_cached=False,
                token_usage=token_usage,
                retry_count=0,
            )
            pipeline._step_results.append(step_result)

            logger.info(
                "CrewAI task '%s' completed in %dms",
                task_step_name,
                duration_ms,
            )

        return results

    async def _check_task_checkpoint(
        self,
        pipeline: Any,
        ctx: Context,
        task_step_name: str,
    ) -> dict[str, Any] | None:
        """Check if a task has a completed checkpoint to resume from.

        Returns the checkpoint output dict if found, None otherwise.
        """
        from sylo.models import CheckpointStatus

        storage = pipeline._storage
        if storage is None:
            return None

        existing = await pipeline._safe_storage_op(
            storage.get_checkpoint, pipeline.execution_id, task_step_name
        )
        if existing is None and pipeline.resume_from is not None:
            existing = await pipeline._safe_storage_op(
                storage.get_checkpoint, pipeline.resume_from, task_step_name
            )
            if existing is not None:
                existing.execution_id = pipeline.execution_id
                await pipeline._safe_storage_op(
                    storage.save_checkpoint, existing
                )

        if (
            existing is not None
            and existing.status == CheckpointStatus.COMPLETED
        ):
            logger.info(
                "CrewAI task '%s' loaded from checkpoint (skipped)",
                task_step_name,
            )

            await pipeline._emit_audit_event(
                event_type="STEP_SKIPPED",
                step_name=task_step_name,
                data={
                    "reason": "checkpoint_hit",
                    "checkpoint_id": existing.checkpoint_id,
                    "original_duration_ms": existing.duration_ms,
                },
            )
            await pipeline._emit_audit_event(
                event_type="CHECKPOINT_LOADED",
                step_name=task_step_name,
                data={
                    "checkpoint_id": existing.checkpoint_id,
                },
            )

            ctx.previous_outputs[task_step_name] = existing.output

            from sylo.core.checkpoint import StepResult

            cost_saved = (
                existing.token_usage.estimated_cost_usd
                if existing.token_usage
                else 0.0
            )
            step_result = StepResult(
                step_name=task_step_name,
                output=existing.output,
                duration_ms=0,
                was_cached=True,
                token_usage=existing.token_usage,
                retry_count=0,
                cost_saved_usd=cost_saved,
                time_saved_ms=existing.duration_ms,
            )
            pipeline._step_results.append(step_result)

            return existing.output

        return None

    async def _run_single_task(
        self,
        task: Any,
        accumulated_context: str,
        inputs: dict[str, Any] | None,
    ) -> Any:
        """Run a single CrewAI task by creating a mini-crew.

        This approach gives us true per-task execution control, allowing
        Sylo to checkpoint after each individual task rather than waiting
        for the entire crew to complete.

        Args:
            task: The CrewAI Task to execute.
            accumulated_context: Concatenated output from previous tasks.
            inputs: Optional inputs dict.

        Returns:
            The task output (CrewOutput or string).
        """
        try:
            from crewai import Crew
        except ImportError:
            raise ImportError(
                "The crewai package is required for this integration. "
                "Install with: pip install crewai"
            )

        # Inject accumulated context into the task description if there's
        # prior context to pass along
        original_description = task.description
        if accumulated_context.strip():
            task.description = (
                f"{original_description}\n\n"
                f"Context from previous tasks:\n{accumulated_context}"
            )

        try:
            # Create a mini-crew with just this one task
            agent = task.agent
            mini_crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=False,
                **{k: v for k, v in self._crew_kwargs.items() if k not in ("agents", "tasks", "verbose")},
            )

            # Apply compatibility patch before executing
            _patch_crewai_litellm_compatibility()

            # Run in a thread since CrewAI's kickoff() is synchronous
            result = await asyncio.to_thread(
                mini_crew.kickoff,
                inputs=inputs or {},
            )

            return result
        finally:
            # Restore original description
            task.description = original_description

    async def _run_full_crew(self, inputs: dict[str, Any] | None) -> dict[str, Any]:
        """Run the full crew without Sylo wrapping (for outside-pipeline usage)."""
        try:
            from crewai import Crew
        except ImportError:
            raise ImportError(
                "The crewai package is required for this integration. "
                "Install with: pip install crewai"
            )

        _patch_crewai_litellm_compatibility()

        crew = Crew(
            agents=self._agents,
            tasks=self._tasks,
            verbose=False,
            **self._crew_kwargs,
        )

        result = await asyncio.to_thread(
            crew.kickoff,
            inputs=inputs or {},
        )

        return {"raw_output": str(result)}

    def _get_task_step_name(self, task: Any, index: int) -> str:
        """Generate a unique step name for a CrewAI task.

        Uses the task's description (truncated) or falls back to index.
        """
        if hasattr(task, "description") and task.description:
            # Create a slug from the first few words of the description
            words = task.description.split()[:4]
            slug = "-".join(w.lower().strip(".,;:!?") for w in words if w.strip())
            return f"crewai-task-{index}-{slug}"
        return f"crewai-task-{index}"
