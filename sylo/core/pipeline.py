"""Pipeline context manager — the core primitive of Sylo.

Usage:
    async with sylo.pipeline("my-pipeline", version="1.0") as pipe:
        result = await my_agent_function(inputs)

The context manager handles:
- Generating a unique execution_id (UUID4) per run
- Recording start/end times
- Catching exceptions and marking executions as FAILED
- Persisting execution records to the configured storage backend
- Auto-resumption from the most recent failed execution
- Cost savings report on completion (dev mode)
"""

from __future__ import annotations

import contextvars
import logging
import traceback
import uuid
from datetime import datetime, timezone
from types import TracebackType
from typing import Any

from sylo.config import get_config
from sylo.core.context import Context
from sylo.exceptions import SyloStorageError
from sylo.models import AuditEvent, ExecutionRecord, ExecutionStatus, TokenCost
from sylo.storage import SyloStorage, get_storage

logger = logging.getLogger("sylo")

# Context variable so @sylo.step can access the current pipeline
_current_pipeline: contextvars.ContextVar[Pipeline | None] = contextvars.ContextVar(
    "_current_pipeline", default=None
)


class Pipeline:
    """Async context manager that wraps a pipeline execution.

    Creates an ExecutionRecord on entry, updates it on exit (success
    or failure), and persists everything to the configured storage backend.

    In development mode, storage errors are logged but never crash the
    user's pipeline. In production mode, they raise SyloStorageError.

    Attributes:
        name: The pipeline name.
        version: The pipeline version string.
        execution_id: Unique ID for this execution (set on __aenter__).
        record: The full ExecutionRecord (set on __aenter__).
        context: The Context object available to all steps.
        resume_from: Optional execution ID to resume from.
        metadata: Arbitrary user-defined metadata for this execution.
    """

    def __init__(
        self,
        name: str,
        version: str = "0.0.0",
        metadata: dict[str, Any] | None = None,
        resume_from: str | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.execution_id: str = ""
        self.record: ExecutionRecord | None = None
        self.context: Context | None = None
        self.resume_from: str | None = resume_from
        self.metadata: dict[str, Any] = metadata or {}

        self._storage: SyloStorage | None = None
        self._sequence_counter: int = 0
        self._step_counter: int = 0
        self._step_results: list[Any] = []  # StepResult instances
        self._context_token: contextvars.Token | None = None

    async def __aenter__(self) -> Pipeline:
        """Start the pipeline execution.

        Generates an execution ID, creates the initial ExecutionRecord,
        initializes the Context, and auto-detects resumption if applicable.
        """
        config = get_config()
        self._storage = get_storage(config)

        self.execution_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Count how many times this pipeline has run today
        run_number = await self._count_todays_runs()

        self.record = ExecutionRecord(
            execution_id=self.execution_id,
            pipeline_name=self.name,
            pipeline_version=self.version,
            status=ExecutionStatus.RUNNING,
            started_at=now,
            token_cost=TokenCost(),
            metadata=self.metadata,
        )

        # Initialize the Context for step functions
        self.context = Context(
            execution_id=self.execution_id,
            pipeline_name=self.name,
            run_number=run_number,
            metadata=self.metadata,
        )

        # Auto-detect resumption: find last FAILED/RUNNING execution
        if self.resume_from is None:
            await self._auto_detect_resumption()

        # Set the current pipeline context var so @sylo.step can access it
        self._context_token = _current_pipeline.set(self)

        # Persist the initial record
        await self._safe_storage_op(
            self._storage.save_execution, self.record
        )

        # Emit PIPELINE_STARTED audit event
        resuming = self.resume_from is not None
        await self._emit_audit_event(
            event_type="PIPELINE_STARTED",
            data={
                "pipeline_name": self.name,
                "pipeline_version": self.version,
                "environment": config.environment,
                "resuming_from": self.resume_from,
            },
        )

        if resuming:
            logger.info(
                "Sylo pipeline started: %s (execution: %s, resuming from: %s)",
                self.name,
                self.execution_id[:8],
                self.resume_from[:8] if self.resume_from else "",
            )
        else:
            logger.info(
                "Sylo pipeline started: %s (execution: %s)",
                self.name,
                self.execution_id[:8],
            )

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Complete the pipeline execution.

        On success: marks COMPLETED, records end time, prints cost report.
        On failure: marks FAILED, records error details, then re-raises.

        Returns:
            False — exceptions are always re-raised after recording.
        """
        assert self.record is not None
        assert self._storage is not None

        # Reset the context var
        if self._context_token is not None:
            _current_pipeline.reset(self._context_token)
            self._context_token = None

        now = datetime.now(timezone.utc)
        self.record.completed_at = now

        if exc_val is not None:
            # Pipeline failed
            self.record.status = ExecutionStatus.FAILED
            self.record.error = (
                f"{type(exc_val).__name__}: {exc_val}"
            )

            await self._emit_audit_event(
                event_type="PIPELINE_FAILED",
                data={
                    "error_type": type(exc_val).__name__,
                    "error_message": str(exc_val),
                    "traceback": traceback.format_exception(
                        exc_type, exc_val, exc_tb
                    ),
                },
            )

            logger.error(
                "Sylo pipeline failed: %s (execution: %s) - %s",
                self.name,
                self.execution_id[:8],
                exc_val,
            )
        else:
            # Pipeline succeeded
            self.record.status = ExecutionStatus.COMPLETED

            await self._emit_audit_event(
                event_type="PIPELINE_COMPLETED",
                data={
                    "duration_seconds": (
                        now - self.record.started_at
                    ).total_seconds(),
                    "token_cost": self.record.token_cost.model_dump(),
                },
            )

            logger.info(
                "Sylo pipeline completed: %s (execution: %s)",
                self.name,
                self.execution_id[:8],
            )

        # Persist the final record
        await self._safe_storage_op(
            self._storage.save_execution, self.record
        )

        # Print cost savings report in dev mode
        config = get_config()
        if config.is_development and self._step_results:
            self._print_cost_report()

        # Never swallow exceptions — re-raise so the user sees them
        return False

    def _print_cost_report(self) -> None:
        """Print the cost savings report to console (dev mode only).

        Shows step completion stats, token usage, costs, and savings
        from checkpoint resumption.
        """
        from sylo.core.checkpoint import StepResult

        results: list[StepResult] = self._step_results

        steps_completed = sum(1 for r in results if not r.was_cached and r.output is not None)
        steps_skipped = sum(1 for r in results if r.was_cached)
        steps_retried = sum(1 for r in results if r.retry_count > 0)

        total_tokens = sum(
            r.token_usage.total_tokens for r in results if r.token_usage
        )
        total_cost = sum(
            r.token_usage.estimated_cost_usd for r in results if r.token_usage and not r.was_cached
        )
        cost_saved = sum(r.cost_saved_usd for r in results if r.was_cached)
        time_saved_ms = sum(r.time_saved_ms for r in results if r.was_cached)

        lines = [
            "",
            "[SUCCESS] Sylo execution complete",
            f"  Steps: {steps_completed} completed, {steps_skipped} skipped, {steps_retried} retried",
            f"  Tokens: {total_tokens:,} total | Est. cost: ${total_cost:.3f}",
        ]

        if steps_skipped > 0 and cost_saved > 0:
            # Find the first cached step for the message
            cached_steps = [r for r in results if r.was_cached]
            first_cached = cached_steps[0] if cached_steps else None
            time_saved_s = time_saved_ms / 1000

            if first_cached:
                lines.append(
                    f'  Resumed from checkpoint: step "{first_cached.step_name}" '
                    f"(saved ${cost_saved:.3f}, {time_saved_s:.0f}s)"
                )

        print("\n".join(lines))

    async def _auto_detect_resumption(self) -> None:
        """Auto-detect if we should resume from a previous failed execution.

        Queries storage for the most recent FAILED or RUNNING execution
        of this pipeline. If found, loads its checkpoints so that
        @sylo.step can skip already-completed steps.
        """
        if self._storage is None:
            return

        try:
            recent = await self._safe_storage_op(
                self._storage.list_executions, self.name, 5
            )
            if not recent:
                return

            # Find the most recent FAILED execution
            for execution in recent:
                if execution.status in (
                    ExecutionStatus.FAILED,
                    ExecutionStatus.RUNNING,
                ):
                    self.resume_from = execution.execution_id
                    logger.info(
                        "Auto-resuming from execution %s",
                        execution.execution_id[:8],
                    )

                    # Pre-load checkpoints from the failed execution into
                    # a new execution — the step decorator will find them
                    # by querying for the resume_from execution_id
                    break
        except Exception:
            # Non-critical — if auto-detection fails, just start fresh
            logger.debug("Auto-resumption detection failed, starting fresh")

    async def _count_todays_runs(self) -> int:
        """Count how many times this pipeline has run today.

        Returns:
            Run number (1-indexed) for today.
        """
        if self._storage is None:
            return 1

        try:
            recent = await self._safe_storage_op(
                self._storage.list_executions, self.name, 100
            )
            if not recent:
                return 1

            today = datetime.now(timezone.utc).date()
            todays_runs = sum(
                1 for r in recent if r.started_at.date() == today
            )
            return todays_runs + 1
        except Exception:
            return 1

    async def _emit_audit_event(
        self,
        event_type: str,
        step_name: str | None = None,
        data: dict[str, Any] | None = None,
        duration_ms: int = 0,
    ) -> None:
        """Create and persist an audit event.

        Args:
            event_type: One of the defined audit event types.
            step_name: The step this event relates to, if any.
            data: Event-specific payload.
            duration_ms: Duration in milliseconds, if applicable.
        """
        assert self._storage is not None

        self._sequence_counter += 1
        event = AuditEvent(
            execution_id=self.execution_id,
            pipeline_name=self.name,
            event_type=event_type,
            step_name=step_name,
            sequence_number=self._sequence_counter,
            data=data or {},
            duration_ms=duration_ms,
        )

        await self._safe_storage_op(
            self._storage.append_audit_event,
            self.execution_id,
            event,
        )

    async def _safe_storage_op(self, func: Any, *args: Any) -> Any:
        """Execute a storage operation with environment-aware error handling.

        In development mode: catches and logs errors, never crashes.
        In production mode: lets SyloStorageError propagate.
        """
        config = get_config()
        try:
            return await func(*args)
        except Exception as exc:
            if config.is_production:
                raise SyloStorageError(
                    f"Storage operation failed: {exc}"
                ) from exc
            else:
                logger.warning(
                    "Sylo storage operation failed (non-fatal in %s mode): %s",
                    config.environment,
                    exc,
                )
                return None
