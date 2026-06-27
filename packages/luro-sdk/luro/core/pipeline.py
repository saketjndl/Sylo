"""Pipeline context manager — the core primitive of Luro.

Usage:
    async with luro.pipeline("my-pipeline", version="1.0") as pipe:
        result = await my_agent_function(inputs)

The context manager handles:
- Generating a unique execution_id (UUID4) per run
- Recording start/end times
- Catching exceptions and marking executions as FAILED
- Persisting execution records to the configured storage backend
"""

from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime, timezone
from types import TracebackType
from typing import Any

from luro.config import get_config
from luro.exceptions import LuroStorageError
from luro.models import AuditEvent, ExecutionRecord, ExecutionStatus, TokenCost
from luro.storage import LuroStorage, get_storage

logger = logging.getLogger("luro")


class Pipeline:
    """Async context manager that wraps a pipeline execution.

    Creates an ExecutionRecord on entry, updates it on exit (success
    or failure), and persists everything to the configured storage backend.

    In development mode, storage errors are logged but never crash the
    user's pipeline. In production mode, they raise LuroStorageError.

    Attributes:
        name: The pipeline name.
        version: The pipeline version string.
        execution_id: Unique ID for this execution (set on __aenter__).
        record: The full ExecutionRecord (set on __aenter__).
        resume_from: Optional checkpoint ID to resume from (Brief 02).
        metadata: Arbitrary user-defined metadata for this execution.
    """

    def __init__(
        self,
        name: str,
        version: str = "0.0.0",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.execution_id: str = ""
        self.record: ExecutionRecord | None = None
        self.resume_from: str | None = None
        self.metadata: dict[str, Any] = metadata or {}

        self._storage: LuroStorage | None = None
        self._sequence_counter: int = 0

    async def __aenter__(self) -> Pipeline:
        """Start the pipeline execution.

        Generates an execution ID, creates the initial ExecutionRecord,
        and persists it to storage.
        """
        config = get_config()
        self._storage = get_storage(config)

        self.execution_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        self.record = ExecutionRecord(
            execution_id=self.execution_id,
            pipeline_name=self.name,
            pipeline_version=self.version,
            status=ExecutionStatus.RUNNING,
            started_at=now,
            token_cost=TokenCost(),
            metadata=self.metadata,
        )

        # Persist the initial record
        await self._safe_storage_op(
            self._storage.save_execution, self.record
        )

        # Emit PIPELINE_STARTED audit event
        await self._emit_audit_event(
            event_type="PIPELINE_STARTED",
            data={
                "pipeline_name": self.name,
                "pipeline_version": self.version,
                "environment": config.environment,
            },
        )

        logger.info(
            "Luro pipeline started: %s (execution: %s)",
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

        On success: marks COMPLETED, records end time.
        On failure: marks FAILED, records error details, then re-raises.

        Returns:
            False — exceptions are always re-raised after recording.
        """
        assert self.record is not None
        assert self._storage is not None

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
                "Luro pipeline failed: %s (execution: %s) — %s",
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
                "Luro pipeline completed: %s (execution: %s)",
                self.name,
                self.execution_id[:8],
            )

        # Persist the final record
        await self._safe_storage_op(
            self._storage.save_execution, self.record
        )

        # Never swallow exceptions — re-raise so the user sees them
        return False

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
        In production mode: lets LuroStorageError propagate.
        """
        config = get_config()
        try:
            return await func(*args)
        except Exception as exc:
            if config.is_production:
                raise LuroStorageError(
                    f"Storage operation failed: {exc}"
                ) from exc
            else:
                logger.warning(
                    "Luro storage operation failed (non-fatal in %s mode): %s",
                    config.environment,
                    exc,
                )
                return None
