"""Abstract storage interface for Luro SDK.

All storage backends (local, Redis, cloud) implement this interface.
This ensures that the rest of the SDK never couples to a specific
storage mechanism.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from luro.models import AuditEvent, Checkpoint, ExecutionRecord


class LuroStorage(ABC):
    """Abstract base class for all Luro storage backends.

    Implementations must handle their own connection management
    and serialization. All methods are async to support non-blocking
    I/O across all backends uniformly.
    """

    @abstractmethod
    async def save_execution(self, record: ExecutionRecord) -> None:
        """Save or update an execution record.

        Args:
            record: The execution record to persist.
        """
        ...

    @abstractmethod
    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """Retrieve an execution record by ID.

        Args:
            execution_id: The UUID of the execution.

        Returns:
            The execution record, or None if not found.
        """
        ...

    @abstractmethod
    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save a step checkpoint.

        Args:
            checkpoint: The checkpoint data to persist.
        """
        ...

    @abstractmethod
    async def get_checkpoint(
        self, execution_id: str, step_name: str
    ) -> Checkpoint | None:
        """Retrieve a checkpoint for a specific step.

        Args:
            execution_id: The UUID of the execution.
            step_name: The name of the step.

        Returns:
            The checkpoint, or None if not found.
        """
        ...

    @abstractmethod
    async def list_executions(
        self, pipeline_name: str, limit: int = 20
    ) -> list[ExecutionRecord]:
        """List recent executions for a pipeline.

        Args:
            pipeline_name: The pipeline to query.
            limit: Maximum number of records to return.

        Returns:
            List of execution records, newest first.
        """
        ...

    @abstractmethod
    async def append_audit_event(
        self, execution_id: str, event: AuditEvent
    ) -> None:
        """Append an immutable audit event to an execution's log.

        Once written, audit events must never be modified or deleted.

        Args:
            execution_id: The UUID of the execution.
            event: The audit event to append.
        """
        ...
