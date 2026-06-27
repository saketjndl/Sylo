"""Tests for the LocalStorage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from sylo.models import (
    AuditEvent,
    Checkpoint,
    CheckpointStatus,
    ExecutionRecord,
    ExecutionStatus,
)
from sylo.storage.local_store import LocalStorage


class TestLocalStorageExecution:
    """Tests for execution record CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_execution(self, local_storage: LocalStorage):
        """Saved executions should be retrievable by ID."""
        record = ExecutionRecord(
            execution_id="test-exec-1",
            pipeline_name="test-pipeline",
            pipeline_version="1.0",
        )
        await local_storage.save_execution(record)

        retrieved = await local_storage.get_execution("test-exec-1")
        assert retrieved is not None
        assert retrieved.execution_id == "test-exec-1"
        assert retrieved.pipeline_name == "test-pipeline"
        assert retrieved.pipeline_version == "1.0"
        assert retrieved.status == ExecutionStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_nonexistent_execution(self, local_storage: LocalStorage):
        """Getting a non-existent execution should return None."""
        result = await local_storage.get_execution("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_execution(self, local_storage: LocalStorage):
        """Saving an execution with the same ID should update it."""
        record = ExecutionRecord(
            execution_id="test-exec-1",
            pipeline_name="test-pipeline",
        )
        await local_storage.save_execution(record)

        # Update status
        record.status = ExecutionStatus.COMPLETED
        await local_storage.save_execution(record)

        retrieved = await local_storage.get_execution("test-exec-1")
        assert retrieved is not None
        assert retrieved.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_list_executions(self, local_storage: LocalStorage):
        """list_executions should return records for the specified pipeline."""
        # Create executions for two different pipelines
        for i in range(5):
            await local_storage.save_execution(
                ExecutionRecord(
                    execution_id=f"exec-a-{i}",
                    pipeline_name="pipeline-a",
                )
            )
        for i in range(3):
            await local_storage.save_execution(
                ExecutionRecord(
                    execution_id=f"exec-b-{i}",
                    pipeline_name="pipeline-b",
                )
            )

        results_a = await local_storage.list_executions("pipeline-a")
        assert len(results_a) == 5
        assert all(r.pipeline_name == "pipeline-a" for r in results_a)

        results_b = await local_storage.list_executions("pipeline-b")
        assert len(results_b) == 3

    @pytest.mark.asyncio
    async def test_list_executions_with_limit(self, local_storage: LocalStorage):
        """list_executions should respect the limit parameter."""
        for i in range(10):
            await local_storage.save_execution(
                ExecutionRecord(
                    execution_id=f"exec-{i}",
                    pipeline_name="test-pipeline",
                )
            )

        results = await local_storage.list_executions("test-pipeline", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_list_executions_empty(self, local_storage: LocalStorage):
        """list_executions for non-existent pipeline should return empty list."""
        results = await local_storage.list_executions("nonexistent")
        assert results == []


class TestLocalStorageCheckpoint:
    """Tests for checkpoint CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_checkpoint(self, local_storage: LocalStorage):
        """Saved checkpoints should be retrievable by execution_id + step_name."""
        cp = Checkpoint(
            execution_id="exec-1",
            step_name="fetch-emails",
            output={"emails": ["a@b.com"]},
            duration_ms=1200,
        )
        await local_storage.save_checkpoint(cp)

        retrieved = await local_storage.get_checkpoint("exec-1", "fetch-emails")
        assert retrieved is not None
        assert retrieved.step_name == "fetch-emails"
        assert retrieved.output == {"emails": ["a@b.com"]}
        assert retrieved.duration_ms == 1200

    @pytest.mark.asyncio
    async def test_get_nonexistent_checkpoint(self, local_storage: LocalStorage):
        """Getting a non-existent checkpoint should return None."""
        result = await local_storage.get_checkpoint("exec-1", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_checkpoints_per_execution(
        self, local_storage: LocalStorage
    ):
        """Multiple checkpoints can be saved for the same execution."""
        for step in ["step-1", "step-2", "step-3"]:
            await local_storage.save_checkpoint(
                Checkpoint(
                    execution_id="exec-1",
                    step_name=step,
                    output={"step": step},
                )
            )

        for step in ["step-1", "step-2", "step-3"]:
            cp = await local_storage.get_checkpoint("exec-1", step)
            assert cp is not None
            assert cp.output["step"] == step


class TestLocalStorageAudit:
    """Tests for audit event append operations."""

    @pytest.mark.asyncio
    async def test_append_and_read_audit_events(
        self, local_storage: LocalStorage
    ):
        """Appended audit events should be readable in order."""
        for i in range(5):
            event = AuditEvent(
                execution_id="exec-1",
                pipeline_name="test-pipeline",
                event_type="STEP_COMPLETED",
                sequence_number=i + 1,
                data={"index": i},
            )
            await local_storage.append_audit_event("exec-1", event)

        events = await local_storage.get_audit_events("exec-1")
        assert len(events) == 5
        assert events[0].sequence_number == 1
        assert events[4].sequence_number == 5
        assert events[2].data["index"] == 2

    @pytest.mark.asyncio
    async def test_audit_events_are_append_only(
        self, local_storage: LocalStorage
    ):
        """Each append should add to existing events, not replace them."""
        event1 = AuditEvent(
            execution_id="exec-1",
            pipeline_name="test",
            event_type="PIPELINE_STARTED",
            sequence_number=1,
        )
        event2 = AuditEvent(
            execution_id="exec-1",
            pipeline_name="test",
            event_type="PIPELINE_COMPLETED",
            sequence_number=2,
        )

        await local_storage.append_audit_event("exec-1", event1)
        await local_storage.append_audit_event("exec-1", event2)

        events = await local_storage.get_audit_events("exec-1")
        assert len(events) == 2
        assert events[0].event_type == "PIPELINE_STARTED"
        assert events[1].event_type == "PIPELINE_COMPLETED"

    @pytest.mark.asyncio
    async def test_read_empty_audit_log(self, local_storage: LocalStorage):
        """Reading audit events for an execution with no events returns []."""
        events = await local_storage.get_audit_events("nonexistent")
        assert events == []
