"""Tests for the Pipeline context manager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import sylo
from sylo.config import get_config
from sylo.core.pipeline import Pipeline
from sylo.models import ExecutionStatus
from sylo.storage.local_store import LocalStorage


@pytest.fixture
def setup_sylo(tmp_storage_dir: Path):
    """Initialize sylo and patch storage to use temp directory."""
    sylo.init(project="test-project", environment="development", storage="local")

    # Patch the storage factory to use our temp dir
    _original_get_storage = sylo.storage.get_storage

    def _patched_get_storage(config):
        return LocalStorage(root_dir=tmp_storage_dir)

    with patch("sylo.core.pipeline.get_storage", _patched_get_storage):
        yield tmp_storage_dir


class TestPipelineLifecycle:
    """Tests for the pipeline context manager lifecycle."""

    @pytest.mark.asyncio
    async def test_successful_pipeline(self, setup_sylo: Path):
        """Successful pipeline should create a COMPLETED execution record."""
        async with sylo.pipeline("test-pipeline", version="1.0") as pipe:
            assert pipe.execution_id  # should be set
            assert pipe.name == "test-pipeline"
            assert pipe.version == "1.0"
            assert pipe.record is not None
            assert pipe.record.status == ExecutionStatus.RUNNING

        # After exit, record should be COMPLETED
        assert pipe.record.status == ExecutionStatus.COMPLETED
        assert pipe.record.completed_at is not None
        assert pipe.record.error is None

    @pytest.mark.asyncio
    async def test_failed_pipeline(self, setup_sylo: Path):
        """Failed pipeline should mark execution as FAILED and re-raise."""
        with pytest.raises(ValueError, match="test error"):
            async with sylo.pipeline("test-pipeline") as pipe:
                raise ValueError("test error")

        assert pipe.record.status == ExecutionStatus.FAILED
        assert pipe.record.error is not None
        assert "test error" in pipe.record.error
        assert pipe.record.completed_at is not None

    @pytest.mark.asyncio
    async def test_exception_is_not_swallowed(self, setup_sylo: Path):
        """Pipeline should never swallow exceptions — they must propagate."""
        with pytest.raises(RuntimeError):
            async with sylo.pipeline("test-pipeline") as pipe:
                raise RuntimeError("critical failure")

    @pytest.mark.asyncio
    async def test_execution_id_is_unique(self, setup_sylo: Path):
        """Each pipeline run should get a unique execution ID."""
        ids = []
        for _ in range(5):
            async with sylo.pipeline("test-pipeline") as pipe:
                ids.append(pipe.execution_id)

        assert len(set(ids)) == 5  # all unique

    @pytest.mark.asyncio
    async def test_pipeline_metadata(self, setup_sylo: Path):
        """Pipeline should pass through user-defined metadata."""
        async with sylo.pipeline(
            "test-pipeline",
            metadata={"user_id": "123", "run_type": "scheduled"},
        ) as pipe:
            pass

        assert pipe.record.metadata["user_id"] == "123"
        assert pipe.record.metadata["run_type"] == "scheduled"

    @pytest.mark.asyncio
    async def test_pipeline_records_timing(self, setup_sylo: Path):
        """Pipeline should record start and end times."""
        async with sylo.pipeline("test-pipeline") as pipe:
            started = pipe.record.started_at

        assert pipe.record.started_at == started
        assert pipe.record.completed_at is not None
        assert pipe.record.completed_at >= pipe.record.started_at


class TestPipelineStorage:
    """Tests for pipeline storage persistence."""

    @pytest.mark.asyncio
    async def test_execution_persisted_on_success(self, setup_sylo: Path):
        """Execution record should be saved to storage on success."""
        storage = LocalStorage(root_dir=setup_sylo)

        async with sylo.pipeline("test-pipeline") as pipe:
            exec_id = pipe.execution_id

        # Record should be on disk
        record = await storage.get_execution(exec_id)
        assert record is not None
        assert record.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execution_persisted_on_failure(self, setup_sylo: Path):
        """Execution record should be saved to storage even on failure."""
        storage = LocalStorage(root_dir=setup_sylo)

        with pytest.raises(ValueError):
            async with sylo.pipeline("test-pipeline") as pipe:
                exec_id = pipe.execution_id
                raise ValueError("boom")

        record = await storage.get_execution(exec_id)
        assert record is not None
        assert record.status == ExecutionStatus.FAILED
        assert "boom" in record.error

    @pytest.mark.asyncio
    async def test_audit_events_emitted(self, setup_sylo: Path):
        """Pipeline should emit PIPELINE_STARTED and PIPELINE_COMPLETED events."""
        storage = LocalStorage(root_dir=setup_sylo)

        async with sylo.pipeline("test-pipeline") as pipe:
            exec_id = pipe.execution_id

        events = await storage.get_audit_events(exec_id)
        event_types = [e.event_type for e in events]
        assert "PIPELINE_STARTED" in event_types
        assert "PIPELINE_COMPLETED" in event_types

    @pytest.mark.asyncio
    async def test_failed_pipeline_audit_events(self, setup_sylo: Path):
        """Failed pipeline should emit PIPELINE_STARTED and PIPELINE_FAILED events."""
        storage = LocalStorage(root_dir=setup_sylo)

        with pytest.raises(ValueError):
            async with sylo.pipeline("test-pipeline") as pipe:
                exec_id = pipe.execution_id
                raise ValueError("test error")

        events = await storage.get_audit_events(exec_id)
        event_types = [e.event_type for e in events]
        assert "PIPELINE_STARTED" in event_types
        assert "PIPELINE_FAILED" in event_types
