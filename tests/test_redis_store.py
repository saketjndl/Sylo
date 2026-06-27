"""Tests for the Redis storage backend.

These tests use fakeredis to mock the Redis connection.
To test against a real Redis instance, set SYLO_TEST_REDIS_URL.
"""

from __future__ import annotations

import os

import pytest

from sylo.models import (
    AuditEvent,
    Checkpoint,
    ExecutionRecord,
    ExecutionStatus,
)

# Skip all tests if fakeredis is not installed
fakeredis = pytest.importorskip("fakeredis")


@pytest.fixture
async def redis_storage():
    """Provide a RedisStorage instance backed by fakeredis."""
    import fakeredis.aioredis

    from sylo.storage.redis_store import RedisStorage

    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    storage = RedisStorage(redis_client=fake_redis)
    yield storage
    await fake_redis.aclose()


class TestRedisStorageExecution:
    """Tests for execution record operations in Redis."""

    @pytest.mark.asyncio
    async def test_save_and_get_execution(self, redis_storage):
        record = ExecutionRecord(
            execution_id="redis-exec-1",
            pipeline_name="test-pipeline",
            pipeline_version="1.0",
        )
        await redis_storage.save_execution(record)

        retrieved = await redis_storage.get_execution("redis-exec-1")
        assert retrieved is not None
        assert retrieved.execution_id == "redis-exec-1"
        assert retrieved.pipeline_name == "test-pipeline"

    @pytest.mark.asyncio
    async def test_get_nonexistent_execution(self, redis_storage):
        result = await redis_storage.get_execution("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_executions(self, redis_storage):
        for i in range(5):
            await redis_storage.save_execution(
                ExecutionRecord(
                    execution_id=f"exec-{i}",
                    pipeline_name="my-pipeline",
                )
            )

        results = await redis_storage.list_executions("my-pipeline")
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_list_executions_with_limit(self, redis_storage):
        for i in range(10):
            await redis_storage.save_execution(
                ExecutionRecord(
                    execution_id=f"exec-{i}",
                    pipeline_name="my-pipeline",
                )
            )

        results = await redis_storage.list_executions("my-pipeline", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_list_executions_filters_by_pipeline(self, redis_storage):
        await redis_storage.save_execution(
            ExecutionRecord(execution_id="a-1", pipeline_name="pipeline-a")
        )
        await redis_storage.save_execution(
            ExecutionRecord(execution_id="b-1", pipeline_name="pipeline-b")
        )

        results = await redis_storage.list_executions("pipeline-a")
        assert len(results) == 1
        assert results[0].pipeline_name == "pipeline-a"


class TestRedisStorageCheckpoint:
    """Tests for checkpoint operations in Redis."""

    @pytest.mark.asyncio
    async def test_save_and_get_checkpoint(self, redis_storage):
        cp = Checkpoint(
            execution_id="exec-1",
            step_name="fetch-data",
            output={"count": 42},
        )
        await redis_storage.save_checkpoint(cp)

        retrieved = await redis_storage.get_checkpoint("exec-1", "fetch-data")
        assert retrieved is not None
        assert retrieved.output["count"] == 42

    @pytest.mark.asyncio
    async def test_get_nonexistent_checkpoint(self, redis_storage):
        result = await redis_storage.get_checkpoint("exec-1", "nope")
        assert result is None


class TestRedisStorageAudit:
    """Tests for audit event operations in Redis (using Streams)."""

    @pytest.mark.asyncio
    async def test_append_and_read_audit_events(self, redis_storage):
        for i in range(3):
            event = AuditEvent(
                execution_id="exec-1",
                pipeline_name="test",
                event_type="STEP_COMPLETED",
                sequence_number=i + 1,
            )
            await redis_storage.append_audit_event("exec-1", event)

        events = await redis_storage.get_audit_events("exec-1")
        assert len(events) == 3
        assert events[0].sequence_number == 1
        assert events[2].sequence_number == 3

    @pytest.mark.asyncio
    async def test_read_empty_audit_log(self, redis_storage):
        events = await redis_storage.get_audit_events("nonexistent")
        assert events == []
