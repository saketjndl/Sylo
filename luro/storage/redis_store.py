"""Redis storage backend for Luro SDK.

Uses Redis for fast checkpoint storage and Redis Streams for
append-only audit event logging. Suitable for production workloads
where durability and speed matter.

Key schema:
    luro:execution:{id}                          — JSON string (ExecutionRecord)
    luro:checkpoint:{execution_id}:{step_name}   — JSON string (Checkpoint)
    luro:audit:{execution_id}                    — Redis Stream (AuditEvent entries)
    luro:pipeline_executions:{pipeline_name}     — Sorted Set (score=timestamp, member=execution_id)
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from luro.models import ApprovalRequest, AuditEvent, Checkpoint, ExecutionRecord
from luro.storage.base import LuroStorage

if TYPE_CHECKING:
    from redis.asyncio import Redis  # type: ignore[import-untyped]


logger = logging.getLogger("luro.storage.redis")

# Key prefix for all Luro data in Redis
PREFIX = "luro"


class RedisStorage(LuroStorage):
    """Redis-backed storage for production workloads.

    Uses standard Redis keys for executions and checkpoints,
    and Redis Streams for append-only audit event logging.

    Args:
        redis_client: An async Redis client instance. If not provided,
            one will be created from the redis_url.
        redis_url: Redis connection URL. Used if redis_client is not provided.
    """

    def __init__(
        self,
        redis_client: Redis | None = None,
        redis_url: str = "redis://localhost:6379",
    ) -> None:
        self._client = redis_client
        self._redis_url = redis_url

    async def _get_client(self) -> Redis:
        """Lazily initialize the Redis client."""
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                self._redis_url, decode_responses=True
            )
        return self._client

    def _execution_key(self, execution_id: str) -> str:
        return f"{PREFIX}:execution:{execution_id}"

    def _checkpoint_key(self, execution_id: str, step_name: str) -> str:
        return f"{PREFIX}:checkpoint:{execution_id}:{step_name}"

    def _audit_key(self, execution_id: str) -> str:
        return f"{PREFIX}:audit:{execution_id}"

    def _pipeline_index_key(self, pipeline_name: str) -> str:
        return f"{PREFIX}:pipeline_executions:{pipeline_name}"

    def _approval_key(self, approval_id: str) -> str:
        return f"{PREFIX}:approval:{approval_id}"

    def _approval_step_key(self, execution_id: str, step_name: str) -> str:
        return f"{PREFIX}:approval_step:{execution_id}:{step_name}"

    async def save_execution(self, record: ExecutionRecord) -> None:
        """Save an execution record to Redis."""
        client = await self._get_client()
        key = self._execution_key(record.execution_id)
        await client.set(key, record.model_dump_json())

        # Index by pipeline name for list_executions queries
        index_key = self._pipeline_index_key(record.pipeline_name)
        score = record.started_at.timestamp()
        await client.zadd(index_key, {record.execution_id: score})

        logger.debug("Saved execution %s to Redis", record.execution_id)

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """Retrieve an execution record from Redis."""
        client = await self._get_client()
        data = await client.get(self._execution_key(execution_id))
        if data is None:
            return None
        return ExecutionRecord.model_validate_json(data)

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint to Redis."""
        client = await self._get_client()
        key = self._checkpoint_key(checkpoint.execution_id, checkpoint.step_name)
        await client.set(key, checkpoint.model_dump_json())
        logger.debug(
            "Saved checkpoint %s/%s to Redis",
            checkpoint.execution_id,
            checkpoint.step_name,
        )

    async def get_checkpoint(
        self, execution_id: str, step_name: str
    ) -> Checkpoint | None:
        """Retrieve a checkpoint from Redis."""
        client = await self._get_client()
        data = await client.get(self._checkpoint_key(execution_id, step_name))
        if data is None:
            return None
        return Checkpoint.model_validate_json(data)

    async def list_executions(
        self, pipeline_name: str, limit: int = 20
    ) -> list[ExecutionRecord]:
        """List executions for a pipeline, newest first.

        Uses a Redis sorted set index to efficiently query by pipeline name.
        """
        client = await self._get_client()
        index_key = self._pipeline_index_key(pipeline_name)

        # Get the most recent execution IDs (highest scores = newest)
        execution_ids: list[str] = await client.zrevrange(
            index_key, 0, limit - 1
        )

        records: list[ExecutionRecord] = []
        for exec_id in execution_ids:
            record = await self.get_execution(exec_id)
            if record is not None:
                records.append(record)

        return records

    async def append_audit_event(
        self, execution_id: str, event: AuditEvent
    ) -> None:
        """Append an audit event to a Redis Stream (append-only).

        Uses XADD which only supports appending — entries cannot be
        modified or deleted, ensuring audit log immutability.
        """
        client = await self._get_client()
        stream_key = self._audit_key(execution_id)

        # Store the full event JSON as a single field in the stream entry
        await client.xadd(stream_key, {"data": event.model_dump_json()})

        logger.debug(
            "Appended audit event %s to Redis stream %s",
            event.event_type,
            execution_id,
        )

    async def get_audit_events(self, execution_id: str) -> list[AuditEvent]:
        """Read all audit events from a Redis Stream (convenience method).

        Args:
            execution_id: The UUID of the execution.

        Returns:
            List of audit events in chronological order.
        """
        client = await self._get_client()
        stream_key = self._audit_key(execution_id)

        # XRANGE returns entries in chronological order
        entries = await client.xrange(stream_key)

        events: list[AuditEvent] = []
        for _entry_id, fields in entries:
            event_json = fields.get("data", "{}")
            events.append(AuditEvent.model_validate_json(event_json))

        return events

    async def save_approval_request(self, request: ApprovalRequest) -> None:
        """Save an approval request to Redis."""
        client = await self._get_client()
        data = request.model_dump_json()
        await client.set(self._approval_key(request.approval_id), data)
        await client.set(
            self._approval_step_key(request.execution_id, request.step_name),
            request.approval_id,
        )
        logger.debug("Saved approval request %s to Redis", request.approval_id)

    async def get_approval_request(self, approval_id: str) -> ApprovalRequest | None:
        """Retrieve an approval request from Redis by approval ID."""
        client = await self._get_client()
        data = await client.get(self._approval_key(approval_id))
        if data is None:
            return None
        return ApprovalRequest.model_validate_json(data)

    async def get_approval_request_by_step(
        self, execution_id: str, step_name: str
    ) -> ApprovalRequest | None:
        """Retrieve an approval request from Redis by step name."""
        client = await self._get_client()
        approval_id = await client.get(self._approval_step_key(execution_id, step_name))
        if approval_id is None:
            return None
        return await self.get_approval_request(approval_id)

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
