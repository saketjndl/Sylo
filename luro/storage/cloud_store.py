"""Cloud storage backend for Luro SDK.

Syncs execution data to the Luro Cloud API via HTTP. This backend
is used when an API key is configured, enabling the hosted dashboard,
team access, and persistent storage.

Network failures are retried 3 times with exponential backoff before
raising LuroStorageError.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from luro.exceptions import LuroStorageError
from luro.models import AuditEvent, Checkpoint, ExecutionRecord
from luro.storage.base import LuroStorage

logger = logging.getLogger("luro.storage.cloud")

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0


class CloudStorage(LuroStorage):
    """HTTP-based storage that syncs to the Luro Cloud API.

    All requests include the API key in the Authorization header.
    Network failures are retried with exponential backoff.

    Args:
        api_key: Luro Cloud API key (prefixed with "luro_").
        base_url: Base URL for the Luro Cloud API.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.luro.dev",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "luro-sdk/0.1.0",
                },
                timeout=30.0,
            )
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any] | None:
        """Make an HTTP request with retry logic.

        Retries up to MAX_RETRIES times with exponential backoff
        on network failures or 5xx server errors.

        Args:
            method: HTTP method (GET, POST, PATCH).
            path: API path (e.g., "/v1/executions").
            json_data: JSON request body.

        Returns:
            Parsed JSON response, or None for empty responses.

        Raises:
            LuroStorageError: After all retries are exhausted.
        """
        import asyncio

        client = await self._get_client()
        last_error: Exception | None = None
        backoff = INITIAL_BACKOFF_SECONDS

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.request(
                    method, path, json=json_data
                )
                # Don't retry client errors (4xx), only server errors (5xx)
                if response.status_code >= 500:
                    last_error = LuroStorageError(
                        f"Cloud API returned {response.status_code}: {response.text}"
                    )
                    if attempt < MAX_RETRIES:
                        logger.warning(
                            "Cloud API error (attempt %d/%d): %s",
                            attempt + 1,
                            MAX_RETRIES + 1,
                            response.status_code,
                        )
                        await asyncio.sleep(backoff)
                        backoff *= BACKOFF_MULTIPLIER
                        continue
                    raise last_error

                if response.status_code == 204:
                    return None

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as exc:
                raise LuroStorageError(
                    f"Cloud API request failed: {exc.response.status_code}"
                ) from exc
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Cloud API connection error (attempt %d/%d): %s",
                        attempt + 1,
                        MAX_RETRIES + 1,
                        str(exc),
                    )
                    await asyncio.sleep(backoff)
                    backoff *= BACKOFF_MULTIPLIER
                    continue

        raise LuroStorageError(
            f"Cloud API request failed after {MAX_RETRIES + 1} attempts"
        ) from last_error

    async def save_execution(self, record: ExecutionRecord) -> None:
        """Save an execution record to Luro Cloud."""
        data = record.model_dump(mode="json")
        await self._request("POST", "/v1/executions", json_data=data)
        logger.debug("Synced execution %s to cloud", record.execution_id)

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """Retrieve an execution record from Luro Cloud."""
        try:
            data = await self._request("GET", f"/v1/executions/{execution_id}")
            if data is None:
                return None
            return ExecutionRecord.model_validate(data)
        except LuroStorageError:
            return None

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint to Luro Cloud."""
        data = checkpoint.model_dump(mode="json")
        await self._request(
            "POST",
            f"/v1/executions/{checkpoint.execution_id}/checkpoints",
            json_data=data,
        )
        logger.debug(
            "Synced checkpoint %s/%s to cloud",
            checkpoint.execution_id,
            checkpoint.step_name,
        )

    async def get_checkpoint(
        self, execution_id: str, step_name: str
    ) -> Checkpoint | None:
        """Retrieve a checkpoint from Luro Cloud."""
        try:
            data = await self._request(
                "GET",
                f"/v1/executions/{execution_id}/checkpoints/{step_name}",
            )
            if data is None:
                return None
            return Checkpoint.model_validate(data)
        except LuroStorageError:
            return None

    async def list_executions(
        self, pipeline_name: str, limit: int = 20
    ) -> list[ExecutionRecord]:
        """List executions from Luro Cloud."""
        try:
            data = await self._request(
                "GET",
                f"/v1/executions?pipeline={pipeline_name}&limit={limit}",
            )
            if data is None:
                return []
            return [ExecutionRecord.model_validate(item) for item in data]
        except LuroStorageError:
            return []

    async def append_audit_event(
        self, execution_id: str, event: AuditEvent
    ) -> None:
        """Append an audit event to Luro Cloud."""
        data = event.model_dump(mode="json")
        await self._request(
            "POST",
            f"/v1/executions/{execution_id}/audit",
            json_data=data,
        )
        logger.debug(
            "Synced audit event %s to cloud for execution %s",
            event.event_type,
            execution_id,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
