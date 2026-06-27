"""Tests for error handling behavior across environments."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import sylo
from sylo.config import SyloConfig, set_config
from sylo.exceptions import SyloStorageError
from sylo.storage.local_store import LocalStorage


class TestDevModeErrorHandling:
    """In development mode, storage errors should be logged but not raised."""

    @pytest.mark.asyncio
    async def test_storage_failure_does_not_crash_pipeline(
        self, tmp_storage_dir: Path
    ):
        """In dev mode, a broken storage backend should not crash the pipeline."""
        sylo.init(project="test", environment="development", storage="local")

        # Create a storage mock that always fails
        failing_storage = AsyncMock()
        failing_storage.save_execution = AsyncMock(
            side_effect=IOError("disk full")
        )
        failing_storage.append_audit_event = AsyncMock(
            side_effect=IOError("disk full")
        )

        with patch("sylo.core.pipeline.get_storage", return_value=failing_storage):
            # This should NOT raise despite storage failures
            async with sylo.pipeline("test-pipeline") as pipe:
                pass  # pipeline code runs fine

        # Pipeline should still complete normally
        assert pipe.record.status.value == "COMPLETED"

    @pytest.mark.asyncio
    async def test_storage_failure_logs_warning(
        self, tmp_storage_dir: Path, caplog
    ):
        """In dev mode, storage failures should produce warning logs."""
        sylo.init(project="test", environment="development", storage="local")

        failing_storage = AsyncMock()
        failing_storage.save_execution = AsyncMock(
            side_effect=IOError("disk full")
        )
        failing_storage.append_audit_event = AsyncMock(
            side_effect=IOError("disk full")
        )

        with patch("sylo.core.pipeline.get_storage", return_value=failing_storage):
            with caplog.at_level(logging.WARNING, logger="sylo"):
                async with sylo.pipeline("test-pipeline") as pipe:
                    pass

        assert any("non-fatal" in msg.lower() for msg in caplog.messages)


class TestProdModeErrorHandling:
    """In production mode, storage errors should raise SyloStorageError."""

    @pytest.mark.asyncio
    async def test_storage_failure_raises_in_production(
        self, tmp_storage_dir: Path
    ):
        """In production mode, storage failures must raise SyloStorageError."""
        sylo.init(project="test", environment="production", storage="local")

        failing_storage = AsyncMock()
        failing_storage.save_execution = AsyncMock(
            side_effect=IOError("disk full")
        )
        failing_storage.append_audit_event = AsyncMock(
            side_effect=IOError("disk full")
        )

        with patch("sylo.core.pipeline.get_storage", return_value=failing_storage):
            with pytest.raises(SyloStorageError, match="Storage operation failed"):
                async with sylo.pipeline("test-pipeline") as pipe:
                    pass


class TestCloudRetryBehavior:
    """Cloud storage should retry network failures with exponential backoff."""

    @pytest.mark.asyncio
    async def test_cloud_retries_on_connection_error(self):
        """Cloud storage should retry up to 3 times on connection failures."""
        import httpx

        from sylo.storage.cloud_store import CloudStorage

        cloud = CloudStorage(api_key="sylo_test", base_url="http://localhost:9999")

        # Mock the httpx client to fail with connection error
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        cloud._client = mock_client

        with pytest.raises(SyloStorageError, match="failed after"):
            from sylo.models import ExecutionRecord

            await cloud.save_execution(
                ExecutionRecord(pipeline_name="test")
            )

        # Should have been called 4 times (1 initial + 3 retries)
        assert mock_client.request.call_count == 4

    @pytest.mark.asyncio
    async def test_cloud_retries_on_server_error(self):
        """Cloud storage should retry on 5xx server errors."""
        import httpx

        from sylo.storage.cloud_store import CloudStorage

        cloud = CloudStorage(api_key="sylo_test", base_url="http://localhost:9999")

        # Mock the httpx client to return 500
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        cloud._client = mock_client

        with pytest.raises(SyloStorageError):
            from sylo.models import ExecutionRecord

            await cloud.save_execution(
                ExecutionRecord(pipeline_name="test")
            )

        # Should have been called 4 times (1 initial + 3 retries)
        assert mock_client.request.call_count == 4

    @pytest.mark.asyncio
    async def test_cloud_does_not_retry_client_errors(self):
        """Cloud storage should NOT retry on 4xx client errors."""
        import httpx
        from unittest.mock import Mock

        from sylo.storage.cloud_store import CloudStorage

        cloud = CloudStorage(api_key="sylo_test", base_url="http://localhost:9999")

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "Not Found",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(404),
            )
        )

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        cloud._client = mock_client

        with pytest.raises(SyloStorageError):
            from sylo.models import ExecutionRecord

            await cloud.save_execution(
                ExecutionRecord(pipeline_name="test")
            )

        # Should have been called only once — no retries for 4xx
        assert mock_client.request.call_count == 1
