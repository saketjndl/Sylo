"""Luro SDK — Production operating layer for AI agent pipelines.

Luro wraps existing agent frameworks (LangGraph, CrewAI, OpenAI Agents SDK)
and adds production guarantees: smart checkpointing, permission enforcement,
human approval gates, and immutable audit logging.

Quick start:
    import luro

    luro.init(project="my-project")

    async with luro.pipeline("my-pipeline", version="1.0") as pipe:
        result = await my_agent_function(inputs)

Environment variables:
    LURO_API_KEY       — API key for Luro Cloud (optional)
    LURO_PROJECT       — Project name
    LURO_ENVIRONMENT   — development | staging | production
    LURO_STORAGE       — local | redis | cloud
    LURO_REDIS_URL     — Redis connection URL
"""

from __future__ import annotations

import logging
from typing import Any

from luro.config import LuroConfig, reset_config, set_config
from luro.core.pipeline import Pipeline
from luro.exceptions import (
    LuroApprovalRejectedError,
    LuroCheckpointExpiredError,
    LuroConfigError,
    LuroError,
    LuroPermissionError,
    LuroStorageError,
)
from luro.models import AuditEvent, Checkpoint, ExecutionRecord, ExecutionStatus

__version__ = "0.1.0"

__all__ = [
    # Initialization
    "init",
    "pipeline",
    # Models
    "ExecutionRecord",
    "Checkpoint",
    "AuditEvent",
    "ExecutionStatus",
    # Exceptions
    "LuroError",
    "LuroConfigError",
    "LuroStorageError",
    "LuroPermissionError",
    "LuroApprovalRejectedError",
    "LuroCheckpointExpiredError",
]

# Configure logging with a NullHandler so library users control output
logging.getLogger("luro").addHandler(logging.NullHandler())


def init(
    project: str | None = None,
    api_key: str | None = None,
    environment: str = "development",
    storage: str = "local",
    redis_url: str = "redis://localhost:6379",
    cloud_api_url: str = "https://api.luro.dev",
    notifications: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Initialize the Luro SDK.

    Must be called once at the top of your project before using
    any other Luro functionality. Configures the storage backend,
    environment mode, and optional cloud sync.

    Args:
        project: Project name (required). Can also be set via LURO_PROJECT.
        api_key: Luro Cloud API key. Enables cloud sync when set.
            Can also be set via LURO_API_KEY.
        environment: Execution environment — "development" (default),
            "staging", or "production". Affects error handling behavior.
        storage: Storage backend — "local" (default), "redis", or "cloud".
        redis_url: Redis connection URL. Only used when storage="redis".
        cloud_api_url: Luro Cloud API base URL. Only used when storage="cloud".
        notifications: Notification channel configuration for approval gates.
        **kwargs: Additional configuration passed to LuroConfig.

    Raises:
        LuroConfigError: If required configuration is missing or invalid.

    Example:
        >>> import luro
        >>> luro.init(project="my-project")

        >>> # With cloud sync
        >>> luro.init(
        ...     project="my-project",
        ...     api_key="luro_xxx",
        ...     environment="production",
        ...     storage="cloud",
        ... )
    """
    try:
        config = LuroConfig(
            project=project,  # type: ignore[arg-type]
            api_key=api_key,
            environment=environment,  # type: ignore[arg-type]
            storage=storage,  # type: ignore[arg-type]
            redis_url=redis_url,
            cloud_api_url=cloud_api_url,
            notifications=notifications or {},
            **kwargs,
        )
    except Exception as exc:
        raise LuroConfigError(f"Invalid Luro configuration: {exc}") from exc

    set_config(config)

    # Set up logging for development mode
    luro_logger = logging.getLogger("luro")
    if config.is_development and not luro_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s [luro] %(message)s")
        )
        luro_logger.addHandler(handler)
        luro_logger.setLevel(logging.INFO)

    luro_logger.info(
        "Luro initialized — project=%s, env=%s, storage=%s",
        config.project,
        config.environment,
        config.storage,
    )


def pipeline(
    name: str,
    version: str = "0.0.0",
    metadata: dict[str, Any] | None = None,
) -> Pipeline:
    """Create a pipeline context manager.

    This is the core primitive of Luro. Wrap your agent pipeline
    with this context manager to get automatic execution tracking,
    checkpointing, and audit logging.

    Args:
        name: Pipeline name (e.g., "email-processor", "customer-onboarding").
        version: Pipeline version string. Useful for tracking changes.
        metadata: Arbitrary key-value metadata attached to this execution.

    Returns:
        An async context manager that tracks the pipeline execution.

    Example:
        >>> async with luro.pipeline("my-pipeline", version="1.0") as pipe:
        ...     result = await my_agent_function(inputs)
        ...     print(f"Execution ID: {pipe.execution_id}")

    Raises:
        LuroConfigError: If luro.init() has not been called.
    """
    return Pipeline(name=name, version=version, metadata=metadata)
