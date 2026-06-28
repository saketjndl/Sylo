"""Sylo SDK — Production operating layer for AI agent pipelines.

Sylo wraps existing agent frameworks (LangGraph, CrewAI, OpenAI Agents SDK)
and adds production guarantees: smart checkpointing, permission enforcement,
human approval gates, and immutable audit logging.

Quick start:
    import sylo

    sylo.init(project="my-project")

    async with sylo.pipeline("my-pipeline", version="1.0") as pipe:
        result = await my_agent_function(inputs)

Environment variables:
    SYLO_API_KEY       — API key for Sylo Cloud (optional)
    SYLO_PROJECT       — Project name
    SYLO_ENVIRONMENT   — development | staging | production
    SYLO_STORAGE       — local | redis | cloud
    SYLO_REDIS_URL     — Redis connection URL
"""

from __future__ import annotations

import logging
from typing import Any

from sylo.config import SyloConfig, LuroConfig, reset_config, set_config
from sylo.core.approval import approve, reject, requires_approval
from sylo.core.audit import get_summary, replay
from sylo.core.checkpoint import step
from sylo.core.context import Context
from sylo.core.pipeline import Pipeline
from sylo.core.trust import trust
from sylo.exceptions import (
    SyloApprovalRejectedError,
    SyloCheckpointExpiredError,
    SyloConfigError,
    SyloError,
    SyloPermissionError,
    SyloStorageError,
    LuroApprovalRejectedError,
    LuroCheckpointExpiredError,
    LuroConfigError,
    LuroError,
    LuroPermissionError,
    LuroStorageError,
)
from sylo.models import (
    ApprovalRequest,
    ApprovalStatus,
    AuditEvent,
    Checkpoint,
    ExecutionRecord,
    ExecutionStatus,
    ExecutionSummary,
    StepSummary,
)

__version__ = "0.1.1"

__all__ = [
    # Initialization
    "init",
    "pipeline",
    # Checkpoint engine (Brief 02)
    "step",
    "Context",
    # Audit & Replay (Brief 05)
    "get_summary",
    "replay",
    # Trust broker (Brief 03)
    "trust",
    # Approval gates (Brief 04)
    "requires_approval",
    "approve",
    "reject",
    # Models
    "ExecutionRecord",
    "Checkpoint",
    "ApprovalRequest",
    "ApprovalStatus",
    "AuditEvent",
    "ExecutionStatus",
    "ExecutionSummary",
    "StepSummary",
    # Exceptions (Sylo)
    "SyloError",
    "SyloConfigError",
    "SyloStorageError",
    "SyloPermissionError",
    "SyloApprovalRejectedError",
    "SyloCheckpointExpiredError",
    # Exceptions (Luro Backwards Compat)
    "LuroError",
    "LuroConfigError",
    "LuroStorageError",
    "LuroPermissionError",
    "LuroApprovalRejectedError",
    "LuroCheckpointExpiredError",
]

# Configure logging with a NullHandler so library users control output
logging.getLogger("sylo").addHandler(logging.NullHandler())
logging.getLogger("luro").addHandler(logging.NullHandler())


def init(
    project: str | None = None,
    api_key: str | None = None,
    environment: str = "development",
    storage: str = "local",
    redis_url: str = "redis://localhost:6379",
    cloud_api_url: str = "https://api.sylo.dev",
    notifications: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Initialize the Sylo SDK.

    Must be called once at the top of your project before using
    any other Sylo functionality. Configures the storage backend,
    environment mode, and optional cloud sync.

    Args:
        project: Project name (required). Can also be set via SYLO_PROJECT.
        api_key: Sylo Cloud API key. Enables cloud sync when set.
            Can also be set via SYLO_API_KEY.
        environment: Execution environment — "development" (default),
            "staging", or "production". Affects error handling behavior.
        storage: Storage backend — "local" (default), "redis", or "cloud".
        redis_url: Redis connection URL. Only used when storage="redis".
        cloud_api_url: Sylo Cloud API base URL. Only used when storage="cloud".
        notifications: Notification channel configuration for approval gates.
        **kwargs: Additional configuration passed to SyloConfig.

    Raises:
        SyloConfigError: If required configuration is missing or invalid.

    Example:
        >>> import sylo
        >>> sylo.init(project="my-project")

        >>> # With cloud sync
        >>> sylo.init(
        ...     project="my-project",
        ...     api_key="sylo_xxx",
        ...     environment="production",
        ...     storage="cloud",
        ... )
    """
    try:
        config = SyloConfig(
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
        raise SyloConfigError(f"Invalid Sylo configuration: {exc}") from exc

    set_config(config)

    # Set up logging for development mode
    sylo_logger = logging.getLogger("sylo")
    if config.is_development and not sylo_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s [sylo] %(message)s")
        )
        sylo_logger.addHandler(handler)
        sylo_logger.setLevel(logging.INFO)

    sylo_logger.info(
        "Sylo initialized — project=%s, env=%s, storage=%s",
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

    This is the core primitive of Sylo. Wrap your agent pipeline
    with this context manager to get automatic execution tracking,
    checkpointing, and audit logging.

    Args:
        name: Pipeline name (e.g., "email-processor", "customer-onboarding").
        version: Pipeline version string. Useful for tracking changes.
        metadata: Arbitrary key-value metadata attached to this execution.

    Returns:
        An async context manager that tracks the pipeline execution.

    Example:
        >>> async with sylo.pipeline("my-pipeline", version="1.0") as pipe:
        ...     result = await my_agent_function(inputs)
        ...     print(f"Execution ID: {pipe.execution_id}")

    Raises:
        SyloConfigError: If sylo.init() has not been called.
    """
    return Pipeline(name=name, version=version, metadata=metadata)
