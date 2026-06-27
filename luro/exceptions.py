"""Luro SDK exception hierarchy.

All custom exceptions inherit from LuroError, making it easy
to catch any Luro-specific error with a single except clause.
"""


class LuroError(Exception):
    """Base exception for all Luro SDK errors."""

    pass


class LuroConfigError(LuroError):
    """Raised when SDK configuration is invalid or missing.

    Examples:
        - luro.init() called without a project name
        - Invalid environment value (not development/staging/production)
        - Invalid storage backend specified
    """

    pass


class LuroStorageError(LuroError):
    """Raised when a storage operation fails in production mode.

    In development mode, storage failures are logged as warnings
    and do not raise exceptions. In production mode, this exception
    is raised to ensure failures are not silently ignored.
    """

    pass


class LuroPermissionError(LuroError):
    """Raised when an agent step attempts to access an undeclared resource.

    The Trust Broker enforces that each step can only access resources
    it explicitly declared via @luro.trust(). Accessing anything else
    raises this error immediately.

    Note: Implemented in Brief 03 — Trust Broker.
    """

    pass


class LuroApprovalRejectedError(LuroError):
    """Raised when a human reviewer rejects an approval request.

    When a step decorated with @luro.requires_approval is rejected,
    the pipeline fails cleanly with this error.

    Note: Implemented in Brief 04 — Human Approval Gates.
    """

    pass


class LuroCheckpointExpiredError(LuroError):
    """Raised when attempting to resume from an expired checkpoint.

    Checkpoints may have a TTL. If a checkpoint has expired,
    the step must be re-executed rather than using the stale data.

    Note: Implemented in Brief 02 — Checkpoint Engine.
    """

    pass
