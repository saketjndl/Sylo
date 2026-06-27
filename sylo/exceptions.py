"""Sylo SDK exception hierarchy.

All custom exceptions inherit from SyloError, making it easy
to catch any Sylo-specific error with a single except clause.
"""


class SyloError(Exception):
    """Base exception for all Sylo SDK errors."""

    pass


class SyloConfigError(SyloError):
    """Raised when SDK configuration is invalid or missing.

    Examples:
        - sylo.init() called without a project name
        - Invalid environment value (not development/staging/production)
        - Invalid storage backend specified
    """

    pass


class SyloStorageError(SyloError):
    """Raised when a storage operation fails in production mode.

    In development mode, storage failures are logged as warnings
    and do not raise exceptions. In production mode, this exception
    is raised to ensure failures are not silently ignored.
    """

    pass


class SyloPermissionError(SyloError):
    """Raised when an agent step attempts to access an undeclared resource.

    The Trust Broker enforces that each step can only access resources
    it explicitly declared via @sylo.trust(). Accessing anything else
    raises this error immediately.
    """

    pass


class SyloApprovalRejectedError(SyloError):
    """Raised when a human reviewer rejects an approval request.

    When a step decorated with @sylo.requires_approval is rejected,
    the pipeline fails cleanly with this error.
    """

    pass


class SyloCheckpointExpiredError(SyloError):
    """Raised when attempting to resume from an expired checkpoint."""

    pass


# Backwards compatibility aliases
LuroError = SyloError
LuroConfigError = SyloConfigError
LuroStorageError = SyloStorageError
LuroPermissionError = SyloPermissionError
LuroApprovalRejectedError = SyloApprovalRejectedError
LuroCheckpointExpiredError = SyloCheckpointExpiredError
