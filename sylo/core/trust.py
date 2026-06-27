"""Trust Broker for Sylo SDK.

Enforces runtime permission sandboxing on agent steps.
Steps declare their permissions using the @sylo.trust() decorator.
Accesses to resources are checked at runtime via ctx.access().
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Literal

from sylo.exceptions import SyloPermissionError

logger = logging.getLogger("sylo")


def trust(
    can_read: list[str] | None = None,
    can_write: list[str] | None = None,
    can_execute: list[str] | None = None,
    can_delete: list[str] | None = None,
) -> Callable:
    """Decorator to declare permissions required by a pipeline step.

    Permissions should be strings in "service.resource" format (e.g. "gmail.messages").
    Using "*" acts as a wildcard, allowing any resource, but logs a warning.

    Args:
        can_read: List of resources this step can read.
        can_write: List of resources this step can write to.
        can_execute: List of resources this step can execute.
        can_delete: List of resources this step can delete.

    Returns:
        Decorated function with trust declarations attached.

    Example:
        @sylo.step("send-email")
        @sylo.trust(
            can_read=["gmail.messages", "context.user_email"],
            can_write=["gmail.drafts"],
            can_execute=["gmail.send"]
        )
        async def send_email_step(ctx):
            ...
    """
    declarations = {
        "read": can_read or [],
        "write": can_write or [],
        "execute": can_execute or [],
        "delete": can_delete or [],
    }

    # Log warnings if wildcards are used
    for action, resources in declarations.items():
        if "*" in resources:
            logger.warning(
                "Sylo Trust: Wildcard '*' permission declared for '%s' action. "
                "This allows unrestricted access.",
                action,
            )

    def decorator(func: Callable) -> Callable:
        # Attach permissions to the function so they can be introspected
        func._sylo_trust_declarations = declarations  # type: ignore[attr-defined]
        func._luro_trust_declarations = declarations  # backwards compat
        return func

    return decorator


def check_permission(
    declared_resources: list[str],
    requested_resource: str,
) -> bool:
    """Check if a requested resource matches any declared resource.

    Supports exact matches, suffix wildcards (e.g., 'gmail.*'), and
    global wildcards ('*').

    Args:
        declared_resources: List of permitted resource patterns.
        requested_resource: The resource string being accessed.

    Returns:
        True if permitted, False otherwise.
    """
    for declared in declared_resources:
        if declared == "*":
            return True
        if declared == requested_resource:
            return True
        if declared.endswith(".*"):
            prefix = declared[:-2]
            if requested_resource.startswith(prefix + ".") or requested_resource == prefix:
                return True
    return False
