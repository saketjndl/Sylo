"""Luro Context object — passed into every @luro.step function.

The Context carries execution state, provides access to prior step
outputs, and will later support permission-checked resource access
(Brief 03 — Trust Broker).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("luro")


class Context:
    """Execution context passed to every pipeline step.

    The Context gives each step access to execution metadata,
    outputs from prior steps, and arbitrary user-defined metadata.

    Attributes:
        execution_id: Unique ID for this pipeline execution.
        pipeline_name: Name of the currently running pipeline.
        run_number: How many times this pipeline has run today.
        previous_outputs: Dict mapping step names to their output dicts.
        metadata: Arbitrary user-defined metadata for this execution.
    """

    def __init__(
        self,
        execution_id: str,
        pipeline_name: str,
        run_number: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.execution_id = execution_id
        self.pipeline_name = pipeline_name
        self.run_number = run_number
        self.previous_outputs: dict[str, Any] = {}
        self.metadata: dict[str, Any] = metadata or {}

        # Internal: tracks resource accesses for trust broker (Brief 03)
        self._resource_accesses: list[dict[str, Any]] = []
        # Internal: trust declarations set by @luro.trust (Brief 03)
        self._trust_declarations: dict[str, list[str]] | None = None

    def get_output(self, step_name: str) -> Any:
        """Get the output of a previously completed step.

        Args:
            step_name: Name of the step whose output you want.

        Returns:
            The output dict from that step.

        Raises:
            KeyError: If the step has not run or has no output.
        """
        if step_name not in self.previous_outputs:
            raise KeyError(
                f"No output found for step '{step_name}'. "
                f"Available steps: {list(self.previous_outputs.keys())}"
            )
        return self.previous_outputs[step_name]

    async def access(
        self,
        resource: str,
        action: str = "read",
        params: dict[str, Any] | None = None,
        handler: Any = None,
    ) -> Any:
        """Access an external resource through the permission-checked context.

        This method is the gateway for all external resource access.
        In Brief 03 (Trust Broker), it enforces permission declarations.
        For now, it records the access and calls the handler directly.

        Args:
            resource: Resource identifier in "service.resource" format.
            action: One of "read", "write", "execute", "delete".
            params: Parameters to pass to the handler.
            handler: The callable that performs the actual resource access.

        Returns:
            The result of calling handler(params).
        """
        access_record = {
            "resource": resource,
            "action": action,
            "step_name": None,  # Set by the step decorator
        }
        self._resource_accesses.append(access_record)

        if handler is not None:
            if params:
                return await handler(**params) if callable(handler) else handler
            return await handler() if callable(handler) else handler

        return None
