"""Sylo Context object — passed into every @sylo.step function.

The Context carries execution state, provides access to prior step
outputs, and enforces permission-checked resource access (Brief 03 — Trust Broker).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from sylo.exceptions import SyloPermissionError
from sylo.core.costs import estimate_cost
from sylo.core.trust import check_permission
from sylo.models import TokenUsage

logger = logging.getLogger("sylo")


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

        # Internal: tracks active step name during execution
        self._current_step_name: str | None = None
        # Internal: tracks resource accesses for audit log
        self._resource_accesses: list[dict[str, Any]] = []
        # Internal: trust declarations set by @sylo.trust
        self._trust_declarations: dict[str, list[str]] | None = None
        # Internal: tracks which declared permissions were actually used
        self._permissions_used: set[tuple[str, str]] = set()  # set of (action, resource)
        # Internal: count of blocked permission attempts
        self._violations_attempted: int = 0
        # Internal: manually recorded token usage from ctx.record_token_usage()
        self._recorded_token_usage: TokenUsage | None = None

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
        action: Literal["read", "write", "execute", "delete"] = "read",
        params: dict[str, Any] | None = None,
        handler: Any = None,
    ) -> Any:
        """Access an external resource through the permission-checked context.

        This method is the gateway for all external resource access.
        It checks declared permissions at runtime if @sylo.trust is used.

        Args:
            resource: Resource identifier in "service.resource" format.
            action: One of "read", "write", "execute", "delete".
            params: Parameters to pass to the handler.
            handler: The callable that performs the actual resource access.

        Returns:
            The result of calling handler(params).

        Raises:
            SyloPermissionError: If the step does not have permission to access the resource.
        """
        # Import to avoid circular imports
        from sylo.core.pipeline import _current_pipeline

        pipeline = _current_pipeline.get(None)

        # Track the resource access attempt
        access_record = {
            "resource": resource,
            "action": action,
            "step_name": self._current_step_name,
        }
        self._resource_accesses.append(access_record)

        # If trust enforcement is declared, check it
        if self._trust_declarations is not None:
            allowed_patterns = self._trust_declarations.get(action, [])
            permitted = check_permission(allowed_patterns, resource)

            if not permitted:
                self._violations_attempted += 1

                # Record violation in audit log
                if pipeline is not None:
                    await pipeline._emit_audit_event(
                        event_type="PERMISSION_VIOLATION",
                        step_name=self._current_step_name,
                        data={
                            "resource": resource,
                            "action": action,
                            "declared_permissions": allowed_patterns,
                        },
                    )

                raise SyloPermissionError(
                    f"Sylo Trust: Step '{self._current_step_name}' attempted to {action} "
                    f"undeclared resource '{resource}'."
                )

            # Record that this specific permission pattern was used
            # We match the resource to the declared patterns that allowed it
            for pattern in allowed_patterns:
                if check_permission([pattern], resource):
                    self._permissions_used.add((action, pattern))

            # Record successful check in audit log
            if pipeline is not None:
                await pipeline._emit_audit_event(
                    event_type="PERMISSION_CHECKED",
                    step_name=self._current_step_name,
                    data={
                        "resource": resource,
                        "action": action,
                        "status": "ALLOWED",
                    },
                )

        # Call the actual handler
        if handler is not None:
            if callable(handler):
                import inspect
                res = handler(**params) if params else handler()
                if inspect.isawaitable(res):
                    return await res
                return res
            return handler

        return None

    def record_token_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Record token usage for the current step.

        Use this to manually track LLM token costs when the step's
        return value doesn't include a ``usage`` dict.

        Args:
            prompt_tokens: Number of input/prompt tokens consumed.
            completion_tokens: Number of output/completion tokens generated.
            model: Model name for cost estimation (e.g., ``"gpt-4o"``).
            total_tokens: Override total token count. If ``None``,
                computed as ``prompt_tokens + completion_tokens``.

        Example:
            >>> @sylo.step("analyze")
            ... async def analyze(ctx: sylo.Context) -> dict:
            ...     ctx.record_token_usage(
            ...         prompt_tokens=450,
            ...         completion_tokens=120,
            ...         model="gpt-4o",
            ...     )
            ...     return {"result": "..."}
        """
        from sylo.core.pipeline import _current_pipeline

        computed_total = total_tokens if total_tokens is not None else (prompt_tokens + completion_tokens)

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=computed_total,
            model=model,
        )
        usage.estimated_cost_usd = estimate_cost(usage)

        # Update pipeline-level token cost
        pipeline = _current_pipeline.get(None)
        if pipeline is not None and pipeline.record is not None:
            pipeline.record.token_cost.total_tokens += computed_total
            pipeline.record.token_cost.estimated_cost_usd += usage.estimated_cost_usd

        # Store on context for the checkpoint engine to pick up
        self._recorded_token_usage = usage

        logger.debug(
            "Recorded token usage for step '%s': %d tokens, $%.4f",
            self._current_step_name,
            computed_total,
            usage.estimated_cost_usd,
        )

