"""Checkpoint engine — Sylo's flagship feature.

The @sylo.step decorator saves the output of each agent step so
that if the pipeline fails, it resumes from the last successful
step instead of restarting from scratch.

Features:
- Automatic checkpoint save/restore per step
- Token usage extraction and cost estimation
- Retry with configurable exponential backoff
- Step timing and duration tracking
- Input hashing for cache invalidation
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from sylo.config import get_config
from sylo.core.context import Context
from sylo.core.costs import estimate_cost, extract_token_usage
from sylo.models import (
    AuditEvent,
    Checkpoint,
    CheckpointStatus,
    TokenUsage,
)

logger = logging.getLogger("sylo")


class StepResult:
    """Metadata about a completed step execution.

    Attached to the context after each step runs so the pipeline
    can aggregate results for the cost savings report.
    """

    def __init__(
        self,
        step_name: str,
        output: Any,
        duration_ms: int,
        was_cached: bool,
        token_usage: TokenUsage | None,
        retry_count: int,
        cost_saved_usd: float = 0.0,
        time_saved_ms: int = 0,
    ) -> None:
        self.step_name = step_name
        self.output = output
        self.duration_ms = duration_ms
        self.was_cached = was_cached
        self.token_usage = token_usage
        self.retry_count = retry_count
        self.cost_saved_usd = cost_saved_usd
        self.time_saved_ms = time_saved_ms


def step(
    name: str,
    max_retries: int = 0,
    retry_delay: float = 1.0,
) -> Callable:
    """Decorator that wraps an async function as a Sylo pipeline step.

    The decorated function receives a `sylo.Context` as its first argument.
    The decorator handles:
    - Checking for an existing checkpoint (skip if found)
    - Executing the function and saving a new checkpoint
    - Extracting token usage from the return value
    - Retrying on failure with exponential backoff
    - Recording step timing

    Args:
        name: Unique name for this step within the pipeline.
        max_retries: Maximum number of retry attempts on failure. Default 0.
        retry_delay: Base delay in seconds between retries. Doubles each retry.

    Returns:
        Decorated async function.

    Example:
        @sylo.step("fetch-emails", max_retries=3, retry_delay=2.0)
        async def fetch_emails(ctx: sylo.Context) -> dict:
            result = await call_llm(...)
            return result
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(ctx: Context, *args: Any, **kwargs: Any) -> Any:
            # Import here to avoid circular imports
            from sylo.core.pipeline import _current_pipeline

            pipeline = _current_pipeline.get(None)
            if pipeline is None:
                # Running outside a pipeline context — just call the function
                return await func(ctx, *args, **kwargs)

            storage = pipeline._storage
            step_index = pipeline._step_counter
            pipeline._step_counter += 1

            # Compute input hash for cache invalidation
            input_hash = _compute_input_hash(args, kwargs)

            # Check for existing checkpoint (resumption logic)
            existing_checkpoint = None
            if storage is not None:
                existing_checkpoint = await pipeline._safe_storage_op(
                    storage.get_checkpoint, pipeline.execution_id, name
                )

            if (
                existing_checkpoint is not None
                and existing_checkpoint.status == CheckpointStatus.COMPLETED
            ):
                # Checkpoint hit — skip re-execution
                logger.info(
                    "Step '%s' loaded from checkpoint (skipped)", name
                )

                # Emit STEP_SKIPPED + CHECKPOINT_LOADED audit events
                await pipeline._emit_audit_event(
                    event_type="STEP_SKIPPED",
                    step_name=name,
                    data={
                        "reason": "checkpoint_hit",
                        "checkpoint_id": existing_checkpoint.checkpoint_id,
                        "original_duration_ms": existing_checkpoint.duration_ms,
                        "original_cost_usd": (
                            existing_checkpoint.token_usage.estimated_cost_usd
                            if existing_checkpoint.token_usage
                            else 0.0
                        ),
                    },
                )
                await pipeline._emit_audit_event(
                    event_type="CHECKPOINT_LOADED",
                    step_name=name,
                    data={
                        "checkpoint_id": existing_checkpoint.checkpoint_id,
                    },
                )

                # Add to context for subsequent steps
                ctx.previous_outputs[name] = existing_checkpoint.output

                # Record as cached step result
                cost_saved = (
                    existing_checkpoint.token_usage.estimated_cost_usd
                    if existing_checkpoint.token_usage
                    else 0.0
                )
                step_result = StepResult(
                    step_name=name,
                    output=existing_checkpoint.output,
                    duration_ms=0,
                    was_cached=True,
                    token_usage=existing_checkpoint.token_usage,
                    retry_count=0,
                    cost_saved_usd=cost_saved,
                    time_saved_ms=existing_checkpoint.duration_ms,
                )
                pipeline._step_results.append(step_result)

                return existing_checkpoint.output

            # No checkpoint — execute the step
            await pipeline._emit_audit_event(
                event_type="STEP_STARTED",
                step_name=name,
                data={"step_index": step_index, "max_retries": max_retries},
            )

            # Setup trust declarations for the step run
            ctx._current_step_name = name
            ctx._trust_declarations = getattr(func, "_sylo_trust_declarations", getattr(func, "_luro_trust_declarations", None))
            ctx._permissions_used.clear()
            ctx._violations_attempted = 0

            config = get_config()
            if ctx._trust_declarations is None and config.is_production:
                logger.warning(
                    "⚠ Sylo Trust: Step \"%s\" has no trust declaration. Running without enforcement.",
                    name,
                )

            last_error: Exception | None = None
            retry_count = 0
            result: Any = None
            step_started = datetime.now(timezone.utc)

            try:
                for attempt in range(max_retries + 1):
                    try:
                        attempt_start = datetime.now(timezone.utc)
                        result = await func(ctx, *args, **kwargs)
                        last_error = None
                        break  # Success
                    except Exception as exc:
                        last_error = exc
                        retry_count = attempt + 1

                        if attempt < max_retries:
                            delay = retry_delay * (2**attempt)
                            logger.warning(
                                "Step '%s' failed (attempt %d/%d), retrying in %.1fs: %s",
                                name,
                                attempt + 1,
                                max_retries + 1,
                                delay,
                                exc,
                            )

                            await pipeline._emit_audit_event(
                                event_type="STEP_RETRIED",
                                step_name=name,
                                data={
                                    "attempt": attempt + 1,
                                    "max_retries": max_retries,
                                    "error": str(exc),
                                    "retry_delay": delay,
                                },
                            )

                            await asyncio.sleep(delay)
                        else:
                            # All retries exhausted
                            retry_count = attempt + 1
            finally:
                # This always runs after step attempts are complete, whether success or fail
                # Emit trust summary and warnings if trust was active
                if ctx._trust_declarations is not None:
                    declared_list = []
                    used_list = []
                    unused_list = []

                    for action, patterns in ctx._trust_declarations.items():
                        for pattern in patterns:
                            perm_str = f"{action}:{pattern}"
                            declared_list.append(perm_str)
                            if (action, pattern) in ctx._permissions_used:
                                used_list.append(perm_str)
                            else:
                                unused_list.append(perm_str)
                                # Log least privilege warning in dev mode
                                if config.is_development:
                                    logger.warning(
                                        "⚠ Sylo Trust: Step \"%s\" declared \"%s\" but never accessed it.\n"
                                        "  Consider removing unused permissions.",
                                        name,
                                        pattern,
                                    )

                    await pipeline._emit_audit_event(
                        event_type="TRUST_SUMMARY",
                        step_name=name,
                        data={
                            "declared_permissions": declared_list,
                            "permissions_used": used_list,
                            "permissions_unused": unused_list,
                            "violations_attempted": ctx._violations_attempted,
                        },
                    )

            step_ended = datetime.now(timezone.utc)
            duration_ms = int(
                (step_ended - step_started).total_seconds() * 1000
            )

            if last_error is not None:
                # Step failed after all retries
                failed_checkpoint = Checkpoint(
                    execution_id=pipeline.execution_id,
                    step_name=name,
                    step_index=step_index,
                    status=CheckpointStatus.FAILED,
                    input_hash=input_hash,
                    output={},
                    started_at=step_started,
                    completed_at=step_ended,
                    duration_ms=duration_ms,
                    retry_count=retry_count,
                )

                if storage is not None:
                    await pipeline._safe_storage_op(
                        storage.save_checkpoint, failed_checkpoint
                    )

                await pipeline._emit_audit_event(
                    event_type="STEP_FAILED",
                    step_name=name,
                    duration_ms=duration_ms,
                    data={
                        "error_type": type(last_error).__name__,
                        "error_message": str(last_error),
                        "retry_count": retry_count,
                    },
                )

                step_result = StepResult(
                    step_name=name,
                    output=None,
                    duration_ms=duration_ms,
                    was_cached=False,
                    token_usage=None,
                    retry_count=retry_count,
                )
                pipeline._step_results.append(step_result)

                raise last_error

            # Step succeeded — extract token usage and save checkpoint
            token_usage = extract_token_usage(result) if isinstance(result, dict) else None
            token_usage_from_manual = False

            # Check for manually recorded token usage via ctx.record_token_usage()
            if token_usage is None and hasattr(ctx, '_recorded_token_usage') and ctx._recorded_token_usage is not None:
                token_usage = ctx._recorded_token_usage
                token_usage_from_manual = True
            # Clear for next step
            ctx._recorded_token_usage = None

            completed_checkpoint = Checkpoint(
                execution_id=pipeline.execution_id,
                step_name=name,
                step_index=step_index,
                status=CheckpointStatus.COMPLETED,
                input_hash=input_hash,
                output=result if isinstance(result, dict) else {"result": result},
                started_at=step_started,
                completed_at=step_ended,
                duration_ms=duration_ms,
                token_usage=token_usage or TokenUsage(),
                retry_count=max(0, retry_count - 1),  # Don't count the successful attempt
            )

            if storage is not None:
                await pipeline._safe_storage_op(
                    storage.save_checkpoint, completed_checkpoint
                )

            # Record token usage event
            if token_usage is not None:
                await pipeline._emit_audit_event(
                    event_type="TOKEN_USAGE_RECORDED",
                    step_name=name,
                    data={
                        "prompt_tokens": token_usage.prompt_tokens,
                        "completion_tokens": token_usage.completion_tokens,
                        "total_tokens": token_usage.total_tokens,
                        "model": token_usage.model,
                        "estimated_cost_usd": token_usage.estimated_cost_usd,
                    },
                )

                # Update pipeline-level token cost (skip if already done via ctx.record_token_usage)
                if pipeline.record is not None and not token_usage_from_manual:
                    pipeline.record.token_cost.total_tokens += token_usage.total_tokens
                    pipeline.record.token_cost.estimated_cost_usd += token_usage.estimated_cost_usd

            # Emit STEP_COMPLETED and CHECKPOINT_SAVED events
            await pipeline._emit_audit_event(
                event_type="STEP_COMPLETED",
                step_name=name,
                duration_ms=duration_ms,
                data={
                    "step_index": step_index,
                    "has_token_usage": token_usage is not None,
                },
            )
            await pipeline._emit_audit_event(
                event_type="CHECKPOINT_SAVED",
                step_name=name,
                data={
                    "checkpoint_id": completed_checkpoint.checkpoint_id,
                },
            )

            # Store output for subsequent steps
            output_dict = result if isinstance(result, dict) else {"result": result}
            ctx.previous_outputs[name] = output_dict

            step_result = StepResult(
                step_name=name,
                output=output_dict,
                duration_ms=duration_ms,
                was_cached=False,
                token_usage=token_usage,
                retry_count=max(0, retry_count - 1),
            )
            pipeline._step_results.append(step_result)

            logger.info(
                "Step '%s' completed in %dms", name, duration_ms
            )

            return result

        # Attach step metadata to the wrapper for introspection
        wrapper._sylo_step_name = name  # type: ignore[attr-defined]
        wrapper._luro_step_name = name  # backwards compat
        wrapper._luro_max_retries = max_retries  # backwards compat
        wrapper._luro_retry_delay = retry_delay  # backwards compat

        return wrapper

    return decorator


def _compute_input_hash(*args: Any, **kwargs: Any) -> str:
    """Compute a SHA-256 hash of the step inputs for cache invalidation.

    Args:
        *args: Positional arguments to the step function.
        **kwargs: Keyword arguments to the step function.

    Returns:
        Hex string of the SHA-256 hash.
    """
    try:
        data = json.dumps(
            {"args": str(args), "kwargs": str(kwargs)},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(data.encode()).hexdigest()
    except Exception:
        return hashlib.sha256(b"unhashable").hexdigest()
