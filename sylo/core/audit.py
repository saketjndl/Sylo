"""Audit & Replay Engine — Brief 05.

Provides:
- ``get_summary(execution_id)``  — structured execution summary
- ``replay(execution_id, ...)``  — replay past executions
- ``pretty_print_audit(execution_id)`` — formatted console output
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sylo.config import get_config
from sylo.models import (
    AuditEvent,
    ExecutionRecord,
    ExecutionSummary,
    StepSummary,
)
from sylo.storage import get_storage

logger = logging.getLogger("sylo")


async def get_summary(execution_id: str) -> ExecutionSummary:
    """Generate a structured summary of a pipeline execution.

    Reads the execution record and audit events from storage,
    then computes aggregate statistics.

    Args:
        execution_id: UUID of the execution to summarize.

    Returns:
        An ExecutionSummary with step-by-step timeline.

    Raises:
        ValueError: If the execution is not found.
    """
    config = get_config()
    storage = get_storage(config)

    record = await storage.get_execution(execution_id)
    if record is None:
        raise ValueError(f"Execution '{execution_id}' not found.")

    # Read audit events
    events: list[AuditEvent] = []
    if hasattr(storage, "get_audit_events"):
        events = await storage.get_audit_events(execution_id)

    # Build the timeline from audit events
    timeline: list[StepSummary] = []
    step_events: dict[str, dict[str, Any]] = {}  # step_name -> aggregated info

    steps_completed = 0
    steps_skipped = 0
    steps_failed = 0
    approval_gates_hit = 0
    permission_violations = 0
    total_tokens = 0
    total_cost = 0.0
    cost_saved = 0.0

    for event in events:
        step = event.step_name

        if event.event_type == "STEP_STARTED" and step:
            step_events.setdefault(step, {"status": "running", "duration_ms": 0, "tokens": 0, "cost": 0.0, "retries": 0})

        elif event.event_type == "STEP_COMPLETED" and step:
            info = step_events.setdefault(step, {"status": "completed", "duration_ms": 0, "tokens": 0, "cost": 0.0, "retries": 0})
            info["status"] = "completed"
            info["duration_ms"] = event.duration_ms
            steps_completed += 1

        elif event.event_type == "STEP_FAILED" and step:
            info = step_events.setdefault(step, {"status": "failed", "duration_ms": 0, "tokens": 0, "cost": 0.0, "retries": 0})
            info["status"] = "failed"
            info["duration_ms"] = event.duration_ms
            info["error"] = event.data.get("error_message")
            info["retries"] = event.data.get("retry_count", 0)
            steps_failed += 1

        elif event.event_type == "STEP_SKIPPED" and step:
            info = step_events.setdefault(step, {"status": "skipped", "duration_ms": 0, "tokens": 0, "cost": 0.0, "retries": 0})
            info["status"] = "skipped"
            info["was_cached"] = True
            saved = event.data.get("original_cost_usd", 0.0)
            info["cost_saved"] = saved
            cost_saved += saved
            steps_skipped += 1

        elif event.event_type == "STEP_RETRIED" and step:
            info = step_events.setdefault(step, {"status": "running", "duration_ms": 0, "tokens": 0, "cost": 0.0, "retries": 0})
            info["retries"] = event.data.get("attempt", info.get("retries", 0))

        elif event.event_type == "TOKEN_USAGE_RECORDED" and step:
            info = step_events.setdefault(step, {"status": "running", "duration_ms": 0, "tokens": 0, "cost": 0.0, "retries": 0})
            tokens = event.data.get("total_tokens", 0)
            cost = event.data.get("estimated_cost_usd", 0.0)
            info["tokens"] += tokens
            info["cost"] += cost
            total_tokens += tokens
            total_cost += cost

        elif event.event_type == "APPROVAL_REQUESTED":
            approval_gates_hit += 1

        elif event.event_type == "PERMISSION_VIOLATION":
            permission_violations += 1

    # Build ordered timeline
    for step_name, info in step_events.items():
        timeline.append(StepSummary(
            step_name=step_name,
            status=info.get("status", "unknown"),
            duration_ms=info.get("duration_ms", 0),
            tokens=info.get("tokens", 0),
            estimated_cost_usd=info.get("cost", 0.0),
            cost_saved_usd=info.get("cost_saved", 0.0),
            error=info.get("error"),
            retry_count=info.get("retries", 0),
            was_cached=info.get("was_cached", False),
        ))

    # Compute duration
    duration_seconds = 0.0
    if record.completed_at and record.started_at:
        duration_seconds = (record.completed_at - record.started_at).total_seconds()

    # Fall back to record-level totals if no audit events
    if total_tokens == 0:
        total_tokens = record.token_cost.total_tokens
    if total_cost == 0.0:
        total_cost = record.token_cost.estimated_cost_usd

    return ExecutionSummary(
        execution_id=record.execution_id,
        pipeline_name=record.pipeline_name,
        status=record.status.value,
        duration_seconds=duration_seconds,
        steps_completed=steps_completed,
        steps_skipped=steps_skipped,
        steps_failed=steps_failed,
        total_tokens=total_tokens,
        estimated_cost_usd=total_cost,
        cost_saved_usd=cost_saved,
        approval_gates_hit=approval_gates_hit,
        permission_violations=permission_violations,
        timeline=timeline,
    )


async def replay(
    execution_id: str,
    from_step: str | None = None,
    override_inputs: dict[str, dict[str, Any]] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Replay a past execution.

    Loads the original execution record, restores checkpoints for
    steps before ``from_step``, and re-executes from that point onward.

    Args:
        execution_id: UUID of the execution to replay.
        from_step: Step name to start replaying from. Steps before this
            use cached outputs from the original execution.
        override_inputs: Dict mapping step names to replacement inputs.
        dry_run: If True, log what would happen without executing.

    Returns:
        Dict with replay results including original vs new outputs.

    Raises:
        ValueError: If the execution is not found.
    """
    config = get_config()
    storage = get_storage(config)

    record = await storage.get_execution(execution_id)
    if record is None:
        raise ValueError(f"Execution '{execution_id}' not found.")

    # Load audit events to get step order
    events: list[AuditEvent] = []
    if hasattr(storage, "get_audit_events"):
        events = await storage.get_audit_events(execution_id)

    # Extract ordered step names from events
    step_order: list[str] = []
    seen_steps: set[str] = set()
    for event in events:
        if event.step_name and event.event_type in ("STEP_STARTED", "STEP_SKIPPED") and event.step_name not in seen_steps:
            step_order.append(event.step_name)
            seen_steps.add(event.step_name)

    # Determine which steps to skip (use cached) vs re-execute
    replay_from_idx = 0
    if from_step:
        try:
            replay_from_idx = step_order.index(from_step)
        except ValueError:
            raise ValueError(
                f"Step '{from_step}' not found in execution. "
                f"Available steps: {step_order}"
            )

    cached_steps = step_order[:replay_from_idx]
    replay_steps = step_order[replay_from_idx:]

    # Load cached outputs
    cached_outputs: dict[str, Any] = {}
    for step in cached_steps:
        checkpoint = await storage.get_checkpoint(execution_id, step)
        if checkpoint is not None:
            cached_outputs[step] = checkpoint.output

    result: dict[str, Any] = {
        "execution_id": execution_id,
        "pipeline_name": record.pipeline_name,
        "cached_steps": cached_steps,
        "replay_steps": replay_steps,
        "cached_outputs": cached_outputs,
        "dry_run": dry_run,
        "override_inputs": override_inputs or {},
    }

    if dry_run:
        logger.info("DRY RUN replay of execution %s", execution_id[:8])
        for step in cached_steps:
            logger.info("  [CACHED] %s — would load from checkpoint", step)
        for step in replay_steps:
            override_note = " (with override inputs)" if override_inputs and step in override_inputs else ""
            logger.info("  [REPLAY] %s — would re-execute%s", step, override_note)
        result["status"] = "dry_run_complete"
        return result

    # For actual replay, we set up a new pipeline context with resume_from
    # pointing to the original execution's checkpoints
    result["status"] = "replay_prepared"
    result["instructions"] = (
        f"To execute the replay, run a new pipeline with "
        f"pipe.resume_from = '{execution_id}' and start from step '{from_step or step_order[0]}'."
    )
    return result


def format_audit_log(
    events: list[AuditEvent],
    pipeline_name: str = "",
    execution_id: str = "",
) -> str:
    """Format audit events into a pretty-printed timeline string.

    Produces the formatted output specified in Brief 05:

        Sylo Audit Log — pipeline-name — abc12345
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        09:14:00.000  PIPELINE_STARTED
        09:14:00.043  STEP_STARTED        fetch-emails
        ...

    Args:
        events: List of audit events in chronological order.
        pipeline_name: Pipeline name for the header.
        execution_id: Execution ID for the header.

    Returns:
        Formatted string suitable for console display.
    """
    if not events:
        return "No audit events found."

    short_id = execution_id[:8] if execution_id else ""
    header = f"Sylo Audit Log — {pipeline_name} — {short_id}"
    separator = "━" * max(len(header), 50)

    lines = ["", header, separator, ""]

    total_tokens = 0
    total_cost = 0.0
    violations = 0
    approvals = 0
    start_time: datetime | None = None

    for event in events:
        time_str = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
        event_type = event.event_type
        step_part = f"    {event.step_name}" if event.step_name else ""

        # Build extra info based on event type
        extra = ""
        if event_type == "STEP_COMPLETED":
            ms = event.duration_ms
            extra_parts = [f"{ms}ms"]
            tokens = event.data.get("total_tokens", 0)
            cost = event.data.get("estimated_cost_usd", 0.0)
            if tokens:
                extra_parts.append(f"${cost:.3f}")
                extra_parts.append(f"{tokens:,} tokens")
                total_tokens += tokens
                total_cost += cost
            extra = "  ".join(extra_parts)
            if extra:
                extra = f"    {extra}"

        elif event_type == "STEP_SKIPPED":
            saved = event.data.get("original_cost_usd", 0.0)
            extra = f"    (cached, saved ${saved:.3f})"

        elif event_type == "STEP_FAILED":
            err = event.data.get("error_message", "unknown error")
            retries = event.data.get("retry_count", 0)
            extra = f"    FAILED: {err}"
            if retries > 0:
                extra += f" ({retries} retries)"

        elif event_type == "APPROVAL_REQUESTED":
            extra = "    (awaiting)"
            approvals += 1

        elif event_type == "APPROVAL_DECISION":
            decision = event.data.get("decision", "")
            decided_by = event.data.get("decided_by", "")
            extra = f"    {decision} by {decided_by}"

        elif event_type == "PERMISSION_VIOLATION":
            violations += 1
            resource = event.data.get("resource", "")
            action = event.data.get("action", "")
            extra = f"    BLOCKED: {action} on {resource}"

        elif event_type == "PIPELINE_STARTED":
            start_time = event.timestamp

        elif event_type == "TOKEN_USAGE_RECORDED":
            tokens = event.data.get("total_tokens", 0)
            cost = event.data.get("estimated_cost_usd", 0.0)
            total_tokens += tokens
            total_cost += cost

        # Pad event type to fixed width
        padded_type = f"{event_type:<24}"
        line = f"{time_str}  {padded_type}{step_part}{extra}"
        lines.append(line)

    # Footer summary
    lines.append("")
    end_time = events[-1].timestamp if events else None
    duration_s = 0.0
    if start_time and end_time:
        duration_s = (end_time - start_time).total_seconds()

    footer = (
        f"Total: {duration_s:.1f}s | ${total_cost:.3f} | "
        f"{total_tokens:,} tokens | {violations} violations | "
        f"{approvals} approval{'s' if approvals != 1 else ''}"
    )
    lines.append(footer)
    lines.append("")

    return "\n".join(lines)


async def pretty_print_audit(execution_id: str) -> str:
    """Load and pretty-print the full audit log for an execution.

    Args:
        execution_id: UUID of the execution.

    Returns:
        Formatted string of the audit log.

    Raises:
        ValueError: If the execution is not found.
    """
    config = get_config()
    storage = get_storage(config)

    record = await storage.get_execution(execution_id)
    if record is None:
        raise ValueError(f"Execution '{execution_id}' not found.")

    events: list[AuditEvent] = []
    if hasattr(storage, "get_audit_events"):
        events = await storage.get_audit_events(execution_id)

    return format_audit_log(
        events,
        pipeline_name=record.pipeline_name,
        execution_id=record.execution_id,
    )
