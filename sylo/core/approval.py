"""Human Approval Gates — Brief 04.

Pauses pipeline execution before irreversible or dangerous actions,
waiting for a human to explicitly approve or reject before continuing.

Features:
- @sylo.requires_approval decorator
- Notification dispatch (Email, Slack, Webhook with HMAC)
- Polling mechanism with configurable intervals and timeout policies
- Local development HTTP server on port 7749
- Programmatic helpers: sylo.approve() and sylo.reject()
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import hmac
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Literal

import httpx

from sylo.config import get_config
from sylo.core.context import Context
from sylo.exceptions import SyloApprovalRejectedError, SyloError
from sylo.models import ApprovalRequest, ApprovalStatus, CheckpointStatus, ExecutionStatus
from sylo.storage import get_storage

logger = logging.getLogger("sylo")

# Global state for local dev server
_server_lock = threading.Lock()
_server_instance: HTTPServer | None = None
_server_thread: threading.Thread | None = None


class SafeDict(dict):
    """Dictionary that returns `{key}` for missing template variables."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


def requires_approval(
    title: str,
    description: str,
    action_class: str = "destructive",
    timeout_hours: float = 24.0,
    on_timeout: Literal["abort", "auto_approve", "escalate"] = "abort",
    notify: list[str] | None = None,
    metadata_keys: list[str] | None = None,
    poll_interval_seconds: float = 30.0,
) -> Callable:
    """Decorator that pauses pipeline execution awaiting human approval.

    Args:
        title: Short title summarizing the action requiring approval.
        description: Template string describing the action. Supports formatting
            with variables from context metadata or prior step outputs.
        action_class: Category of action ("destructive", "financial", "external", etc.).
        timeout_hours: Hours before the approval request expires.
        on_timeout: Behavior on expiry ("abort", "auto_approve", "escalate").
        notify: List of notification channels to trigger ("email", "slack", "webhook").
        metadata_keys: Keys from context metadata to include in context snapshot.
        poll_interval_seconds: Seconds between polling checks for approval decision.

    Returns:
        Decorated async function.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(ctx: Context, *args: Any, **kwargs: Any) -> Any:
            from sylo.core.pipeline import _current_pipeline

            pipeline = _current_pipeline.get(None)
            if pipeline is None:
                # Running outside pipeline context — execute directly
                return await func(ctx, *args, **kwargs)

            storage = pipeline._storage
            step_name = getattr(func, "_sylo_step_name", getattr(func, "_luro_step_name", func.__name__))

            # If checkpoint exists and completed, skip approval gate entirely
            if storage is not None:
                existing_cp = await pipeline._safe_storage_op(
                    storage.get_checkpoint, pipeline.execution_id, step_name
                )
                if existing_cp is not None and existing_cp.status == CheckpointStatus.COMPLETED:
                    return await func(ctx, *args, **kwargs)

            # Check if an approval request already exists for this step
            req: ApprovalRequest | None = None
            if storage is not None:
                req = await pipeline._safe_storage_op(
                    storage.get_approval_request_by_step, pipeline.execution_id, step_name
                )

            if req is None:
                # Build context snapshot
                context_snapshot: dict[str, Any] = {}
                if metadata_keys:
                    for key in metadata_keys:
                        if key in ctx.metadata:
                            context_snapshot[key] = ctx.metadata[key]
                        elif key in ctx.previous_outputs:
                            context_snapshot[key] = ctx.previous_outputs[key]
                        elif key in kwargs:
                            context_snapshot[key] = kwargs[key]

                # Format description template
                formatted_desc = description.format_map(SafeDict(context_snapshot))

                now = datetime.now(timezone.utc)
                expires_at = now + timedelta(hours=timeout_hours)

                req = ApprovalRequest(
                    execution_id=pipeline.execution_id,
                    pipeline_name=pipeline.name,
                    step_name=step_name,
                    title=title,
                    description=formatted_desc,
                    action_class=action_class,
                    status=ApprovalStatus.PENDING,
                    created_at=now,
                    expires_at=expires_at,
                    context_snapshot=context_snapshot,
                    notify_channels=notify or [],
                )

                if storage is not None:
                    await pipeline._safe_storage_op(storage.save_approval_request, req)

                if pipeline.record is not None:
                    pipeline.record.status = ExecutionStatus.AWAITING_APPROVAL
                    if storage is not None:
                        await pipeline._safe_storage_op(storage.save_execution, pipeline.record)

                await pipeline._emit_audit_event(
                    event_type="APPROVAL_REQUESTED",
                    step_name=step_name,
                    data={
                        "approval_id": req.approval_id,
                        "title": req.title,
                        "action_class": req.action_class,
                        "expires_at": req.expires_at.isoformat(),
                    },
                )

                # Trigger notifications
                await _send_notifications(req)

                config = get_config()
                if config.is_development or config.storage == "local":
                    _print_console_notice(req)
                    start_local_server(port=7749)

            # Poll for decision
            while req.status == ApprovalStatus.PENDING:
                now = datetime.now(timezone.utc)
                if now >= req.expires_at:
                    req.status = ApprovalStatus.TIMED_OUT
                    if storage is not None:
                        await pipeline._safe_storage_op(storage.save_approval_request, req)
                    break

                await asyncio.sleep(poll_interval_seconds)

                if storage is not None:
                    updated_req = await pipeline._safe_storage_op(
                        storage.get_approval_request, req.approval_id
                    )
                    if updated_req is not None and updated_req.status != ApprovalStatus.PENDING:
                        req = updated_req
                        break

            # Handle decision
            decided_time = req.decided_at or datetime.now(timezone.utc)
            time_to_decision_minutes = round(
                (decided_time - req.created_at).total_seconds() / 60.0, 2
            )

            if req.status == ApprovalStatus.APPROVED:
                await pipeline._emit_audit_event(
                    event_type="APPROVAL_DECISION",
                    step_name=step_name,
                    data={
                        "approval_id": req.approval_id,
                        "decision": "APPROVED",
                        "decided_by": req.decided_by or "unknown",
                        "decision_note": req.decision_note or "",
                        "time_to_decision_minutes": time_to_decision_minutes,
                    },
                )
                if pipeline.record is not None:
                    pipeline.record.status = ExecutionStatus.RUNNING
                    if storage is not None:
                        await pipeline._safe_storage_op(storage.save_execution, pipeline.record)
                return await func(ctx, *args, **kwargs)

            elif req.status == ApprovalStatus.REJECTED:
                await pipeline._emit_audit_event(
                    event_type="APPROVAL_DECISION",
                    step_name=step_name,
                    data={
                        "approval_id": req.approval_id,
                        "decision": "REJECTED",
                        "decided_by": req.decided_by or "unknown",
                        "decision_note": req.decision_note or "",
                        "time_to_decision_minutes": time_to_decision_minutes,
                    },
                )
                if pipeline.record is not None:
                    pipeline.record.status = ExecutionStatus.FAILED
                    if storage is not None:
                        await pipeline._safe_storage_op(storage.save_execution, pipeline.record)
                raise SyloApprovalRejectedError(
                    f"Approval request '{req.approval_id}' for step '{step_name}' was rejected."
                )

            elif req.status == ApprovalStatus.TIMED_OUT:
                if on_timeout == "auto_approve":
                    req.status = ApprovalStatus.APPROVED
                    req.decided_by = "timeout_auto_approve"
                    req.decided_at = datetime.now(timezone.utc)
                    if storage is not None:
                        await pipeline._safe_storage_op(storage.save_approval_request, req)

                    await pipeline._emit_audit_event(
                        event_type="APPROVAL_DECISION",
                        step_name=step_name,
                        data={
                            "approval_id": req.approval_id,
                            "decision": "APPROVED",
                            "decided_by": "timeout_auto_approve",
                            "decision_note": "Auto-approved on timeout",
                            "time_to_decision_minutes": time_to_decision_minutes,
                        },
                    )
                    if pipeline.record is not None:
                        pipeline.record.status = ExecutionStatus.RUNNING
                        if storage is not None:
                            await pipeline._safe_storage_op(storage.save_execution, pipeline.record)
                    return await func(ctx, *args, **kwargs)
                else:
                    await pipeline._emit_audit_event(
                        event_type="APPROVAL_DECISION",
                        step_name=step_name,
                        data={
                            "approval_id": req.approval_id,
                            "decision": "REJECTED",
                            "decided_by": "timeout",
                            "decision_note": f"Timed out ({on_timeout})",
                            "time_to_decision_minutes": time_to_decision_minutes,
                        },
                    )
                    if pipeline.record is not None:
                        pipeline.record.status = ExecutionStatus.FAILED
                        if storage is not None:
                            await pipeline._safe_storage_op(storage.save_execution, pipeline.record)
                    raise SyloApprovalRejectedError(
                        f"Approval request '{req.approval_id}' for step '{step_name}' timed out ({on_timeout})."
                    )

            raise SyloApprovalRejectedError(
                f"Unhandled approval status '{req.status}' for step '{step_name}'."
            )

        return wrapper

    return decorator


async def approve(
    approval_id: str, decided_by: str = "developer", note: str | None = None
) -> ApprovalRequest:
    """Programmatically approve a pending approval request."""
    config = get_config()
    storage = get_storage(config)
    req = await storage.get_approval_request(approval_id)
    if req is None:
        raise SyloError(f"Approval request '{approval_id}' not found.")
    if req.status != ApprovalStatus.PENDING:
        raise SyloError(f"Approval request '{approval_id}' is already {req.status.value}.")

    req.status = ApprovalStatus.APPROVED
    req.decided_at = datetime.now(timezone.utc)
    req.decided_by = decided_by
    req.decision_note = note
    await storage.save_approval_request(req)
    logger.info("Approval request %s approved by %s", approval_id, decided_by)
    return req


async def reject(
    approval_id: str, decided_by: str = "developer", note: str | None = None
) -> ApprovalRequest:
    """Programmatically reject a pending approval request."""
    config = get_config()
    storage = get_storage(config)
    req = await storage.get_approval_request(approval_id)
    if req is None:
        raise SyloError(f"Approval request '{approval_id}' not found.")
    if req.status != ApprovalStatus.PENDING:
        raise SyloError(f"Approval request '{approval_id}' is already {req.status.value}.")

    req.status = ApprovalStatus.REJECTED
    req.decided_at = datetime.now(timezone.utc)
    req.decided_by = decided_by
    req.decision_note = note
    await storage.save_approval_request(req)
    logger.info("Approval request %s rejected by %s", approval_id, decided_by)
    return req


def _print_console_notice(req: ApprovalRequest) -> None:
    """Print the human-readable approval prompt to console."""
    hours = round((req.expires_at - req.created_at).total_seconds() / 3600, 1)
    lines = [
        "",
        "⏸ Sylo Approval Required",
        f"  Pipeline: {req.pipeline_name}",
        f"  Step: {req.step_name}",
        f"  Action: {req.description} ({req.action_class.upper()})",
        "",
        f"  Approve: http://localhost:7749/approve/{req.approval_id}",
        f"  Reject:  http://localhost:7749/reject/{req.approval_id}",
        "",
        f"  Expires in: {hours} hours",
        "  Waiting for decision...",
        "",
    ]
    print("\n".join(lines))


async def _send_notifications(req: ApprovalRequest) -> None:
    """Dispatch notifications to configured channels."""
    config = get_config()
    notifications = config.notifications or {}
    channels = req.notify_channels

    if not channels or not notifications:
        return

    async with httpx.AsyncClient(timeout=10.0) as client:
        for channel in channels:
            try:
                if channel == "email" and "email" in notifications:
                    email_cfg = notifications["email"]
                    if email_cfg.get("provider") == "resend" and email_cfg.get("api_key"):
                        await client.post(
                            "https://api.resend.com/emails",
                            headers={"Authorization": f"Bearer {email_cfg['api_key']}"},
                            json={
                                "from": email_cfg.get("from", "saketjndl2005@gmail.com"),
                                "to": email_cfg.get("to", []),
                                "subject": f"[Sylo Approval Required] {req.title}",
                                "html": f"<p>{req.description}</p><p>Approve ID: {req.approval_id}</p>",
                            },
                        )
                elif channel == "slack" and "slack" in notifications:
                    slack_cfg = notifications["slack"]
                    if slack_cfg.get("webhook_url"):
                        await client.post(
                            slack_cfg["webhook_url"],
                            json={
                                "text": f"⏸ *Sylo Approval Required*: {req.title}\n{req.description}\n`{req.approval_id}`"
                            },
                        )
                elif channel == "webhook" and "webhook" in notifications:
                    wh_cfg = notifications["webhook"]
                    url = wh_cfg.get("url")
                    secret = wh_cfg.get("secret", "").encode("utf-8")
                    if url:
                        payload = req.model_dump_json().encode("utf-8")
                        headers = {"Content-Type": "application/json"}
                        if secret:
                            sig = hmac.new(secret, payload, hashlib.sha256).hexdigest()
                            headers["X-Sylo-Signature"] = f"sha256={sig}"
                        await client.post(url, headers=headers, content=payload)
            except Exception as exc:
                logger.warning("Failed to send %s notification: %s", channel, exc)


class ApprovalHandler(BaseHTTPRequestHandler):
    """HTTP request handler for local development approval endpoints."""

    def do_GET(self) -> None:
        parts = [p for p in self.path.strip("/").split("/") if p]
        if len(parts) == 2 and parts[0] in ("approve", "reject"):
            action, approval_id = parts[0], parts[1]
            try:
                asyncio.run(
                    approve(approval_id, decided_by="local_developer")
                    if action == "approve"
                    else reject(approval_id, decided_by="local_developer")
                )
                status_word = "Approved" if action == "approve" else "Rejected"
                color = "#10B981" if action == "approve" else "#EF4444"
                html = f"""<!DOCTYPE html>
<html><head><title>Sylo Approval</title>
<style>body {{ font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #F9FAFB; }}
.card {{ background: white; padding: 2rem; border-radius: 0.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); text-align: center; }}
h1 {{ color: {color}; }}</style></head>
<body><div class="card"><h1>{status_word}</h1><p>Request <code>{approval_id}</code> has been {status_word.lower()}.</p><p>You may close this window.</p></div></body></html>"""
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))
            except Exception as exc:
                self.send_response(400)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Error processing request: {exc}".encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("LocalServer: " + format, *args)


def start_local_server(port: int = 7749) -> None:
    """Start the local approval HTTP server in a daemon thread if not running."""
    global _server_instance, _server_thread
    with _server_lock:
        if _server_instance is not None:
            return
        try:
            server = HTTPServer(("localhost", port), ApprovalHandler)
            _server_instance = server
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            _server_thread = thread
            logger.debug("Local approval server started on http://localhost:%d", port)
        except OSError as exc:
            logger.debug("Could not bind local approval server to port %d: %s", port, exc)


def stop_local_server() -> None:
    """Stop the local approval HTTP server."""
    global _server_instance, _server_thread
    with _server_lock:
        if _server_instance is not None:
            try:
                _server_instance.shutdown()
                _server_instance.server_close()
            except Exception:
                pass
            _server_instance = None
            _server_thread = None
