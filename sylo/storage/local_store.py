"""Local filesystem storage backend for Sylo SDK.

Stores execution data as JSON files in ~/.sylo/executions/.
Designed for development and testing — no external dependencies required.

Directory structure:
    ~/.sylo/executions/{execution_id}/
        execution.json      — ExecutionRecord
        checkpoints/
            {step_name}.json — Checkpoint
        audit.jsonl          — Append-only audit events (one JSON object per line)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from sylo.models import ApprovalRequest, AuditEvent, Checkpoint, ExecutionRecord
from sylo.storage.base import SyloStorage

logger = logging.getLogger("sylo.storage.local")

# Default root directory for local storage
DEFAULT_ROOT = Path.home() / ".sylo" / "executions"


class LocalStorage(SyloStorage):
    """File-based storage backend for development and testing.

    All data is stored as human-readable JSON files on disk.
    Audit events use JSONL (JSON Lines) format for append-only writes.

    Args:
        root_dir: Base directory for storage. Defaults to ~/.luro/executions/
    """

    def __init__(self, root_dir: Path | str | None = None) -> None:
        import os
        if root_dir is None:
            env_dir = os.environ.get("SYLO_STORAGE_DIR")
            root_dir = Path(env_dir) if env_dir else DEFAULT_ROOT
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def _execution_dir(self, execution_id: str) -> Path:
        """Get the directory for a specific execution."""
        return self._root / execution_id

    def _execution_file(self, execution_id: str) -> Path:
        """Get the path to an execution's record file."""
        return self._execution_dir(execution_id) / "execution.json"

    def _checkpoint_dir(self, execution_id: str) -> Path:
        """Get the checkpoint directory for an execution."""
        return self._execution_dir(execution_id) / "checkpoints"

    def _checkpoint_file(self, execution_id: str, step_name: str) -> Path:
        """Get the path to a specific checkpoint file."""
        # Sanitize step_name for filesystem safety
        safe_name = step_name.replace("/", "_").replace("\\", "_")
        return self._checkpoint_dir(execution_id) / f"{safe_name}.json"

    def _audit_file(self, execution_id: str) -> Path:
        """Get the path to an execution's audit log file."""
        return self._execution_dir(execution_id) / "audit.jsonl"

    def _approval_dir(self, execution_id: str) -> Path:
        """Get the approval directory for an execution."""
        return self._execution_dir(execution_id) / "approvals"

    def _approval_file(self, execution_id: str, step_name: str) -> Path:
        """Get the path to a specific approval request file."""
        safe_name = step_name.replace("/", "_").replace("\\", "_")
        return self._approval_dir(execution_id) / f"{safe_name}.json"

    def _approval_index_file(self, approval_id: str) -> Path:
        """Get the path to an approval index file."""
        return self._root / "_approvals" / f"{approval_id}.json"

    async def save_execution(self, record: ExecutionRecord) -> None:
        """Save an execution record as a JSON file."""

        def _write() -> None:
            path = self._execution_file(record.execution_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

        await asyncio.to_thread(_write)
        logger.debug("Saved execution %s", record.execution_id)

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """Load an execution record from disk."""

        def _read() -> ExecutionRecord | None:
            path = self._execution_file(execution_id)
            if not path.exists():
                return None
            data = path.read_text(encoding="utf-8")
            return ExecutionRecord.model_validate_json(data)

        return await asyncio.to_thread(_read)

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint as a JSON file."""

        def _write() -> None:
            path = self._checkpoint_file(
                checkpoint.execution_id, checkpoint.step_name
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                checkpoint.model_dump_json(indent=2), encoding="utf-8"
            )

        await asyncio.to_thread(_write)
        logger.debug(
            "Saved checkpoint %s/%s",
            checkpoint.execution_id,
            checkpoint.step_name,
        )

    async def get_checkpoint(
        self, execution_id: str, step_name: str
    ) -> Checkpoint | None:
        """Load a checkpoint from disk."""

        def _read() -> Checkpoint | None:
            path = self._checkpoint_file(execution_id, step_name)
            if not path.exists():
                return None
            data = path.read_text(encoding="utf-8")
            return Checkpoint.model_validate_json(data)

        return await asyncio.to_thread(_read)

    async def list_executions(
        self, pipeline_name: str, limit: int = 20
    ) -> list[ExecutionRecord]:
        """List executions for a pipeline, sorted by start time (newest first)."""

        def _list() -> list[ExecutionRecord]:
            records: list[ExecutionRecord] = []
            if not self._root.exists():
                return records

            for exec_dir in self._root.iterdir():
                if not exec_dir.is_dir():
                    continue
                exec_file = exec_dir / "execution.json"
                if not exec_file.exists():
                    continue
                try:
                    data = exec_file.read_text(encoding="utf-8")
                    record = ExecutionRecord.model_validate_json(data)
                    if record.pipeline_name == pipeline_name:
                        records.append(record)
                except Exception:
                    logger.warning(
                        "Failed to read execution from %s", exec_file
                    )
                    continue

            # Sort by started_at descending
            records.sort(key=lambda r: r.started_at, reverse=True)
            return records[:limit]

        return await asyncio.to_thread(_list)

    async def append_audit_event(
        self, execution_id: str, event: AuditEvent
    ) -> None:
        """Append an audit event to the JSONL log file (append-only)."""

        def _append() -> None:
            path = self._audit_file(execution_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")

        await asyncio.to_thread(_append)
        logger.debug(
            "Appended audit event %s to execution %s",
            event.event_type,
            execution_id,
        )

    async def get_audit_events(self, execution_id: str) -> list[AuditEvent]:
        """Read all audit events for an execution (convenience method).

        Args:
            execution_id: The UUID of the execution.

        Returns:
            List of audit events in chronological order.
        """

        def _read() -> list[AuditEvent]:
            path = self._audit_file(execution_id)
            if not path.exists():
                return []
            events: list[AuditEvent] = []
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    events.append(AuditEvent.model_validate_json(line))
            return events

        return await asyncio.to_thread(_read)

    async def save_approval_request(self, request: ApprovalRequest) -> None:
        """Save an approval request as a JSON file and index it."""

        def _write() -> None:
            path = self._approval_file(request.execution_id, request.step_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(request.model_dump_json(indent=2), encoding="utf-8")

            idx_path = self._approval_index_file(request.approval_id)
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            idx_data = json.dumps(
                {"execution_id": request.execution_id, "step_name": request.step_name}
            )
            idx_path.write_text(idx_data, encoding="utf-8")

        await asyncio.to_thread(_write)
        logger.debug("Saved approval request %s", request.approval_id)

    async def get_approval_request(self, approval_id: str) -> ApprovalRequest | None:
        """Load an approval request by ID using the index."""

        def _read() -> ApprovalRequest | None:
            idx_path = self._approval_index_file(approval_id)
            if not idx_path.exists():
                return None
            try:
                data = json.loads(idx_path.read_text(encoding="utf-8"))
                exec_id = data["execution_id"]
                step_name = data["step_name"]
            except Exception:
                return None
            path = self._approval_file(exec_id, step_name)
            if not path.exists():
                return None
            return ApprovalRequest.model_validate_json(path.read_text(encoding="utf-8"))

        return await asyncio.to_thread(_read)

    async def get_approval_request_by_step(
        self, execution_id: str, step_name: str
    ) -> ApprovalRequest | None:
        """Load an approval request by execution ID and step name."""

        def _read() -> ApprovalRequest | None:
            path = self._approval_file(execution_id, step_name)
            if not path.exists():
                return None
            return ApprovalRequest.model_validate_json(path.read_text(encoding="utf-8"))

        return await asyncio.to_thread(_read)

    async def list_approval_requests(self) -> list[ApprovalRequest]:
        """List all approval requests stored on disk."""
        def _read() -> list[ApprovalRequest]:
            results = []
            for path in self._root.rglob("*.json"):
                if path.parent.name == "approvals":
                    try:
                        results.append(ApprovalRequest.model_validate_json(path.read_text(encoding="utf-8")))
                    except Exception:
                        pass
            return results

        return await asyncio.to_thread(_read)
