"""Luro SDK data models.

All data structures use Pydantic v2 for validation, serialization,
and schema generation. These models define the canonical shape of
every record Luro produces.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    """Status of a pipeline execution."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"


class CheckpointStatus(str, Enum):
    """Status of an individual step checkpoint."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ApprovalStatus(str, Enum):
    """Status of a human approval request."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    TIMED_OUT = "TIMED_OUT"


class TokenUsage(BaseModel):
    """Token usage for a single step, following the OpenAI/Anthropic format."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str | None = None
    estimated_cost_usd: float = 0.0


class TokenCost(BaseModel):
    """Aggregate token cost for an entire pipeline execution."""

    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class Checkpoint(BaseModel):
    """A saved snapshot of a step's output.

    Checkpoints enable resume-from-failure: if a pipeline crashes,
    steps with completed checkpoints are skipped on retry, saving
    both time and money.
    """

    checkpoint_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    step_name: str
    step_index: int = 0
    status: CheckpointStatus = CheckpointStatus.COMPLETED
    input_hash: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    duration_ms: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    retry_count: int = 0


class AuditEvent(BaseModel):
    """A single immutable event in the audit log.

    Audit events form a complete, append-only record of everything
    that happened during a pipeline execution. Once written, they
    are never modified or deleted.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    pipeline_name: str
    event_type: str
    step_name: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence_number: int = 0
    data: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0
    token_usage: TokenUsage | None = None
    is_replay: bool = False


class ApprovalRequest(BaseModel):
    """A human approval gate request for an irreversible or dangerous action."""

    approval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    pipeline_name: str
    step_name: str
    title: str
    description: str
    action_class: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    decided_at: datetime | None = None
    decided_by: str | None = None
    decision_note: str | None = None
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    notify_channels: list[str] = Field(default_factory=list)


class ExecutionRecord(BaseModel):
    """Complete record of a single pipeline execution.

    Every time luro.pipeline() runs, one ExecutionRecord is created.
    It contains the execution metadata, all checkpoints, all audit
    events, and the aggregate token cost.
    """

    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_name: str
    pipeline_version: str = "0.0.0"
    status: ExecutionStatus = ExecutionStatus.RUNNING
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error: str | None = None
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    audit_events: list[AuditEvent] = Field(default_factory=list)
    token_cost: TokenCost = Field(default_factory=TokenCost)
    metadata: dict[str, Any] = Field(default_factory=dict)
