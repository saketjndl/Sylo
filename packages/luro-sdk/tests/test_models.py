"""Tests for Luro SDK Pydantic data models."""

from __future__ import annotations

from datetime import datetime, timezone

from luro.models import (
    AuditEvent,
    Checkpoint,
    CheckpointStatus,
    ExecutionRecord,
    ExecutionStatus,
    TokenCost,
    TokenUsage,
)


class TestExecutionStatus:
    """Tests for the ExecutionStatus enum."""

    def test_status_values(self):
        assert ExecutionStatus.RUNNING == "RUNNING"
        assert ExecutionStatus.COMPLETED == "COMPLETED"
        assert ExecutionStatus.FAILED == "FAILED"
        assert ExecutionStatus.AWAITING_APPROVAL == "AWAITING_APPROVAL"


class TestTokenCost:
    """Tests for the TokenCost model."""

    def test_defaults(self):
        cost = TokenCost()
        assert cost.total_tokens == 0
        assert cost.estimated_cost_usd == 0.0

    def test_with_values(self):
        cost = TokenCost(total_tokens=1500, estimated_cost_usd=0.045)
        assert cost.total_tokens == 1500
        assert cost.estimated_cost_usd == 0.045


class TestTokenUsage:
    """Tests for the TokenUsage model."""

    def test_defaults(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.model is None

    def test_with_values(self):
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            model="gpt-4o",
            estimated_cost_usd=0.003,
        )
        assert usage.prompt_tokens == 100
        assert usage.total_tokens == 300
        assert usage.model == "gpt-4o"


class TestExecutionRecord:
    """Tests for the ExecutionRecord model."""

    def test_minimal_record(self):
        record = ExecutionRecord(pipeline_name="test-pipeline")
        assert record.pipeline_name == "test-pipeline"
        assert record.pipeline_version == "0.0.0"
        assert record.status == ExecutionStatus.RUNNING
        assert record.execution_id  # should be auto-generated UUID
        assert record.started_at  # should be auto-generated timestamp
        assert record.completed_at is None
        assert record.error is None
        assert record.checkpoints == []
        assert record.audit_events == []
        assert record.token_cost.total_tokens == 0

    def test_full_record(self):
        now = datetime.now(timezone.utc)
        record = ExecutionRecord(
            execution_id="test-uuid",
            pipeline_name="test-pipeline",
            pipeline_version="2.0",
            status=ExecutionStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            metadata={"key": "value"},
        )
        assert record.execution_id == "test-uuid"
        assert record.pipeline_version == "2.0"
        assert record.status == ExecutionStatus.COMPLETED
        assert record.metadata["key"] == "value"

    def test_serialization_roundtrip(self):
        """Record should survive JSON serialization and deserialization."""
        record = ExecutionRecord(
            pipeline_name="test-pipeline",
            pipeline_version="1.0",
            metadata={"count": 42},
        )
        json_str = record.model_dump_json()
        restored = ExecutionRecord.model_validate_json(json_str)
        assert restored.pipeline_name == record.pipeline_name
        assert restored.execution_id == record.execution_id
        assert restored.metadata["count"] == 42


class TestCheckpoint:
    """Tests for the Checkpoint model."""

    def test_minimal_checkpoint(self):
        cp = Checkpoint(execution_id="exec-1", step_name="step-1")
        assert cp.execution_id == "exec-1"
        assert cp.step_name == "step-1"
        assert cp.status == CheckpointStatus.COMPLETED
        assert cp.step_index == 0
        assert cp.output == {}
        assert cp.retry_count == 0
        assert cp.checkpoint_id  # auto-generated

    def test_serialization_roundtrip(self):
        cp = Checkpoint(
            execution_id="exec-1",
            step_name="step-1",
            output={"result": "hello"},
            duration_ms=1500,
        )
        json_str = cp.model_dump_json()
        restored = Checkpoint.model_validate_json(json_str)
        assert restored.output["result"] == "hello"
        assert restored.duration_ms == 1500


class TestAuditEvent:
    """Tests for the AuditEvent model."""

    def test_minimal_event(self):
        event = AuditEvent(
            execution_id="exec-1",
            pipeline_name="test-pipeline",
            event_type="PIPELINE_STARTED",
        )
        assert event.execution_id == "exec-1"
        assert event.event_type == "PIPELINE_STARTED"
        assert event.step_name is None
        assert event.sequence_number == 0
        assert event.is_replay is False
        assert event.event_id  # auto-generated

    def test_step_event(self):
        event = AuditEvent(
            execution_id="exec-1",
            pipeline_name="test-pipeline",
            event_type="STEP_COMPLETED",
            step_name="fetch-emails",
            duration_ms=2500,
            data={"tokens": 847},
        )
        assert event.step_name == "fetch-emails"
        assert event.duration_ms == 2500
        assert event.data["tokens"] == 847

    def test_serialization_roundtrip(self):
        event = AuditEvent(
            execution_id="exec-1",
            pipeline_name="test",
            event_type="STEP_COMPLETED",
            sequence_number=5,
            data={"nested": {"value": True}},
        )
        json_str = event.model_dump_json()
        restored = AuditEvent.model_validate_json(json_str)
        assert restored.sequence_number == 5
        assert restored.data["nested"]["value"] is True
