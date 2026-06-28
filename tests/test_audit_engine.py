"""Tests for the Audit & Replay Engine (Brief 05)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

import sylo
from sylo.config import reset_config, set_config, SyloConfig
from sylo.core.audit import format_audit_log, get_summary, replay
from sylo.models import (
    AuditEvent,
    ExecutionRecord,
    ExecutionStatus,
    ExecutionSummary,
    StepSummary,
)
from sylo.storage.local_store import LocalStorage


@pytest.fixture(autouse=True)
def reset_sylo():
    """Reset global config before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def init_sylo():
    """Initialize Sylo with local storage."""
    import os
    os.environ.pop("SYLO_PROJECT", None)
    config = SyloConfig(project="test-project")
    set_config(config)
    return config


def _patch_storage(storage):
    """Monkeypatch get_storage in all modules that import it."""
    import sylo.storage as storage_mod
    import sylo.core.audit as audit_mod
    import sylo.core.pipeline as pipeline_mod

    originals = {
        "storage_mod": storage_mod.get_storage,
        "audit_mod": audit_mod.get_storage,
    }

    factory = lambda config: storage
    storage_mod.get_storage = factory
    audit_mod.get_storage = factory

    return originals


def _unpatch_storage(originals):
    """Restore original get_storage functions."""
    import sylo.storage as storage_mod
    import sylo.core.audit as audit_mod

    storage_mod.get_storage = originals["storage_mod"]
    audit_mod.get_storage = originals["audit_mod"]


class TestExecutionSummaryModel:
    """Tests for the ExecutionSummary and StepSummary models."""

    def test_step_summary_creation(self):
        step = StepSummary(
            step_name="fetch-data",
            status="completed",
            duration_ms=1200,
            tokens=500,
            estimated_cost_usd=0.025,
        )
        assert step.step_name == "fetch-data"
        assert step.status == "completed"
        assert step.duration_ms == 1200
        assert step.tokens == 500
        assert not step.was_cached

    def test_execution_summary_creation(self):
        summary = ExecutionSummary(
            execution_id="abc-123",
            pipeline_name="test-pipeline",
            status="COMPLETED",
            duration_seconds=12.5,
            steps_completed=3,
            steps_skipped=1,
            total_tokens=2000,
            estimated_cost_usd=0.05,
            cost_saved_usd=0.01,
        )
        assert summary.pipeline_name == "test-pipeline"
        assert summary.steps_completed == 3
        assert summary.steps_skipped == 1
        assert summary.cost_saved_usd == 0.01

    def test_summary_serialization_roundtrip(self):
        summary = ExecutionSummary(
            execution_id="abc-123",
            pipeline_name="test-pipeline",
            status="COMPLETED",
            timeline=[
                StepSummary(step_name="step-1", status="completed"),
                StepSummary(step_name="step-2", status="skipped", was_cached=True),
            ],
        )
        json_str = summary.model_dump_json()
        restored = ExecutionSummary.model_validate_json(json_str)
        assert len(restored.timeline) == 2
        assert restored.timeline[1].was_cached


class TestGetSummary:
    """Tests for the get_summary() function."""

    @pytest.mark.asyncio
    async def test_summary_from_completed_execution(self, tmp_path, init_sylo):
        storage = LocalStorage(root_dir=tmp_path / "sylo_test")
        originals = _patch_storage(storage)

        try:
            now = datetime.now(timezone.utc)
            record = ExecutionRecord(
                execution_id="test-exec-001",
                pipeline_name="my-pipeline",
                pipeline_version="1.0",
                status=ExecutionStatus.COMPLETED,
                started_at=now - timedelta(seconds=10),
                completed_at=now,
            )
            await storage.save_execution(record)

            events = [
                AuditEvent(execution_id="test-exec-001", pipeline_name="my-pipeline",
                          event_type="PIPELINE_STARTED", sequence_number=1),
                AuditEvent(execution_id="test-exec-001", pipeline_name="my-pipeline",
                          event_type="STEP_STARTED", step_name="fetch", sequence_number=2),
                AuditEvent(execution_id="test-exec-001", pipeline_name="my-pipeline",
                          event_type="STEP_COMPLETED", step_name="fetch", sequence_number=3,
                          duration_ms=500),
                AuditEvent(execution_id="test-exec-001", pipeline_name="my-pipeline",
                          event_type="TOKEN_USAGE_RECORDED", step_name="fetch", sequence_number=4,
                          data={"total_tokens": 100, "estimated_cost_usd": 0.01}),
                AuditEvent(execution_id="test-exec-001", pipeline_name="my-pipeline",
                          event_type="PIPELINE_COMPLETED", sequence_number=5),
            ]
            for event in events:
                await storage.append_audit_event("test-exec-001", event)

            summary = await get_summary("test-exec-001")
            assert summary.pipeline_name == "my-pipeline"
            assert summary.status == "COMPLETED"
            assert summary.steps_completed == 1
            assert summary.total_tokens == 100
            assert summary.estimated_cost_usd == 0.01
            assert summary.duration_seconds == pytest.approx(10.0, abs=1.0)
            assert len(summary.timeline) == 1
            assert summary.timeline[0].step_name == "fetch"
        finally:
            _unpatch_storage(originals)

    @pytest.mark.asyncio
    async def test_summary_not_found_raises(self, tmp_path, init_sylo):
        storage = LocalStorage(root_dir=tmp_path / "sylo_test")
        originals = _patch_storage(storage)

        try:
            with pytest.raises(ValueError, match="not found"):
                await get_summary("nonexistent-id")
        finally:
            _unpatch_storage(originals)

    @pytest.mark.asyncio
    async def test_summary_counts_skipped_steps(self, tmp_path, init_sylo):
        storage = LocalStorage(root_dir=tmp_path / "sylo_test")
        originals = _patch_storage(storage)

        try:
            now = datetime.now(timezone.utc)
            record = ExecutionRecord(
                execution_id="test-exec-002",
                pipeline_name="pipe",
                status=ExecutionStatus.COMPLETED,
                started_at=now - timedelta(seconds=5),
                completed_at=now,
            )
            await storage.save_execution(record)

            events = [
                AuditEvent(execution_id="test-exec-002", pipeline_name="pipe",
                          event_type="STEP_SKIPPED", step_name="cached-step", sequence_number=1,
                          data={"original_cost_usd": 0.05}),
                AuditEvent(execution_id="test-exec-002", pipeline_name="pipe",
                          event_type="STEP_STARTED", step_name="new-step", sequence_number=2),
                AuditEvent(execution_id="test-exec-002", pipeline_name="pipe",
                          event_type="STEP_COMPLETED", step_name="new-step", sequence_number=3,
                          duration_ms=200),
            ]
            for event in events:
                await storage.append_audit_event("test-exec-002", event)

            summary = await get_summary("test-exec-002")
            assert summary.steps_skipped == 1
            assert summary.steps_completed == 1
            assert summary.cost_saved_usd == 0.05
            assert any(s.was_cached for s in summary.timeline)
        finally:
            _unpatch_storage(originals)

    @pytest.mark.asyncio
    async def test_summary_counts_violations_and_approvals(self, tmp_path, init_sylo):
        storage = LocalStorage(root_dir=tmp_path / "sylo_test")
        originals = _patch_storage(storage)

        try:
            now = datetime.now(timezone.utc)
            record = ExecutionRecord(
                execution_id="test-exec-003",
                pipeline_name="pipe",
                status=ExecutionStatus.COMPLETED,
                started_at=now,
                completed_at=now + timedelta(seconds=1),
            )
            await storage.save_execution(record)

            events = [
                AuditEvent(execution_id="test-exec-003", pipeline_name="pipe",
                          event_type="APPROVAL_REQUESTED", step_name="delete", sequence_number=1),
                AuditEvent(execution_id="test-exec-003", pipeline_name="pipe",
                          event_type="PERMISSION_VIOLATION", step_name="bad", sequence_number=2,
                          data={"resource": "secret.data", "action": "read"}),
            ]
            for event in events:
                await storage.append_audit_event("test-exec-003", event)

            summary = await get_summary("test-exec-003")
            assert summary.approval_gates_hit == 1
            assert summary.permission_violations == 1
        finally:
            _unpatch_storage(originals)


class TestReplay:
    """Tests for the replay() function."""

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, tmp_path, init_sylo):
        storage = LocalStorage(root_dir=tmp_path / "sylo_test")
        originals = _patch_storage(storage)

        try:
            now = datetime.now(timezone.utc)
            record = ExecutionRecord(
                execution_id="replay-001",
                pipeline_name="pipe",
                status=ExecutionStatus.COMPLETED,
                started_at=now,
                completed_at=now + timedelta(seconds=5),
            )
            await storage.save_execution(record)

            events = [
                AuditEvent(execution_id="replay-001", pipeline_name="pipe",
                          event_type="STEP_STARTED", step_name="step-a", sequence_number=1),
                AuditEvent(execution_id="replay-001", pipeline_name="pipe",
                          event_type="STEP_STARTED", step_name="step-b", sequence_number=2),
                AuditEvent(execution_id="replay-001", pipeline_name="pipe",
                          event_type="STEP_STARTED", step_name="step-c", sequence_number=3),
            ]
            for event in events:
                await storage.append_audit_event("replay-001", event)

            result = await replay("replay-001", from_step="step-b", dry_run=True)
            assert result["status"] == "dry_run_complete"
            assert result["cached_steps"] == ["step-a"]
            assert result["replay_steps"] == ["step-b", "step-c"]
        finally:
            _unpatch_storage(originals)

    @pytest.mark.asyncio
    async def test_replay_not_found_raises(self, tmp_path, init_sylo):
        storage = LocalStorage(root_dir=tmp_path / "sylo_test")
        originals = _patch_storage(storage)

        try:
            with pytest.raises(ValueError, match="not found"):
                await replay("nonexistent")
        finally:
            _unpatch_storage(originals)

    @pytest.mark.asyncio
    async def test_replay_invalid_step_raises(self, tmp_path, init_sylo):
        storage = LocalStorage(root_dir=tmp_path / "sylo_test")
        originals = _patch_storage(storage)

        try:
            now = datetime.now(timezone.utc)
            record = ExecutionRecord(
                execution_id="replay-002",
                pipeline_name="pipe",
                status=ExecutionStatus.COMPLETED,
                started_at=now,
            )
            await storage.save_execution(record)

            events = [
                AuditEvent(execution_id="replay-002", pipeline_name="pipe",
                          event_type="STEP_STARTED", step_name="only-step", sequence_number=1),
            ]
            for event in events:
                await storage.append_audit_event("replay-002", event)

            with pytest.raises(ValueError, match="not found"):
                await replay("replay-002", from_step="nonexistent-step")
        finally:
            _unpatch_storage(originals)


class TestFormatAuditLog:
    """Tests for the format_audit_log() pretty-printer."""

    def test_empty_events(self):
        result = format_audit_log([])
        assert "No audit events found" in result

    def test_basic_formatting(self):
        now = datetime.now(timezone.utc)
        events = [
            AuditEvent(
                execution_id="fmt-001", pipeline_name="pipe",
                event_type="PIPELINE_STARTED", timestamp=now, sequence_number=1,
            ),
            AuditEvent(
                execution_id="fmt-001", pipeline_name="pipe",
                event_type="STEP_STARTED", step_name="fetch",
                timestamp=now + timedelta(milliseconds=43), sequence_number=2,
            ),
            AuditEvent(
                execution_id="fmt-001", pipeline_name="pipe",
                event_type="STEP_COMPLETED", step_name="fetch",
                timestamp=now + timedelta(seconds=2), sequence_number=3,
                duration_ms=1957,
            ),
            AuditEvent(
                execution_id="fmt-001", pipeline_name="pipe",
                event_type="PIPELINE_COMPLETED",
                timestamp=now + timedelta(seconds=3), sequence_number=4,
            ),
        ]

        result = format_audit_log(events, pipeline_name="pipe", execution_id="fmt-001-xxx")
        assert "Sylo Audit Log" in result
        assert "pipe" in result
        assert "fmt-001-" in result
        assert "PIPELINE_STARTED" in result
        assert "STEP_COMPLETED" in result
        assert "fetch" in result

    def test_permission_violation_formatting(self):
        now = datetime.now(timezone.utc)
        events = [
            AuditEvent(
                execution_id="fmt-002", pipeline_name="pipe",
                event_type="PERMISSION_VIOLATION", step_name="bad-step",
                timestamp=now, sequence_number=1,
                data={"resource": "secret.data", "action": "read"},
            ),
        ]
        result = format_audit_log(events)
        assert "BLOCKED" in result
        assert "secret.data" in result

    def test_approval_formatting(self):
        now = datetime.now(timezone.utc)
        events = [
            AuditEvent(
                execution_id="fmt-003", pipeline_name="pipe",
                event_type="APPROVAL_REQUESTED", step_name="delete",
                timestamp=now, sequence_number=1,
            ),
            AuditEvent(
                execution_id="fmt-003", pipeline_name="pipe",
                event_type="APPROVAL_DECISION", step_name="delete",
                timestamp=now + timedelta(minutes=5), sequence_number=2,
                data={"decision": "APPROVED", "decided_by": "admin@co.com"},
            ),
        ]
        result = format_audit_log(events)
        assert "awaiting" in result
        assert "APPROVED" in result
        assert "admin@co.com" in result

    def test_footer_summary(self):
        now = datetime.now(timezone.utc)
        events = [
            AuditEvent(
                execution_id="fmt-004", pipeline_name="pipe",
                event_type="PIPELINE_STARTED", timestamp=now, sequence_number=1,
            ),
            AuditEvent(
                execution_id="fmt-004", pipeline_name="pipe",
                event_type="TOKEN_USAGE_RECORDED", step_name="step1",
                timestamp=now + timedelta(seconds=1), sequence_number=2,
                data={"total_tokens": 500, "estimated_cost_usd": 0.025},
            ),
            AuditEvent(
                execution_id="fmt-004", pipeline_name="pipe",
                event_type="PIPELINE_COMPLETED",
                timestamp=now + timedelta(seconds=10), sequence_number=3,
            ),
        ]
        result = format_audit_log(events)
        assert "Total:" in result
        assert "$0.025" in result
        assert "500" in result


class TestRecordTokenUsage:
    """Tests for ctx.record_token_usage() (Phase 1 fix)."""

    @pytest.mark.asyncio
    async def test_record_token_usage_basic(self, tmp_path):
        import os
        os.environ.pop("SYLO_PROJECT", None)
        config = SyloConfig(project="test", storage="local")
        set_config(config)

        storage = LocalStorage(root_dir=tmp_path / "sylo_test")
        originals = _patch_storage(storage)

        try:
            @sylo.step("token-test")
            async def my_step(ctx: sylo.Context) -> dict:
                ctx.record_token_usage(prompt_tokens=100, completion_tokens=50, model="gpt-4o")
                return {"result": "ok"}

            async with sylo.pipeline("test-pipe") as pipe:
                result = await my_step(pipe.context)

            assert result == {"result": "ok"}
            assert pipe.record.token_cost.total_tokens == 150
            assert pipe.record.token_cost.estimated_cost_usd > 0
        finally:
            _unpatch_storage(originals)

    @pytest.mark.asyncio
    async def test_record_token_usage_custom_total(self, tmp_path):
        import os
        os.environ.pop("SYLO_PROJECT", None)
        config = SyloConfig(project="test", storage="local")
        set_config(config)

        storage = LocalStorage(root_dir=tmp_path / "sylo_test")
        originals = _patch_storage(storage)

        try:
            @sylo.step("token-test-2")
            async def my_step(ctx: sylo.Context) -> dict:
                ctx.record_token_usage(prompt_tokens=100, completion_tokens=50, total_tokens=200)
                return {"done": True}

            async with sylo.pipeline("test-pipe") as pipe:
                await my_step(pipe.context)

            assert pipe.record.token_cost.total_tokens == 200
        finally:
            _unpatch_storage(originals)
