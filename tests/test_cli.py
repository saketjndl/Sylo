"""Tests for the Sylo CLI (Brief 05)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

from sylo.cli import cli
from sylo.config import reset_config, set_config, SyloConfig
from sylo.models import AuditEvent, ExecutionRecord, ExecutionStatus
from sylo.storage.local_store import LocalStorage


@pytest.fixture(autouse=True)
def reset_sylo():
    """Reset global config before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


def _patch_storage(storage):
    """Monkeypatch get_storage in all modules that import it."""
    import sylo.storage as storage_mod
    import sylo.core.audit as audit_mod

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


@pytest.fixture
def setup_storage(tmp_path):
    """Set up local storage with test data."""
    storage = LocalStorage(root_dir=tmp_path / "sylo_test")
    originals = _patch_storage(storage)

    now = datetime.now(timezone.utc)

    records = [
        ExecutionRecord(
            execution_id="exec-aaa-111-222-333",
            pipeline_name="customer-pipeline",
            pipeline_version="1.0",
            status=ExecutionStatus.COMPLETED,
            started_at=now - timedelta(minutes=10),
            completed_at=now - timedelta(minutes=9),
        ),
        ExecutionRecord(
            execution_id="exec-bbb-444-555-666",
            pipeline_name="customer-pipeline",
            pipeline_version="1.0",
            status=ExecutionStatus.FAILED,
            started_at=now - timedelta(minutes=5),
            completed_at=now - timedelta(minutes=4),
            error="ConnectionError: API timeout",
        ),
    ]

    async def _setup():
        for record in records:
            await storage.save_execution(record)

        events = [
            AuditEvent(execution_id="exec-aaa-111-222-333", pipeline_name="customer-pipeline",
                      event_type="PIPELINE_STARTED", sequence_number=1, timestamp=now - timedelta(minutes=10)),
            AuditEvent(execution_id="exec-aaa-111-222-333", pipeline_name="customer-pipeline",
                      event_type="STEP_STARTED", step_name="fetch", sequence_number=2,
                      timestamp=now - timedelta(minutes=10, seconds=-1)),
            AuditEvent(execution_id="exec-aaa-111-222-333", pipeline_name="customer-pipeline",
                      event_type="STEP_COMPLETED", step_name="fetch", sequence_number=3,
                      duration_ms=1500, timestamp=now - timedelta(minutes=9, seconds=30)),
            AuditEvent(execution_id="exec-aaa-111-222-333", pipeline_name="customer-pipeline",
                      event_type="PIPELINE_COMPLETED", sequence_number=4,
                      timestamp=now - timedelta(minutes=9)),
        ]
        for event in events:
            await storage.append_audit_event("exec-aaa-111-222-333", event)

    asyncio.run(_setup())

    yield storage

    _unpatch_storage(originals)


class TestCLIExecList:
    """Tests for `sylo executions list`."""

    def test_list_all_executions(self, runner, setup_storage):
        result = runner.invoke(cli, ["executions", "list"])
        assert result.exit_code == 0
        assert "exec-aaa" in result.output
        assert "exec-bbb" in result.output
        assert "customer-pipeline" in result.output

    def test_list_by_pipeline(self, runner, setup_storage):
        result = runner.invoke(cli, ["executions", "list", "--pipeline", "customer-pipeline"])
        assert result.exit_code == 0
        assert "customer-pipeline" in result.output

    def test_list_empty(self, runner, setup_storage):
        result = runner.invoke(cli, ["executions", "list", "--pipeline", "nonexistent"])
        assert result.exit_code == 0
        assert "No executions found" in result.output


class TestCLIExecInspect:
    """Tests for `sylo executions inspect`."""

    def test_inspect_execution(self, runner, setup_storage):
        result = runner.invoke(cli, ["executions", "inspect", "exec-aaa-111-222-333"])
        assert result.exit_code == 0
        assert "customer-pipeline" in result.output
        assert "COMPLETED" in result.output
        assert "fetch" in result.output

    def test_inspect_not_found(self, runner, setup_storage):
        result = runner.invoke(cli, ["executions", "inspect", "nonexistent-id"])
        assert result.exit_code != 0


class TestCLIAudit:
    """Tests for `sylo audit`."""

    def test_audit_log(self, runner, setup_storage):
        result = runner.invoke(cli, ["audit", "exec-aaa-111-222-333"])
        assert result.exit_code == 0
        assert "Sylo Audit Log" in result.output
        assert "PIPELINE_STARTED" in result.output
        assert "STEP_COMPLETED" in result.output
        assert "fetch" in result.output

    def test_audit_not_found(self, runner, setup_storage):
        result = runner.invoke(cli, ["audit", "nonexistent-id"])
        assert result.exit_code != 0


class TestCLIReplay:
    """Tests for `sylo executions replay`."""

    def test_replay_dry_run(self, runner, setup_storage):
        result = runner.invoke(cli, ["executions", "replay", "exec-aaa-111-222-333", "--dry-run"])
        assert result.exit_code == 0
        assert "Replay" in result.output

    def test_replay_not_found(self, runner, setup_storage):
        result = runner.invoke(cli, ["executions", "replay", "nonexistent-id", "--dry-run"])
        assert result.exit_code != 0
