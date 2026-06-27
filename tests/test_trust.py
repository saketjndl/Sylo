"""Tests for the Trust Broker (Brief 03).

Tests cover:
- Undeclared resource access raises SyloPermissionError
- Declared resources are accessible
- Unused permissions generate least privilege warnings in dev mode
- Trust summary correctly computed and emitted to audit log
- Global wildcard "*" allows all accesses
- Prefix wildcards "service.*" allow accesses
- Production warning when step has no trust declarations
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

import sylo
from sylo.exceptions import SyloPermissionError
from sylo.storage.local_store import LocalStorage


@pytest.fixture
def setup_sylo(tmp_storage_dir: Path):
    """Initialize sylo with local storage in a temp directory."""
    sylo.init(project="test-project", environment="development", storage="local")

    def _patched_get_storage(config):
        return LocalStorage(root_dir=tmp_storage_dir)

    with patch("sylo.core.pipeline.get_storage", _patched_get_storage):
        yield tmp_storage_dir


class TestTrustBrokerEnforcement:
    """Tests for permission sandboxing and runtime enforcement."""

    @pytest.mark.asyncio
    async def test_access_undeclared_resource_raises(self, setup_sylo: Path):
        """Accessing an undeclared resource should raise SyloPermissionError."""

        @sylo.step("send-email")
        @sylo.trust(
            can_read=["gmail.labels"],
            can_write=["gmail.drafts"],
        )
        async def send_email_step(ctx: sylo.Context) -> str:
            # Attempt to read gmail.messages (which is undeclared)
            await ctx.access("gmail.messages", action="read", handler="dummy_data")
            return "ok"

        async with sylo.pipeline("test-pipeline") as pipe:
            with pytest.raises(SyloPermissionError, match="attempted to read undeclared resource"):
                await send_email_step(pipe.context)

    @pytest.mark.asyncio
    async def test_access_declared_resource_succeeds(self, setup_sylo: Path):
        """Accessing a declared resource should succeed and call the handler."""

        @sylo.step("send-email")
        @sylo.trust(
            can_read=["gmail.messages"],
            can_write=["gmail.drafts"],
        )
        async def send_email_step(ctx: sylo.Context) -> str:
            val = await ctx.access(
                "gmail.messages",
                action="read",
                handler=lambda: "messages_list",
            )
            return val

        async with sylo.pipeline("test-pipeline") as pipe:
            res = await send_email_step(pipe.context)

        assert res == "messages_list"

    @pytest.mark.asyncio
    async def test_global_wildcard_allows_all(self, setup_sylo: Path, caplog):
        """Global wildcard '*' allows any resource but logs a warning during definition."""
        with caplog.at_level(logging.WARNING, logger="sylo"):
            @sylo.trust(can_read=["*"])
            async def wildcard_dummy(ctx: sylo.Context):
                pass

        # Warning should be logged when the decorator is defined/executed
        assert any("Wildcard '*' permission declared" in msg for msg in caplog.messages)

        @sylo.step("wildcard-step")
        @sylo.trust(can_read=["*"])
        async def wildcard_step(ctx: sylo.Context) -> dict:
            r1 = await ctx.access("gmail.messages", action="read", handler="msg")
            r2 = await ctx.access("notion.pages", action="read", handler="notion")
            return {"r1": r1, "r2": r2}

        async with sylo.pipeline("test-pipeline") as pipe:
            res = await wildcard_step(pipe.context)

        assert res == {"r1": "msg", "r2": "notion"}

    @pytest.mark.asyncio
    async def test_prefix_wildcard_allows_matching_resources(self, setup_sylo: Path):
        """Prefix wildcards (e.g. 'gmail.*') should allow access to matching resources."""

        @sylo.step("gmail-step")
        @sylo.trust(can_read=["gmail.*"])
        async def gmail_step(ctx: sylo.Context) -> str:
            # Matches prefix 'gmail.*'
            await ctx.access("gmail.messages", action="read", handler="ok")
            # Does not match prefix 'gmail.*'
            await ctx.access("slack.channels", action="read", handler="fail")
            return "done"

        async with sylo.pipeline("test-pipeline") as pipe:
            with pytest.raises(SyloPermissionError, match="attempted to read undeclared resource"):
                await gmail_step(pipe.context)

    @pytest.mark.asyncio
    async def test_least_privilege_warnings(self, setup_sylo: Path, caplog):
        """Unused declared permissions should generate warnings in development mode."""

        @sylo.step("least-privilege")
        @sylo.trust(
            can_read=["gmail.messages", "gmail.labels"],
            can_write=["gmail.drafts"],
        )
        async def least_privilege_step(ctx: sylo.Context) -> str:
            # We access gmail.messages, but NEVER gmail.labels or gmail.drafts
            await ctx.access("gmail.messages", action="read", handler="ok")
            return "done"

        with caplog.at_level(logging.WARNING, logger="sylo"):
            async with sylo.pipeline("test-pipeline") as pipe:
                await least_privilege_step(pipe.context)

        warnings = [msg for msg in caplog.messages if "Consider removing unused permissions" in msg]
        assert len(warnings) == 2
        assert any('declared "gmail.labels"' in w for w in warnings)
        assert any('declared "gmail.drafts"' in w for w in warnings)

    @pytest.mark.asyncio
    async def test_trust_summary_emitted(self, setup_sylo: Path):
        """A TRUST_SUMMARY audit event should be correctly calculated and saved."""
        storage = LocalStorage(root_dir=setup_sylo)

        @sylo.step("summary-step")
        @sylo.trust(
            can_read=["gmail.messages", "gmail.labels"],
            can_write=["gmail.drafts"],
        )
        async def summary_step(ctx: sylo.Context) -> str:
            # 1 success read
            await ctx.access("gmail.messages", action="read", handler="ok")
            # 1 violation attempt (which we catch)
            try:
                await ctx.access("slack.channels", action="read", handler="oops")
            except SyloPermissionError:
                pass
            return "done"

        async with sylo.pipeline("test-pipeline") as pipe:
            await summary_step(pipe.context)
            exec_id = pipe.execution_id

        events = await storage.get_audit_events(exec_id)
        summary_events = [e for e in events if e.event_type == "TRUST_SUMMARY"]
        assert len(summary_events) == 1

        summary = summary_events[0]
        assert summary.step_name == "summary-step"
        assert sorted(summary.data["declared_permissions"]) == sorted([
            "read:gmail.messages",
            "read:gmail.labels",
            "write:gmail.drafts",
        ])
        assert summary.data["permissions_used"] == ["read:gmail.messages"]
        assert sorted(summary.data["permissions_unused"]) == sorted([
            "read:gmail.labels",
            "write:gmail.drafts",
        ])
        assert summary.data["violations_attempted"] == 1

    @pytest.mark.asyncio
    async def test_production_untrusted_step_warning(self, setup_sylo: Path, caplog):
        """Steps without @sylo.trust should log a warning when run in production mode."""
        # Initialize Sylo in production mode
        sylo.init(project="test-project", environment="production", storage="local")

        @sylo.step("untrusted-step")
        async def untrusted_step(ctx: sylo.Context) -> str:
            return "unrestricted"

        with caplog.at_level(logging.WARNING, logger="sylo"):
            async with sylo.pipeline("test-pipeline") as pipe:
                await untrusted_step(pipe.context)

        assert any("Step \"untrusted-step\" has no trust declaration" in msg for msg in caplog.messages)
