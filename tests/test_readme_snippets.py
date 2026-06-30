"""Tests verifying every code snippet and statement in README.md."""

import asyncio
import os
import time
import urllib.request
import pytest
import sylo
from sylo.exceptions import SyloPermissionError


@pytest.fixture(autouse=True)
def setup_local_sylo(tmp_path):
    """Initialize Sylo with local file storage in a temporary directory before each test."""
    os.environ["SYLO_STORAGE_DIR"] = str(tmp_path / ".sylo")
    sylo.init(project="customer-operations", environment="development", storage="local")
    yield
    os.environ.pop("SYLO_STORAGE_DIR", None)
    sylo.reset_config()


def _auto_approve_http_thread(delay: float = 0.5):
    """Background thread that auto-approves via HTTP server on localhost:7749."""
    from pathlib import Path
    import json
    time.sleep(delay)
    # Loop to find pending approval
    for _ in range(25):
        storage_dir = os.environ.get("SYLO_STORAGE_DIR")
        if storage_dir and os.path.exists(storage_dir):
            for af in Path(storage_dir).rglob("*.json"):
                if af.parent.name == "approvals":
                    try:
                        with open(af, "r", encoding="utf-8") as fp:
                            data = json.load(fp)
                        if data.get("status") == "PENDING":
                            approval_id = data["approval_id"]
                            url = f"http://localhost:7749/approve/{approval_id}"
                            urllib.request.urlopen(url, timeout=5)
                            return
                    except Exception:
                        pass
        time.sleep(0.2)


@pytest.mark.asyncio
async def test_readme_quickstart_five_minute_integration():
    """Verify the 5-minute Quickstart code snippet in README.md (lines 116-182)."""
    sylo.init(project="customer-operations", environment="development")

    @sylo.step("fetch-customer-data", max_retries=3, retry_delay=1.0)
    @sylo.trust(can_read=["crm.customers"])
    async def fetch_customer(ctx: sylo.Context, customer_id: str) -> dict:
        """Fetch customer details through permission-checked context access."""
        async def _api_call():
            return {"id": customer_id, "name": "Acme Corp", "status": "inactive"}

        return await ctx.access("crm.customers", action="read", handler=_api_call)

    @sylo.step("analyze-churn-risk")
    async def analyze_churn(ctx: sylo.Context) -> dict:
        """Analyze risk using previous step outputs."""
        customer = ctx.get_output("fetch-customer-data")
        ctx.record_token_usage(prompt_tokens=450, completion_tokens=120, model="gpt-4o")
        return {"customer_id": customer["id"], "risk_score": 0.89, "recommendation": "terminate"}

    @sylo.step("delete-account")
    @sylo.requires_approval(
        title="Confirm Account Deletion",
        description="About to permanently delete account for {customer_id} (Risk Score: {risk_score})",
        action_class="destructive",
        timeout_hours=24,
        on_timeout="abort",
        notify=["slack", "email"],
        metadata_keys=["customer_id", "risk_score"],
        poll_interval_seconds=0.2,
    )
    async def delete_account(ctx: sylo.Context) -> dict:
        """Dangerous action guarded by a human approval gate."""
        customer_id = ctx.metadata["customer_id"]
        return {"status": "deleted", "customer_id": customer_id}

    # Launch background auto-approver thread
    import threading
    t = threading.Thread(target=_auto_approve_http_thread, args=(0.5,), daemon=True)
    t.start()

    async with sylo.pipeline("churn-remediation", metadata={"customer_id": "cust_8842"}) as pipe:
        data = await fetch_customer(pipe.context, "cust_8842")
        assert data == {"id": "cust_8842", "name": "Acme Corp", "status": "inactive"}

        analysis = await analyze_churn(pipe.context)
        assert analysis == {"customer_id": "cust_8842", "risk_score": 0.89, "recommendation": "terminate"}

        pipe.context.metadata.update(analysis)

        result = await delete_account(pipe.context)
        assert result == {"status": "deleted", "customer_id": "cust_8842"}


@pytest.mark.asyncio
async def test_readme_smart_checkpointing_snippet():
    """Verify Smart Checkpointing & Cost Tracking snippet (lines 211-220)."""
    # Check MODEL_PRICES built-in table
    assert "gpt-4o" in sylo.MODEL_PRICES
    assert sylo.MODEL_PRICES["gpt-4o"]["input"] == 0.0025  # $2.50 per 1M tokens
    assert sylo.MODEL_PRICES["gpt-4o"]["output"] == 0.01   # $10.00 per 1M tokens
    assert "claude-3-5-sonnet" in sylo.MODEL_PRICES

    call_counter = 0

    @sylo.step("fetch-data")
    async def fetch_data(ctx: sylo.Context) -> dict:
        return {"records": [1, 2, 3]}

    @sylo.step("generate-report", max_retries=5, retry_delay=2.0)
    async def generate_report(ctx: sylo.Context) -> dict:
        nonlocal call_counter
        call_counter += 1
        raw_data = ctx.get_output("fetch-data")
        assert raw_data == {"records": [1, 2, 3]}
        ctx.record_token_usage(prompt_tokens=1200, completion_tokens=350, model="claude-3-5-sonnet")
        return {"report_url": "https://..."}

    first_exec_id = ""
    async with sylo.pipeline("report-pipeline") as pipe:
        first_exec_id = pipe.execution_id
        await fetch_data(pipe.context)
        res1 = await generate_report(pipe.context)
        assert res1 == {"report_url": "https://..."}
        assert call_counter == 1

    # Run again with resume_from: should skip generate_report from checkpoint
    async with sylo.pipeline("report-pipeline", resume_from=first_exec_id) as pipe:
        await fetch_data(pipe.context)
        res2 = await generate_report(pipe.context)
        assert res2 == {"report_url": "https://..."}
        assert call_counter == 1  # Was not re-run!


@pytest.mark.asyncio
async def test_readme_zero_trust_security_broker_snippet():
    """Verify Zero-Trust Security Broker snippet (lines 233-247)."""
    post_message_called = False

    async def post_message():
        nonlocal post_message_called
        post_message_called = True
        return "slack_sent"

    async def delete_bucket():
        return "bucket_deleted"

    @sylo.step("send-slack-notification")
    @sylo.trust(
        can_read=["slack.channels", "users.profile"],
        can_write=["slack.messages"],
        can_execute=[],
        can_delete=[]
    )
    async def notify_team(ctx: sylo.Context):
        # Succeeds: matches declared can_write pattern
        res = await ctx.access("slack.messages", action="write", handler=post_message)
        assert res == "slack_sent"

        # Raises SyloPermissionError
        await ctx.access("aws.s3.buckets", action="delete", handler=delete_bucket)

    async with sylo.pipeline("security-pipeline") as pipe:
        with pytest.raises(SyloPermissionError) as exc_info:
            await notify_team(pipe.context)
        assert "attempted to delete undeclared resource 'aws.s3.buckets'" in str(exc_info.value)
    assert post_message_called


@pytest.mark.asyncio
async def test_readme_human_approval_gates_snippet():
    """Verify Human Approval Gates & programmatic sylo.approve() snippet (lines 254-266)."""
    @sylo.step("wire-transfer")
    @sylo.requires_approval(
        title="Wire Transfer Request",
        description="Transferring ${amount} to account {recipient}",
        action_class="financial",
        timeout_hours=4.0,
        on_timeout="escalate",
        notify=["slack", "webhook"],
        metadata_keys=["amount", "recipient"],
        poll_interval_seconds=0.1,
    )
    async def wire_transfer(ctx: sylo.Context) -> dict:
        return {"status": "transferred", "amount": ctx.metadata["amount"]}

    async def programmatic_approver():
        for _ in range(25):
            await asyncio.sleep(0.2)
            from sylo.storage import get_storage
            from sylo.config import get_config
            storage = get_storage(get_config())
            reqs = await storage.list_approval_requests()
            for r in reqs:
                if r.status == sylo.ApprovalStatus.PENDING:
                    await sylo.approve(r.approval_id, decided_by="supervisor")
                    return

    asyncio.create_task(programmatic_approver())

    async with sylo.pipeline("finance-pipeline", metadata={"amount": 5000, "recipient": "Acme Bank"}) as pipe:
        res = await wire_transfer(pipe.context)
        assert res == {"status": "transferred", "amount": 5000}


def test_readme_programmatic_configuration():
    """Verify programmatic init and environment variables (lines 293-318)."""
    sylo.init(
        project="autonomous-researcher",
        api_key="sylo_live_xxxxx",
        environment="production",
        storage="local",
        notifications={
            "slack": {"webhook_url": "https://hooks.slack.com/services/..."},
            "email": {"provider": "resend", "api_key": "re_123456", "from": "saketjndl2005@gmail.com"}
        }
    )
    cfg = sylo.config.get_config()
    assert cfg.project == "autonomous-researcher"
    assert cfg.api_key == "sylo_live_xxxxx"
    assert cfg.environment == "production"
    assert cfg.notifications["slack"]["webhook_url"] == "https://hooks.slack.com/services/..."


def test_readme_framework_integrations_imports():
    """Verify that adapters documented in Framework Integrations table exist and import clean."""
    from sylo.integrations.langgraph import SyloGraph
    from sylo.integrations.openai_agents import wrap_agent, WrappedAgent
    from sylo.integrations.crewai import SyloCrew
    assert SyloGraph
    assert wrap_agent
    assert WrappedAgent
    assert SyloCrew
