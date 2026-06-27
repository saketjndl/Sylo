"""Tests for Sylo human approval gates (Brief 04)."""

import asyncio
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sylo
from sylo.core.approval import stop_local_server
from sylo.exceptions import SyloApprovalRejectedError
from sylo.models import ApprovalStatus, ExecutionStatus


@pytest.fixture(autouse=True)
def cleanup_local_server():
    yield
    stop_local_server()


@pytest.mark.asyncio
async def test_approval_flow_approved(init_sylo):
    init_sylo()

    @sylo.requires_approval(
        title="Delete Records",
        description="Deleting {count} records",
        metadata_keys=["count"],
        poll_interval_seconds=0.05,
    )
    async def delete_records(ctx):
        return {"deleted": ctx.metadata.get("count", 0)}

    async def run_pipeline():
        async with sylo.pipeline("test-approval", metadata={"count": 5}) as pipe:
            return await delete_records(pipe.context)

    task = asyncio.create_task(run_pipeline())
    await asyncio.sleep(0.15)

    # Check that approval request was created
    storage = sylo.config.get_config()
    store = sylo.storage.get_storage(storage)
    # Find execution records
    execs = await store.list_executions("test-approval")
    assert len(execs) > 0
    exec_id = execs[0].execution_id

    req = await store.get_approval_request_by_step(exec_id, "delete_records")
    assert req is not None
    assert req.status == ApprovalStatus.PENDING
    assert req.description == "Deleting 5 records"

    # Approve programmatically
    await sylo.approve(req.approval_id, decided_by="admin@co.com")

    result = await task
    assert result == {"deleted": 5}

    updated_req = await store.get_approval_request(req.approval_id)
    assert updated_req.status == ApprovalStatus.APPROVED
    assert updated_req.decided_by == "admin@co.com"


@pytest.mark.asyncio
async def test_approval_flow_rejected(init_sylo):
    init_sylo()

    @sylo.requires_approval(
        title="Dangerous Action",
        description="Will explode",
        poll_interval_seconds=0.05,
    )
    async def explode(ctx):
        return "boom"

    async def run_pipeline():
        async with sylo.pipeline("test-reject") as pipe:
            return await explode(pipe.context)

    task = asyncio.create_task(run_pipeline())
    await asyncio.sleep(0.15)

    store = sylo.storage.get_storage(sylo.config.get_config())
    execs = await store.list_executions("test-reject")
    req = await store.get_approval_request_by_step(execs[0].execution_id, "explode")
    assert req is not None

    await sylo.reject(req.approval_id, decided_by="sec-team")

    with pytest.raises(SyloApprovalRejectedError):
        await task


@pytest.mark.asyncio
async def test_approval_timeout_auto_approve(init_sylo):
    init_sylo()

    @sylo.requires_approval(
        title="Quick Action",
        description="Auto approving",
        timeout_hours=0.00001,  # Expire almost immediately
        on_timeout="auto_approve",
        poll_interval_seconds=0.05,
    )
    async def quick_action(ctx):
        return "done"

    async with sylo.pipeline("test-timeout-approve") as pipe:
        result = await quick_action(pipe.context)
        assert result == "done"


@pytest.mark.asyncio
async def test_approval_timeout_abort(init_sylo):
    init_sylo()

    @sylo.requires_approval(
        title="Abort Action",
        description="Aborting on timeout",
        timeout_hours=0.00001,
        on_timeout="abort",
        poll_interval_seconds=0.05,
    )
    async def abort_action(ctx):
        return "done"

    with pytest.raises(SyloApprovalRejectedError):
        async with sylo.pipeline("test-timeout-abort") as pipe:
            await abort_action(pipe.context)


@pytest.mark.asyncio
async def test_local_dev_server_endpoint(init_sylo):
    init_sylo(storage="local", environment="development")

    @sylo.requires_approval(
        title="HTTP Server Test",
        description="Testing local HTTP server",
        poll_interval_seconds=0.05,
    )
    async def server_action(ctx):
        return "http_ok"

    async def run_pipeline():
        async with sylo.pipeline("test-http-server") as pipe:
            return await server_action(pipe.context)

    task = asyncio.create_task(run_pipeline())
    await asyncio.sleep(0.2)

    store = sylo.storage.get_storage(sylo.config.get_config())
    execs = await store.list_executions("test-http-server")
    req = await store.get_approval_request_by_step(execs[0].execution_id, "server_action")
    assert req is not None

    # Hit the local HTTP endpoint
    url = f"http://localhost:7749/approve/{req.approval_id}"
    response = await asyncio.to_thread(urllib.request.urlopen, url)
    assert response.status == 200
    html = response.read().decode("utf-8")
    assert "Approved" in html

    result = await task
    assert result == "http_ok"
