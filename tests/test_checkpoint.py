"""Tests for the Checkpoint Engine (Brief 02).

Tests cover:
- Completed step not re-executed on retry (checkpoint hit)
- Token costs calculated correctly
- Retry with exponential backoff
- Context.previous_outputs contains correct data
- Resumption from mid-pipeline failure
- Cost savings report output
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import sylo
from sylo.core.checkpoint import step, StepResult
from sylo.core.context import Context
from sylo.core.costs import estimate_cost, extract_token_usage, COST_PER_1K_TOKENS
from sylo.models import (
    Checkpoint,
    CheckpointStatus,
    ExecutionRecord,
    ExecutionStatus,
    TokenUsage,
)
from sylo.storage.local_store import LocalStorage


@pytest.fixture
def setup_sylo(tmp_storage_dir: Path):
    """Initialize sylo with local storage in a temp directory."""
    sylo.init(project="test-project", environment="development", storage="local")

    def _patched_get_storage(config):
        return LocalStorage(root_dir=tmp_storage_dir)

    with patch("sylo.core.pipeline.get_storage", _patched_get_storage):
        yield tmp_storage_dir


class TestStepDecorator:
    """Tests for the @sylo.step decorator."""

    @pytest.mark.asyncio
    async def test_step_executes_and_returns_result(self, setup_sylo: Path):
        """A step should execute and return its result."""

        @sylo.step("my-step")
        async def my_step(ctx: sylo.Context) -> dict:
            return {"data": "hello"}

        async with sylo.pipeline("test-pipeline") as pipe:
            result = await my_step(pipe.context)

        assert result == {"data": "hello"}

    @pytest.mark.asyncio
    async def test_step_saves_checkpoint(self, setup_sylo: Path):
        """A step should save a checkpoint after execution."""
        storage = LocalStorage(root_dir=setup_sylo)

        @sylo.step("save-test")
        async def save_step(ctx: sylo.Context) -> dict:
            return {"saved": True}

        async with sylo.pipeline("test-pipeline") as pipe:
            await save_step(pipe.context)
            exec_id = pipe.execution_id

        cp = await storage.get_checkpoint(exec_id, "save-test")
        assert cp is not None
        assert cp.status == CheckpointStatus.COMPLETED
        assert cp.output["saved"] is True

    @pytest.mark.asyncio
    async def test_step_records_duration(self, setup_sylo: Path):
        """Steps should record their execution duration."""
        storage = LocalStorage(root_dir=setup_sylo)

        @sylo.step("timed-step")
        async def timed_step(ctx: sylo.Context) -> dict:
            await asyncio.sleep(0.05)
            return {"done": True}

        async with sylo.pipeline("test-pipeline") as pipe:
            await timed_step(pipe.context)
            exec_id = pipe.execution_id

        cp = await storage.get_checkpoint(exec_id, "timed-step")
        assert cp is not None
        assert cp.duration_ms >= 40  # At least ~50ms minus some tolerance


class TestCheckpointResumption:
    """Tests for checkpoint-based resumption (skipping completed steps)."""

    @pytest.mark.asyncio
    async def test_completed_step_not_reexecuted(self, setup_sylo: Path):
        """A step with an existing COMPLETED checkpoint should be skipped."""
        storage = LocalStorage(root_dir=setup_sylo)
        call_count = 0

        @sylo.step("resumable-step")
        async def resumable_step(ctx: sylo.Context) -> dict:
            nonlocal call_count
            call_count += 1
            return {"result": "fresh"}

        # Run 1: execute the step normally
        async with sylo.pipeline("test-pipeline") as pipe:
            result1 = await resumable_step(pipe.context)
            exec_id = pipe.execution_id

        assert call_count == 1
        assert result1 == {"result": "fresh"}

        # Manually copy the checkpoint to the new execution by pre-saving
        # Simulate what auto-resumption does
        cp = await storage.get_checkpoint(exec_id, "resumable-step")
        assert cp is not None

        # Run 2: create a new execution but pre-load the checkpoint
        async with sylo.pipeline("test-pipeline") as pipe2:
            # Pre-save the checkpoint under the new execution_id
            cp.execution_id = pipe2.execution_id
            cp.checkpoint_id = "reused-cp"
            await storage.save_checkpoint(cp)

            result2 = await resumable_step(pipe2.context)

        # Step should NOT have been called again
        assert call_count == 1
        # Should return the cached output
        assert result2 == {"result": "fresh"}

    @pytest.mark.asyncio
    async def test_previous_outputs_populated(self, setup_sylo: Path):
        """Context.previous_outputs should contain outputs from prior steps."""

        @sylo.step("step-1")
        async def step_1(ctx: sylo.Context) -> dict:
            return {"value": 42}

        @sylo.step("step-2")
        async def step_2(ctx: sylo.Context) -> dict:
            prev = ctx.previous_outputs["step-1"]
            return {"doubled": prev["value"] * 2}

        async with sylo.pipeline("test-pipeline") as pipe:
            await step_1(pipe.context)
            result = await step_2(pipe.context)

        assert result == {"doubled": 84}
        assert pipe.context.previous_outputs["step-1"]["value"] == 42
        assert pipe.context.previous_outputs["step-2"]["doubled"] == 84


class TestTokenCostEstimation:
    """Tests for token cost calculation."""

    def test_estimate_cost_gpt4o(self):
        """GPT-4o costs should match hardcoded rates."""
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            model="gpt-4o",
        )
        cost = estimate_cost(usage)
        # 1000/1000 * 0.0025 + 500/1000 * 0.01 = 0.0025 + 0.005 = 0.0075
        assert cost == pytest.approx(0.0075, abs=0.0001)

    def test_estimate_cost_gpt4o_mini(self):
        """GPT-4o-mini should be much cheaper."""
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            model="gpt-4o-mini",
        )
        cost = estimate_cost(usage)
        # 1000/1000 * 0.00015 + 500/1000 * 0.0006 = 0.00015 + 0.0003 = 0.00045
        assert cost == pytest.approx(0.00045, abs=0.0001)

    def test_estimate_cost_claude_sonnet(self):
        """Claude Sonnet costs should match hardcoded rates."""
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            model="claude-sonnet-4-6",
        )
        cost = estimate_cost(usage)
        # 1000/1000 * 0.003 + 500/1000 * 0.015 = 0.003 + 0.0075 = 0.0105
        assert cost == pytest.approx(0.0105, abs=0.0001)

    def test_estimate_cost_unknown_model_uses_default(self):
        """Unknown models should use the default rate."""
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            model="unknown-model-v1",
        )
        cost = estimate_cost(usage)
        # 1000/1000 * 0.002 + 500/1000 * 0.008 = 0.002 + 0.004 = 0.006
        assert cost == pytest.approx(0.006, abs=0.0001)

    def test_extract_token_usage_openai_format(self):
        """Should extract usage from OpenAI-style return values."""
        result = {
            "content": "Hello world",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "model": "gpt-4o",
            },
        }
        usage = extract_token_usage(result)
        assert usage is not None
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.model == "gpt-4o"
        assert usage.estimated_cost_usd > 0

    def test_extract_token_usage_none_for_no_usage(self):
        """Should return None when no usage data is present."""
        result = {"content": "Hello world"}
        usage = extract_token_usage(result)
        assert usage is None

    def test_extract_token_usage_none_for_non_dict(self):
        """Should return None for non-dict return values."""
        assert extract_token_usage("hello") is None
        assert extract_token_usage(42) is None

    @pytest.mark.asyncio
    async def test_step_extracts_token_usage(self, setup_sylo: Path):
        """Step should automatically extract token usage from return value."""
        storage = LocalStorage(root_dir=setup_sylo)

        @sylo.step("llm-call")
        async def llm_call(ctx: sylo.Context) -> dict:
            return {
                "content": "Generated text",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300,
                    "model": "gpt-4o",
                },
            }

        async with sylo.pipeline("test-pipeline") as pipe:
            await llm_call(pipe.context)
            exec_id = pipe.execution_id

        cp = await storage.get_checkpoint(exec_id, "llm-call")
        assert cp is not None
        assert cp.token_usage.total_tokens == 300
        assert cp.token_usage.model == "gpt-4o"
        assert cp.token_usage.estimated_cost_usd > 0

        # Pipeline-level token cost should be updated too
        assert pipe.record.token_cost.total_tokens == 300
        assert pipe.record.token_cost.estimated_cost_usd > 0


class TestRetryBehavior:
    """Tests for step retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, setup_sylo: Path):
        """Step should retry up to max_retries times before failing."""
        attempt_count = 0

        @sylo.step("flaky-step", max_retries=2, retry_delay=0.01)
        async def flaky_step(ctx: sylo.Context) -> dict:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError(f"Attempt {attempt_count} failed")
            return {"success": True}

        async with sylo.pipeline("test-pipeline") as pipe:
            result = await flaky_step(pipe.context)

        assert attempt_count == 3  # 1 initial + 2 retries
        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self, setup_sylo: Path):
        """After all retries, the original exception should propagate."""

        @sylo.step("always-fails", max_retries=2, retry_delay=0.01)
        async def always_fails(ctx: sylo.Context) -> dict:
            raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError, match="permanent failure"):
            async with sylo.pipeline("test-pipeline") as pipe:
                await always_fails(pipe.context)

    @pytest.mark.asyncio
    async def test_retry_saves_failed_checkpoint(self, setup_sylo: Path):
        """After exhausting retries, a FAILED checkpoint should be saved."""
        storage = LocalStorage(root_dir=setup_sylo)

        @sylo.step("fail-step", max_retries=1, retry_delay=0.01)
        async def fail_step(ctx: sylo.Context) -> dict:
            raise ValueError("boom")

        with pytest.raises(ValueError):
            async with sylo.pipeline("test-pipeline") as pipe:
                await fail_step(pipe.context)

        # Get exec_id from the pipe that's still in scope
        exec_id = pipe.execution_id
        cp = await storage.get_checkpoint(exec_id, "fail-step")
        assert cp is not None
        assert cp.status == CheckpointStatus.FAILED
        assert cp.retry_count == 2  # 1 initial + 1 retry

    @pytest.mark.asyncio
    async def test_no_retry_when_max_retries_is_zero(self, setup_sylo: Path):
        """Steps with max_retries=0 should fail immediately."""
        call_count = 0

        @sylo.step("no-retry", max_retries=0)
        async def no_retry(ctx: sylo.Context) -> dict:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError):
            async with sylo.pipeline("test-pipeline") as pipe:
                await no_retry(pipe.context)

        assert call_count == 1


class TestContext:
    """Tests for the Context object."""

    def test_context_creation(self):
        """Context should initialize with correct values."""
        ctx = Context(
            execution_id="test-id",
            pipeline_name="test-pipeline",
            run_number=3,
            metadata={"key": "value"},
        )
        assert ctx.execution_id == "test-id"
        assert ctx.pipeline_name == "test-pipeline"
        assert ctx.run_number == 3
        assert ctx.metadata["key"] == "value"
        assert ctx.previous_outputs == {}

    def test_get_output_raises_for_missing_step(self):
        """get_output should raise KeyError for missing steps."""
        ctx = Context(execution_id="test", pipeline_name="test")
        with pytest.raises(KeyError, match="No output found"):
            ctx.get_output("nonexistent")

    def test_get_output_returns_stored_output(self):
        """get_output should return previously stored outputs."""
        ctx = Context(execution_id="test", pipeline_name="test")
        ctx.previous_outputs["step-1"] = {"value": 42}
        assert ctx.get_output("step-1") == {"value": 42}


class TestCostReport:
    """Tests for the cost savings report."""

    @pytest.mark.asyncio
    async def test_cost_report_printed_in_dev_mode(
        self, setup_sylo: Path, capsys
    ):
        """Cost report should be printed to console in dev mode."""

        @sylo.step("step-1")
        async def step_1(ctx: sylo.Context) -> dict:
            return {
                "data": "hello",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "model": "gpt-4o",
                },
            }

        async with sylo.pipeline("test-pipeline") as pipe:
            await step_1(pipe.context)

        captured = capsys.readouterr()
        assert "Sylo execution complete" in captured.out
        assert "Steps:" in captured.out
        assert "Tokens:" in captured.out

    @pytest.mark.asyncio
    async def test_cost_report_shows_skipped_steps(
        self, setup_sylo: Path, capsys
    ):
        """Cost report should show skipped steps when resuming."""
        storage = LocalStorage(root_dir=setup_sylo)

        @sylo.step("cached-step")
        async def cached_step(ctx: sylo.Context) -> dict:
            return {"data": "hello", "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "model": "gpt-4o"}}

        # Run 1: normal execution
        async with sylo.pipeline("test-pipeline") as pipe:
            await cached_step(pipe.context)
            exec_id = pipe.execution_id

        # Copy checkpoint for run 2
        cp = await storage.get_checkpoint(exec_id, "cached-step")
        assert cp is not None

        # Run 2: with pre-loaded checkpoint
        async with sylo.pipeline("test-pipeline") as pipe2:
            cp.execution_id = pipe2.execution_id
            await storage.save_checkpoint(cp)
            await cached_step(pipe2.context)

        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower() or "Resumed" in captured.out


class TestStepOutsidePipeline:
    """Tests for running steps outside a pipeline context."""

    @pytest.mark.asyncio
    async def test_step_runs_without_pipeline(self):
        """Steps should execute normally outside a pipeline (no checkpointing)."""

        @sylo.step("standalone")
        async def standalone(ctx: sylo.Context) -> dict:
            return {"result": "works"}

        ctx = Context(execution_id="test", pipeline_name="test")
        result = await standalone(ctx)
        assert result == {"result": "works"}
