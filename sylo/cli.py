"""Sylo CLI — command-line interface for inspecting pipeline executions.

Commands:
    sylo executions list [--pipeline NAME] [--limit N]
    sylo executions inspect <execution-id>
    sylo executions replay <execution-id> [--from-step STEP] [--dry-run]
    sylo audit <execution-id>
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import click

from sylo.config import SyloConfig, set_config, get_config, SyloConfigError


def _ensure_config() -> None:
    """Ensure a default config exists for CLI operations."""
    try:
        get_config()
    except SyloConfigError:
        # Initialize with defaults for local-only CLI usage
        set_config(SyloConfig(project="cli"))


def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


@click.group()
def cli() -> None:
    """Sylo — Production operating layer for AI agent pipelines."""
    pass


@cli.group()
def executions() -> None:
    """Manage pipeline executions."""
    pass


@executions.command("list")
@click.option("--pipeline", "-p", default=None, help="Filter by pipeline name.")
@click.option("--limit", "-l", default=20, help="Maximum number of results.")
def list_executions(pipeline: str | None, limit: int) -> None:
    """List recent pipeline executions."""
    _ensure_config()

    async def _list() -> None:
        from sylo.storage import get_storage

        config = get_config()
        storage = get_storage(config)

        if pipeline:
            records = await storage.list_executions(pipeline, limit)
        else:
            # List all executions by scanning local storage
            records = []
            if hasattr(storage, "_root"):
                import pathlib
                from sylo.models import ExecutionRecord

                root = storage._root  # type: ignore[attr-defined]
                if root.exists():
                    for exec_dir in root.iterdir():
                        if not exec_dir.is_dir():
                            continue
                        exec_file = exec_dir / "execution.json"
                        if exec_file.exists():
                            try:
                                data = exec_file.read_text(encoding="utf-8")
                                record = ExecutionRecord.model_validate_json(data)
                                records.append(record)
                            except Exception:
                                continue

            records.sort(key=lambda r: r.started_at, reverse=True)
            records = records[:limit]

        if not records:
            click.echo("No executions found.")
            return

        # Table header
        click.echo()
        header = f"{'EXECUTION ID':<38} {'PIPELINE':<25} {'STATUS':<20} {'STARTED':<24} {'DURATION':<12}"
        click.echo(header)
        click.echo("─" * len(header))

        for record in records:
            exec_id = record.execution_id[:8] + "..."
            pipeline_name = record.pipeline_name[:23]
            status = record.status.value
            started = record.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")

            duration = "—"
            if record.completed_at:
                secs = (record.completed_at - record.started_at).total_seconds()
                if secs < 60:
                    duration = f"{secs:.1f}s"
                else:
                    minutes = int(secs // 60)
                    remaining = secs % 60
                    duration = f"{minutes}m {remaining:.0f}s"

            # Color status
            status_display = status
            if status == "COMPLETED":
                status_display = click.style(status, fg="green")
            elif status == "FAILED":
                status_display = click.style(status, fg="red")
            elif status == "AWAITING_APPROVAL":
                status_display = click.style(status, fg="yellow")
            elif status == "RUNNING":
                status_display = click.style(status, fg="cyan")

            click.echo(
                f"{exec_id:<38} {pipeline_name:<25} {status_display:<29} {started:<24} {duration:<12}"
            )

        click.echo()

    _run_async(_list())


@executions.command("inspect")
@click.argument("execution_id")
def inspect_execution(execution_id: str) -> None:
    """Inspect a specific execution in detail."""
    _ensure_config()

    async def _inspect() -> None:
        from sylo.core.audit import get_summary

        try:
            summary = await get_summary(execution_id)
        except ValueError as exc:
            click.echo(click.style(str(exc), fg="red"))
            sys.exit(1)

        click.echo()
        click.echo(click.style(f"Execution: {summary.execution_id}", bold=True))
        click.echo(f"  Pipeline:   {summary.pipeline_name}")
        click.echo(f"  Status:     {summary.status}")
        click.echo(f"  Duration:   {summary.duration_seconds:.1f}s")
        click.echo(f"  Tokens:     {summary.total_tokens:,}")
        click.echo(f"  Cost:       ${summary.estimated_cost_usd:.4f}")
        if summary.cost_saved_usd > 0:
            click.echo(f"  Saved:      ${summary.cost_saved_usd:.4f} (from checkpoints)")
        click.echo(f"  Approvals:  {summary.approval_gates_hit}")
        click.echo(f"  Violations: {summary.permission_violations}")
        click.echo()

        if summary.timeline:
            click.echo(click.style("Steps:", bold=True))
            for step in summary.timeline:
                icon = "✓" if step.status == "completed" else "✗" if step.status == "failed" else "↻" if step.was_cached else "⏸"
                color = "green" if step.status == "completed" else "red" if step.status == "failed" else "yellow"
                status_text = click.style(f"[{step.status.upper()}]", fg=color)

                line = f"  {icon} {step.step_name:<30} {status_text}"
                if step.duration_ms > 0:
                    line += f"  {step.duration_ms}ms"
                if step.tokens > 0:
                    line += f"  {step.tokens:,} tokens  ${step.estimated_cost_usd:.4f}"
                if step.was_cached and step.cost_saved_usd > 0:
                    line += f"  (saved ${step.cost_saved_usd:.4f})"
                if step.error:
                    line += f"\n    Error: {step.error}"
                if step.retry_count > 0:
                    line += f"  ({step.retry_count} retries)"
                click.echo(line)

        click.echo()

    _run_async(_inspect())


@executions.command("replay")
@click.argument("execution_id")
@click.option("--from-step", default=None, help="Step name to replay from.")
@click.option("--dry-run", is_flag=True, help="Show what would happen without executing.")
def replay_execution(execution_id: str, from_step: str | None, dry_run: bool) -> None:
    """Replay a past execution from a specific step."""
    _ensure_config()

    async def _replay() -> None:
        from sylo.core.audit import replay

        try:
            result = await replay(
                execution_id=execution_id,
                from_step=from_step,
                dry_run=dry_run,
            )
        except ValueError as exc:
            click.echo(click.style(str(exc), fg="red"))
            sys.exit(1)

        click.echo()
        click.echo(click.style(f"Replay: {result['execution_id'][:8]}...", bold=True))
        click.echo(f"  Pipeline: {result['pipeline_name']}")
        click.echo(f"  Status:   {result['status']}")
        click.echo()

        if result.get("cached_steps"):
            click.echo("  Cached (from checkpoint):")
            for step in result["cached_steps"]:
                click.echo(f"    ↻ {step}")

        if result.get("replay_steps"):
            click.echo("  Replay (re-execute):")
            for step in result["replay_steps"]:
                click.echo(f"    ▶ {step}")

        if result.get("instructions"):
            click.echo()
            click.echo(f"  {result['instructions']}")

        click.echo()

    _run_async(_replay())


@cli.command("audit")
@click.argument("execution_id")
def audit_log(execution_id: str) -> None:
    """Pretty-print the full audit log for an execution."""
    _ensure_config()

    async def _audit() -> None:
        from sylo.core.audit import pretty_print_audit

        try:
            output = await pretty_print_audit(execution_id)
        except ValueError as exc:
            click.echo(click.style(str(exc), fg="red"))
            sys.exit(1)

        click.echo(output)

    _run_async(_audit())


def main() -> None:
    """Entry point for the sylo CLI."""
    cli()


if __name__ == "__main__":
    main()
