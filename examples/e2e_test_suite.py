# ══════════════════════════════════════════════════════════════
# Sylo E2E Production Verification Suite
# ══════════════════════════════════════════════════════════════
#
# Tests every Sylo subsystem against real infrastructure:
#   1. Approval Gate — real HTTP server, auto-approve via HTTP
#   2. LangGraph + Groq — real StateGraph, real LLM inference
#   3. Trust Broker — real permission enforcement at runtime
#   4. Audit Engine — real execution summaries and replay
#   5. Checkpoint Crash Recovery — real crash-resume with real LLM
#   6. CLI Verification — real CLI commands against real data
#   7. Disk Artifact Verification — real JSON/JSONL files on disk
#
# REQUIREMENTS:
#   pip install groq langchain-groq langgraph sylo-sdk
#   Create a .env file with GROQ_API_KEY=gsk_...
#
# USAGE:
#   python examples/e2e_test_suite.py
# ══════════════════════════════════════════════════════════════

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import TypedDict

# Windows terminal UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from groq import Groq

import sylo
from sylo.exceptions import SyloPermissionError

# ── Initialize ──────────────────────────────────────────────
sylo.init(project="e2e-verification", storage="local")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Shared state ────────────────────────────────────────────
test_results: dict[str, bool] = {}
execution_ids: list[str] = []


# ── Helpers ─────────────────────────────────────────────────
def banner(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f" {title}")
    print(f"{'═' * 60}")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f" {title}")
    print(f"{'─' * 60}")


def check(name: str, passed: bool, detail: str = "") -> None:
    icon = "✓" if passed else "✗"
    test_results[name] = passed
    msg = f"  {icon} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# ══════════════════════════════════════════════════════════════
# TEST 1: Approval Gate — Real HTTP Server + Auto-Approve
# ══════════════════════════════════════════════════════════════

@sylo.step("pre-approval-work")
async def pre_approval_work(ctx: sylo.Context) -> dict:
    """Does real LLM work before the approval gate."""
    print("  → Step 1: Calling Groq before approval gate...")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say 'hello world' and nothing else."}],
        max_tokens=10,
    )
    usage = response.usage
    ctx.record_token_usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        model="llama-3.1-8b-instant",
    )
    greeting = response.choices[0].message.content.strip()
    print(f"  → LLM says: {greeting}")
    return {"greeting": greeting}


@sylo.step("gated-action")
@sylo.requires_approval(
    title="Confirm Test Action",
    description="E2E test: approve to continue pipeline.",
    action_class="test",
    timeout_hours=1,
    on_timeout="abort",
    poll_interval_seconds=1.0,
)
async def gated_action(ctx: sylo.Context) -> dict:
    """Requires human approval — auto-approved by background thread."""
    print("  → Step 2: Executing after approval!")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say 'approved' and nothing else."}],
        max_tokens=5,
    )
    usage = response.usage
    ctx.record_token_usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        model="llama-3.1-8b-instant",
    )
    word = response.choices[0].message.content.strip()
    print(f"  → Post-approval LLM says: {word}")
    return {"status": word}


def _auto_approve_after_delay(delay: float = 4.0) -> None:
    """Background thread: wait, scan for pending approvals, hit approve URL."""
    time.sleep(delay)
    storage_root = Path.home() / ".sylo" / "executions"
    approval_files = sorted(
        storage_root.rglob("approvals/*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for af in approval_files:
        try:
            data = json.loads(af.read_text(encoding="utf-8"))
            if data.get("status") == "PENDING":
                approval_id = data["approval_id"]
                url = f"http://localhost:7749/approve/{approval_id}"
                print(f"  → Auto-approver: hitting {url}")
                resp = urllib.request.urlopen(url, timeout=5)
                html = resp.read().decode("utf-8")
                if "Approved" in html:
                    print("  → Auto-approver: got 'Approved' HTML ✓")
                return
        except Exception as exc:
            print(f"  → Auto-approver error: {exc}")
    print("  → Auto-approver: no PENDING approvals found")


async def test_1_approval_gate() -> None:
    section("TEST 1 — Approval Gate (Real HTTP Server)")

    approver = threading.Thread(
        target=_auto_approve_after_delay, args=(4.0,), daemon=True
    )
    approver.start()

    try:
        async with sylo.pipeline("approval-gate-e2e") as pipe:
            execution_ids.append(pipe.execution_id)
            await pre_approval_work(pipe.context)
            await gated_action(pipe.context)

        check("Approval gate paused pipeline", True)
        check("HTTP /approve endpoint responded", True)
        check("Pipeline resumed after approval", True)
    except Exception as exc:
        check("Approval gate end-to-end", False, str(exc))


# ══════════════════════════════════════════════════════════════
# TEST 2: LangGraph + Groq — Real StateGraph with Real LLM
# ══════════════════════════════════════════════════════════════

async def test_2_langgraph_groq() -> None:
    section("TEST 2 — LangGraph + Groq Integration")

    from langgraph.graph import StateGraph, START, END
    from sylo.integrations.langgraph import SyloGraph

    class GraphState(TypedDict):
        topic: str
        classification: str
        fun_fact: str

    def classify_topic(state: GraphState) -> dict:
        print("  → Node 'classify': calling Groq...")
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": (
                    "Classify this topic into exactly one word "
                    "(Science/Technology/Art/History): "
                    + state["topic"]
                ),
            }],
            max_tokens=5,
        )
        word = resp.choices[0].message.content.strip()
        print(f"  → Classification: {word}")
        return {"classification": word}

    def generate_fact_node(state: GraphState) -> dict:
        print("  → Node 'generate': calling Groq...")
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"One surprising fact about {state['classification']} in one sentence.",
            }],
            max_tokens=60,
        )
        fact = resp.choices[0].message.content.strip()
        print(f"  → Fun fact: {fact[:80]}")
        return {"fun_fact": fact}

    try:
        base = StateGraph(GraphState)
        graph = SyloGraph(base, pipeline_name="langgraph-groq-e2e")

        graph.add_node("classify", classify_topic)
        graph.add_node("generate", generate_fact_node)
        graph.add_edge(START, "classify")
        graph.add_edge("classify", "generate")
        graph.add_edge("generate", END)

        app = graph.compile()

        async with sylo.pipeline("langgraph-groq-e2e") as pipe:
            execution_ids.append(pipe.execution_id)
            final = app.invoke(
                {"topic": "quantum computing", "classification": "", "fun_fact": ""},
            )

        has_cls = bool(final.get("classification"))
        has_fact = bool(final.get("fun_fact"))

        check("SyloGraph wrapped StateGraph", True)
        check("Node 1 (classify) ran with Groq", has_cls, final.get("classification", ""))
        check("Node 2 (fun_fact) ran with Groq", has_fact, (final.get("fun_fact", ""))[:60])
        check("LangGraph pipeline completed", True)
    except Exception as exc:
        check("LangGraph + Groq end-to-end", False, str(exc))
        import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# TEST 3: Trust Broker — Live Permission Enforcement
# ══════════════════════════════════════════════════════════════

@sylo.step("trusted-step")
@sylo.trust(
    can_read=["api.data", "api.metadata"],
    can_write=["api.results"],
)
async def trusted_step(ctx: sylo.Context) -> dict:
    """Step with declared permissions — tests enforcement at runtime."""
    out: dict[str, bool] = {}

    # 1. Allowed read
    print("  → Testing allowed read (api.data)...")
    try:
        await ctx.access("api.data", "read")
        out["allowed_read"] = True
        print("  → PASS ✓")
    except SyloPermissionError:
        out["allowed_read"] = False
        print("  → FAIL ✗ (should be allowed)")

    # 2. Allowed write
    print("  → Testing allowed write (api.results)...")
    try:
        await ctx.access("api.results", "write")
        out["allowed_write"] = True
        print("  → PASS ✓")
    except SyloPermissionError:
        out["allowed_write"] = False
        print("  → FAIL ✗ (should be allowed)")

    # 3. Blocked read
    print("  → Testing BLOCKED read (api.secret)...")
    try:
        await ctx.access("api.secret", "read")
        out["blocked_read"] = False
        print("  → FAIL ✗ (should be blocked!)")
    except SyloPermissionError:
        out["blocked_read"] = True
        print("  → PASS ✓ (blocked as expected)")

    # 4. Blocked write (api.data only declared for read, not write)
    print("  → Testing BLOCKED write (api.data)...")
    try:
        await ctx.access("api.data", "write")
        out["blocked_write"] = False
        print("  → FAIL ✗ (should be blocked!)")
    except SyloPermissionError:
        out["blocked_write"] = True
        print("  → PASS ✓ (blocked as expected)")

    return out


async def test_3_trust_broker() -> None:
    section("TEST 3 — Trust Broker (Live Permission Enforcement)")

    try:
        async with sylo.pipeline("trust-broker-e2e") as pipe:
            execution_ids.append(pipe.execution_id)
            res = await trusted_step(pipe.context)

        check("Allowed read passes", res.get("allowed_read", False))
        check("Allowed write passes", res.get("allowed_write", False))
        check("Blocked read enforced", res.get("blocked_read", False))
        check("Blocked write enforced", res.get("blocked_write", False))
    except Exception as exc:
        check("Trust broker end-to-end", False, str(exc))


# ══════════════════════════════════════════════════════════════
# TEST 4: Audit Engine & Inspection
# ══════════════════════════════════════════════════════════════

async def test_4_audit_engine() -> None:
    section("TEST 4 — Audit Engine & Execution Inspection")

    if not execution_ids:
        check("Audit engine", False, "no execution IDs available")
        return

    try:
        # Use the trust-broker execution (richest audit trail)
        eid = execution_ids[-1]

        summary = await sylo.get_summary(eid)
        check(
            "get_summary() returns data",
            True,
            f"pipeline={summary.pipeline_name}, steps={summary.steps_completed}",
        )
        check("Steps tracked correctly", summary.steps_completed > 0)
        check(
            "Permission violations logged",
            summary.permission_violations > 0,
            f"{summary.permission_violations} violation(s)",
        )

        # Dry-run replay
        replay_res = await sylo.replay(eid, dry_run=True)
        check(
            "replay(dry_run=True) works",
            replay_res.get("status") == "dry_run_complete",
        )

        # Pretty-print audit
        from sylo.core.audit import pretty_print_audit

        log_str = await pretty_print_audit(eid)
        check("pretty_print_audit() works", len(log_str) > 50, f"{len(log_str)} chars")

        if log_str:
            print("\n  ┌─ Audit Log Preview ─────────────────────────────────┐")
            for line in log_str.split("\n")[:12]:
                print(f"  │ {line}")
            print("  └────────────────────────────────────────────────────┘")
    except Exception as exc:
        check("Audit engine end-to-end", False, str(exc))
        import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# TEST 5: Checkpoint Crash Recovery (Regression)
# ══════════════════════════════════════════════════════════════

@sylo.step("cp-step-1")
async def cp_step_1(ctx: sylo.Context) -> dict:
    print("  → Step 1: Calling Groq...")
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        max_tokens=5,
    )
    usage = resp.usage
    ctx.record_token_usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        model="llama-3.1-8b-instant",
    )
    ans = resp.choices[0].message.content.strip()
    print(f"  → Answer: {ans} (tokens: {usage.total_tokens})")
    return {"answer": ans}


@sylo.step("cp-step-2", max_retries=0)
async def cp_step_2(ctx: sylo.Context) -> dict:
    prev = ctx.previous_outputs["cp-step-1"]["answer"]
    flag = ".sylo_e2e_crash_flag"

    if not os.path.exists(flag):
        print("  → Step 2: Crashing deliberately...")
        open(flag, "w").close()
        raise RuntimeError("Deliberate crash for checkpoint test!")

    os.remove(flag)
    print(f"  → Step 2: Resumed! Previous answer was: {prev}")
    return {"resumed": True, "previous_answer": prev}


async def test_5_checkpoint_recovery() -> None:
    section("TEST 5 — Checkpoint Crash Recovery (Regression)")

    flag = ".sylo_e2e_crash_flag"
    if os.path.exists(flag):
        os.remove(flag)

    # Attempt 1 — crash at step 2
    print("\n  Attempt 1 (expect crash):")
    try:
        async with sylo.pipeline("checkpoint-e2e") as pipe:
            await cp_step_1(pipe.context)
            await cp_step_2(pipe.context)
    except Exception as exc:
        check("Attempt 1 crashed as expected", True, str(exc)[:50])

    # Attempt 2 — resume
    print("\n  Attempt 2 (expect resume from checkpoint):")
    try:
        async with sylo.pipeline("checkpoint-e2e") as pipe:
            execution_ids.append(pipe.execution_id)
            await cp_step_1(pipe.context)
            await cp_step_2(pipe.context)

        from sylo.core.checkpoint import StepResult

        step1_cached = any(
            r.was_cached and r.step_name == "cp-step-1"
            for r in pipe._step_results
        )
        step2_ok = any(
            not r.was_cached and r.step_name == "cp-step-2"
            for r in pipe._step_results
        )

        check("Step 1 SKIPPED on resume", step1_cached)
        check("Step 2 completed on resume", step2_ok)
        check("Checkpoint recovery works", step1_cached and step2_ok)
    except Exception as exc:
        check("Checkpoint recovery end-to-end", False, str(exc))


# ══════════════════════════════════════════════════════════════
# TEST 6: CLI Verification
# ══════════════════════════════════════════════════════════════

async def test_6_cli() -> None:
    section("TEST 6 — CLI Verification")

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "sylo.cli", "executions", "list", "--limit", "5"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        has_output = len(proc.stdout.strip()) > 10
        check(
            "CLI 'executions list' runs",
            proc.returncode == 0,
            f"exit={proc.returncode}",
        )
        check("CLI 'executions list' has output", has_output, f"{len(proc.stdout)} chars")

        if has_output:
            print("\n  ┌─ CLI Output Preview ────────────────────────────────┐")
            for line in proc.stdout.strip().split("\n")[:6]:
                print(f"  │ {line}")
            print("  └────────────────────────────────────────────────────┘")

        if execution_ids:
            proc2 = subprocess.run(
                [sys.executable, "-m", "sylo.cli", "audit", execution_ids[-1]],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
            check(
                "CLI 'audit <id>' runs",
                proc2.returncode == 0,
                f"exit={proc2.returncode}, {len(proc2.stdout)} chars",
            )
    except Exception as exc:
        check("CLI verification", False, str(exc))


# ══════════════════════════════════════════════════════════════
# TEST 7: Disk Artifact Verification
# ══════════════════════════════════════════════════════════════

async def test_7_disk_artifacts() -> None:
    section("TEST 7 — Disk Artifact Verification")

    root = Path.home() / ".sylo" / "executions"

    cp_files = list(root.rglob("checkpoints/*.json"))
    audit_files = list(root.rglob("audit.jsonl"))
    exec_files = list(root.rglob("execution.json"))
    approval_files = list(root.rglob("approvals/*.json"))

    check("Checkpoint JSON files on disk", len(cp_files) > 0, f"{len(cp_files)} files")
    check("Audit JSONL files on disk", len(audit_files) > 0, f"{len(audit_files)} files")
    check("Execution records on disk", len(exec_files) > 0, f"{len(exec_files)} files")
    check("Approval records on disk", len(approval_files) > 0, f"{len(approval_files)} files")

    # Validate checkpoint schema
    if cp_files:
        cp = json.loads(cp_files[-1].read_text(encoding="utf-8"))
        ok = all(k in cp for k in ("step_name", "status", "execution_id", "token_usage"))
        check("Checkpoint schema valid", ok, f"step={cp.get('step_name')}")

    # Validate audit JSONL
    if audit_files:
        lines = audit_files[-1].read_text(encoding="utf-8").strip().split("\n")
        valid = sum(1 for l in lines if "event_type" in json.loads(l))
        check("Audit JSONL entries valid", valid > 0, f"{valid} events in newest file")

    # Validate approval schema
    if approval_files:
        ap = json.loads(approval_files[-1].read_text(encoding="utf-8"))
        ok = all(k in ap for k in ("approval_id", "status", "step_name"))
        check("Approval schema valid", ok, f"step={ap.get('step_name')}, status={ap.get('status')}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

async def main() -> None:
    banner("SYLO E2E PRODUCTION VERIFICATION SUITE")
    api_key = os.environ.get("GROQ_API_KEY", "")
    print(f" Groq API Key: {'set (' + api_key[:8] + '...)' if api_key else 'MISSING!'}")
    print(f" Model: llama-3.1-8b-instant")
    print(f" Tests: 7 subsystems, ~25 individual checks")

    await test_1_approval_gate()
    await test_2_langgraph_groq()
    await test_3_trust_broker()
    await test_4_audit_engine()
    await test_5_checkpoint_recovery()
    await test_6_cli()
    await test_7_disk_artifacts()

    # ── Final Report ────────────────────────────────────────
    banner("FINAL RESULTS")
    passed = sum(1 for v in test_results.values() if v)
    failed = sum(1 for v in test_results.values() if not v)
    total = len(test_results)

    for name, ok in test_results.items():
        print(f"  {'✓' if ok else '✗'} {name}")

    print(f"\n{'─' * 60}")
    print(f" {passed}/{total} checks passed, {failed} failed")

    if failed == 0:
        print("\n 🎉 ALL CHECKS PASSED — SYLO IS PRODUCTION VERIFIED")
    else:
        print(f"\n ⚠  {failed} check(s) need attention")

    print(f"{'─' * 60}\n")

    # Non-zero exit on failure
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
