# ══════════════════════════════════════════════════════════════
# Sylo + OpenAI Agents SDK Integration Test
# ══════════════════════════════════════════════════════════════
#
# Tests the OpenAI Agents SDK integration against real Groq API:
#   1. Creates two agents (researcher + summarizer) using Groq
#   2. Runs step 1 → deliberately crashes → re-runs
#   3. Verifies step 1 is SKIPPED from checkpoint on resume
#   4. Prints real token usage from the Agents SDK
#
# REQUIREMENTS:
#   pip install openai-agents openai sylo-sdk
#   Create a .env file with GROQ_API_KEY=gsk_...
#
# USAGE:
#   python examples/test_openai_agents_integration.py
#
# KNOWN LIMITATIONS:
#   - The OpenAI Agents SDK has built-in tracing that tries to phone
#     home to OpenAI's servers. We disable it with set_tracing_disabled().
#   - Groq's OpenAI-compatible endpoint may not support all Agents SDK
#     features (e.g., structured outputs, handoffs).
#   - Token usage extraction from RunResult depends on provider support.
# ══════════════════════════════════════════════════════════════

import asyncio
import os
import sys

# Windows terminal UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import sylo
from sylo.integrations.openai_agents import wrap_agent, _extract_usage_from_result

# ── Initialize ──────────────────────────────────────────────
sylo.init(project="openai-agents-test", storage="local")


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
    msg = f"  {icon} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


async def main() -> None:
    banner("SYLO + OPENAI AGENTS SDK INTEGRATION TEST")

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("  ✗ GROQ_API_KEY not set in environment. Cannot run test.")
        sys.exit(1)

    print(f" Groq API Key: {api_key[:8]}...")
    print(f" Endpoint: https://api.groq.com/openai/v1")
    print(f" Model: openai/gpt-oss-20b")

    # ── Setup: Create agents using Groq ─────────────────────
    section("Setting up OpenAI Agents SDK with Groq backend")

    try:
        from openai import AsyncOpenAI
        from agents import Agent, Runner, set_tracing_disabled
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

        # Disable Agents SDK tracing (it tries to phone home to OpenAI)
        set_tracing_disabled(True)

        # Create a custom OpenAI client pointing at Groq
        groq_client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )

        # Wrap the client in the Agents SDK's model class
        groq_model = OpenAIChatCompletionsModel(
            model="openai/gpt-oss-20b",
            openai_client=groq_client,
        )

        print("  ✓ OpenAI Agents SDK imported and configured")
        print("  ✓ Groq client created with OpenAI-compatible endpoint")
        print("  ✓ Tracing disabled (not phoning home to OpenAI)")

    except ImportError as exc:
        print(f"  ✗ Failed to import required packages: {exc}")
        print("  Install with: pip install openai-agents openai")
        sys.exit(1)

    # ── Create and wrap agents ──────────────────────────────
    section("Creating wrapped agents")

    research_agent = Agent(
        name="Researcher",
        instructions=(
            "You are a research assistant. When given a topic, provide "
            "a concise 2-3 sentence summary of key facts about it. "
            "Be factual and informative."
        ),
        model=groq_model,
    )

    summary_agent = Agent(
        name="Summarizer",
        instructions=(
            "You are a summarizer. Take the provided research text and "
            "condense it into a single clear sentence that captures the "
            "main point."
        ),
        model=groq_model,
    )

    wrapped_research = wrap_agent(research_agent, step_name="research-step")
    wrapped_summary = wrap_agent(summary_agent, step_name="summary-step")

    print(f"  ✓ Research agent wrapped as step: {wrapped_research.step_name}")
    print(f"  ✓ Summary agent wrapped as step: {wrapped_summary.step_name}")

    # ── Attempt 1: Run step 1, then crash ───────────────────
    section("Attempt 1 — Run research step, then crash")

    crash_flag = ".sylo_openai_agents_crash_flag"
    if os.path.exists(crash_flag):
        os.remove(crash_flag)

    step1_output = None
    try:
        async with sylo.pipeline("openai-agents-e2e") as pipe:
            # Step 1: Research
            print("  → Running research agent via Groq...")
            step1_output = await wrapped_research.run(
                pipe.context, "quantum computing breakthroughs"
            )
            print(f"  → Research output: {step1_output[:100]}...")

            # Record token usage manually since we're using the wrapper
            # (The Agents SDK + Groq may not always return usage in raw_responses)
            result = await Runner.run(
                Agent(
                    name="dummy",
                    instructions="Say 'test'",
                    model=groq_model,
                ),
                "test",
            )
            # We already got our real output above, this is just to show
            # we can extract usage if available
            usage_info = _extract_usage_from_result(result)
            print(f"  → Token usage probe: {usage_info}")

            # Deliberate crash before step 2
            print("  → Deliberately crashing before summary step...")
            open(crash_flag, "w").close()
            raise RuntimeError("Deliberate crash for checkpoint test!")

    except RuntimeError as exc:
        if "Deliberate crash" in str(exc):
            check("Attempt 1 crashed as expected", True, str(exc)[:50])
        else:
            check("Attempt 1 unexpected error", False, str(exc))
            raise

    # ── Attempt 2: Resume — step 1 should be SKIPPED ────────
    section("Attempt 2 — Resume from checkpoint")

    if os.path.exists(crash_flag):
        os.remove(crash_flag)

    try:
        async with sylo.pipeline("openai-agents-e2e") as pipe:
            # Step 1: Should be loaded from checkpoint
            print("  → Running research agent (should skip)...")
            step1_resumed = await wrapped_research.run(
                pipe.context, "quantum computing breakthroughs"
            )

            # Step 2: Should actually run
            print("  → Running summary agent via Groq...")
            step2_output = await wrapped_summary.run(
                pipe.context,
                f"Summarize this research: {step1_resumed}",
            )
            print(f"  → Summary output: {step2_output[:100]}...")

        # Verify checkpoint behavior
        from sylo.core.checkpoint import StepResult

        step1_cached = any(
            r.was_cached and r.step_name == "research-step"
            for r in pipe._step_results
        )
        step2_ran = any(
            not r.was_cached and r.step_name == "summary-step"
            for r in pipe._step_results
        )

        check("Step 1 (research) SKIPPED on resume", step1_cached)
        check("Step 2 (summary) ran on resume", step2_ran)
        check(
            "Research output preserved across crash",
            step1_output is not None and len(step1_resumed) > 0,
        )
        check("Summary produced real output", len(step2_output) > 10)

    except Exception as exc:
        check("Attempt 2 (resume)", False, str(exc))
        import traceback
        traceback.print_exc()

    # ── Final Report ────────────────────────────────────────
    banner("OPENAI AGENTS SDK INTEGRATION TEST RESULTS")
    print("  Framework: OpenAI Agents SDK")
    print("  Backend: Groq (openai/gpt-oss-20b)")
    print(f"  Step 1 output: {(step1_output or '')[:80]}...")
    print(f"  Step 1 checkpoint skip: {'✓ Yes' if step1_cached else '✗ No'}")
    print(f"  Step 2 summary: {(step2_output if 'step2_output' in dir() else '')[:80]}...")

    if step1_cached and step2_ran:
        print("\n  🎉 OPENAI AGENTS SDK INTEGRATION VERIFIED")
    else:
        print("\n  ⚠  Partial verification — see details above")

    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
