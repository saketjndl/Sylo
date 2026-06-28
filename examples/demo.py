"""
Sylo Demo Script: Reliable Agent Pipelines
=========================================
Demonstrates checkpointing, crash resumption, and cost tracking.
"""

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import sylo
from sylo.config import set_config, SyloConfig


# 1. Initialize Sylo with local storage (no API key needed)
demo_storage_dir = Path("./_sylo_demo_storage")
if demo_storage_dir.exists():
    shutil.rmtree(demo_storage_dir)

os.environ["SYLO_STORAGE_DIR"] = str(demo_storage_dir.absolute())
config = SyloConfig(
    project="demo-project",
    storage="local",
)
set_config(config)


# Global flag to simulate a crash on the first run
CRASH_SIMULATED = True


@sylo.step("fetch-data")
async def fetch_data(ctx: sylo.Context, query: str) -> dict:
    print(f"  [Step 1: fetch-data] Executing query: '{query}'...")
    await asyncio.sleep(0.5)  # Simulate network latency
    
    # Simulate LLM token usage for data extraction
    ctx.record_token_usage(prompt_tokens=1200, completion_tokens=350, model="gpt-4o")
    print("  [Step 1: fetch-data] Completed! Recorded 1,550 LLM tokens ($0.0068 est. cost).")
    return {"raw_records": ["record_1", "record_2", "record_3"], "status": "ok"}


@sylo.step("process-data")
async def process_data(ctx: sylo.Context, data: dict) -> dict:
    global CRASH_SIMULATED
    print("  [Step 2: process-data] Processing records...")
    await asyncio.sleep(0.5)
    
    if CRASH_SIMULATED:
        print("\n💥 [CRASH SIMULATION] Unexpected system failure during step 2! Pipeline interrupted.")
        CRASH_SIMULATED = False
        raise RuntimeError("Simulated fatal exception in process-data step!")
        
    ctx.record_token_usage(prompt_tokens=800, completion_tokens=200, model="gpt-4o")
    print("  [Step 2: process-data] Completed! Recorded 1,000 LLM tokens ($0.0040 est. cost).")
    return {"processed_count": len(data["raw_records"]), "summary": "All records processed successfully."}


@sylo.step("generate-report")
async def generate_report(ctx: sylo.Context, stats: dict) -> dict:
    print("  [Step 3: generate-report] Generating final output report...")
    await asyncio.sleep(0.5)
    ctx.record_token_usage(prompt_tokens=500, completion_tokens=100, model="gpt-4o")
    print("  [Step 3: generate-report] Completed! Recorded 600 LLM tokens ($0.0022 est. cost).")
    return {"report_url": "https://sylo.dev/reports/demo-123", "final_stats": stats}


async def run_demo_pipeline(run_label: str):
    print(f"\n==================================================")
    print(f"🚀 STARTING PIPELINE RUN: {run_label}")
    print(f"==================================================")
    
    try:
        async with sylo.pipeline("analytics-pipeline", version="1.0") as pipe:
            # Step 1
            data = await fetch_data(pipe.context, query="Q3 sales figures")
            
            # Step 2
            stats = await process_data(pipe.context, data)
            
            # Step 3
            report = await generate_report(pipe.context, stats)
            
            print(f"\n🎉 Pipeline Finished Successfully! Report: {report['report_url']}")
            return pipe
    except RuntimeError as e:
        print(f"🛑 Pipeline terminated with error: {e}")
        return None


async def main():
    print("==================================================")
    print("🌟 SYLO DEMO: Checkpointing & Crash Resumption 🌟")
    print("==================================================")
    
    # Run 1: This will run Step 1, save its checkpoint, and crash during Step 2.
    await run_demo_pipeline("Attempt #1 (Will Crash at Step 2)")
    
    print("\n⏳ Simulating restarting the application after the crash...\n")
    await asyncio.sleep(1)
    
    # Run 2: This will auto-detect the failed run, resume from checkpoint for Step 1, and complete Step 2 & 3.
    pipe = await run_demo_pipeline("Attempt #2 (Auto-Resuming)")
    
    if pipe and pipe.record:
        cost = pipe.record.token_cost
        # Calculate tokens saved vs newly incurred
        saved_tokens = sum(s.token_usage.total_tokens for s in pipe._step_results if s.was_cached and s.token_usage)
        saved_cost = sum(s.cost_saved_usd for s in pipe._step_results if s.was_cached)
        new_tokens = cost.total_tokens
        new_cost = cost.estimated_cost_usd
        total_value_tokens = new_tokens + saved_tokens
        total_value_cost = new_cost + saved_cost

        print("\n==================================================")
        print("💰 COST SAVINGS & RELIABILITY SUMMARY")
        print("==================================================")
        print("By resuming from the checkpoint:")
        print("  ✅ Step 1 ('fetch-data') was loaded instantly from local cache.")
        print(f"  ✅ Saved {saved_tokens:,} tokens (${saved_cost:.4f} est. cost) by skipping re-execution of Step 1.")
        print("  ✅ Avoided repeating API calls and network latency.")
        print(f"  📊 New Run Tokens Incurred: {new_tokens:,} tokens (only for Steps 2 & 3)")
        print(f"  💵 New Run Cost Incurred:   ${new_cost:.4f}")
        print(f"  🏆 Total Value Delivered:   {total_value_tokens:,} tokens worth of work for only ${new_cost:.4f}!")
        print("==================================================\n")

    # Clean up demo storage directory
    if demo_storage_dir.exists():
        shutil.rmtree(demo_storage_dir)


if __name__ == "__main__":
    asyncio.run(main())
