# REQUIREMENTS:
# pip install groq sylo-sdk
#
# Create a .env file with your API key:
# GROQ_API_KEY=gsk_xxxxxxxxxxxx

import asyncio
import glob
import json
import os
import sys
from groq import Groq

# Configure stdout for UTF-8 characters on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import sylo

sylo.init(project="real-world-test", storage="local")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─────────────────────────────────────────
# STEP 1: Summarize a real piece of text
# Uses a real Groq API call
# ─────────────────────────────────────────
@sylo.step("summarize-text", max_retries=2)
async def summarize_text(ctx: sylo.Context) -> dict:
    print("  → Calling Groq API for summarization...")
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": """Summarize this in exactly 2 sentences:

The transformer architecture, introduced in the paper 'Attention Is All You Need' in 2017, revolutionized natural language processing by replacing recurrent neural networks with self-attention mechanisms. This allowed for significantly more parallelization during training and enabled models to capture long-range dependencies in text more effectively. The architecture consists of an encoder and decoder, each made up of layers containing multi-head attention and feed-forward networks. Models like BERT, GPT, and T5 are all based on variations of this foundational architecture."""
            }
        ],
        max_tokens=150
    )
    
    # Extract real token usage from API response
    usage = response.usage
    ctx.record_token_usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        model="llama-3.1-8b-instant"
    )
    
    summary = response.choices[0].message.content
    print(f"  → Summary received: {summary[:80]}...")
    print(f"  → Real tokens used: {usage.prompt_tokens} prompt, {usage.completion_tokens} completion")
    
    return {
        "summary": summary,
        "tokens_used": usage.total_tokens
    }


# ─────────────────────────────────────────
# STEP 2: Analyze the summary
# This step will CRASH on first run deliberately
# ─────────────────────────────────────────
@sylo.step("analyze-summary", max_retries=0)
async def analyze_summary(ctx: sylo.Context) -> dict:
    summary = ctx.previous_outputs["summarize-text"]["summary"]
    
    # On first run: crash AFTER receiving real API response
    # Check if this is a resume run by looking at run metadata
    crash_flag_file = ".sylo_crash_flag"
    
    if not os.path.exists(crash_flag_file):
        # First run: make the API call, then crash
        print("  → Calling Groq API for analysis...")
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user", 
                    "content": f"What field of computer science does this relate to? One word answer only: {summary}"
                }
            ],
            max_tokens=10
        )
        
        usage = response.usage
        ctx.record_token_usage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            model="llama-3.1-8b-instant"
        )
        
        # Create crash flag then simulate crash
        open(crash_flag_file, 'w').close()
        raise RuntimeError("Simulated crash after real API call completed!")
    
    else:
        # Second run: complete normally
        print("  → Calling Groq API for analysis (resumed run)...")
        os.remove(crash_flag_file)
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": f"What field of computer science does this relate to? One word answer only: {summary}"
                }
            ],
            max_tokens=10
        )
        
        usage = response.usage
        ctx.record_token_usage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            model="llama-3.1-8b-instant"
        )
        
        field = response.choices[0].message.content.strip()
        print(f"  → Field identified: {field}")
        
        return {"field": field, "tokens_used": usage.total_tokens}


# ─────────────────────────────────────────
# STEP 3: Generate a fun fact
# Only runs if step 2 succeeded
# ─────────────────────────────────────────
@sylo.step("generate-fact")
async def generate_fact(ctx: sylo.Context) -> dict:
    field = ctx.previous_outputs["analyze-summary"]["field"]
    
    print("  → Calling Groq API for fun fact...")
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": f"Give me one surprising fun fact about {field} in exactly one sentence."
            }
        ],
        max_tokens=100
    )
    
    usage = response.usage
    ctx.record_token_usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        model="llama-3.1-8b-instant"
    )
    
    fact = response.choices[0].message.content.strip()
    print(f"  → Fun fact: {fact}")
    
    return {"fact": fact, "tokens_used": usage.total_tokens}


# ─────────────────────────────────────────
# MAIN: Run the pipeline twice
# ─────────────────────────────────────────
async def main():
    separator = "─" * 50
    
    print("\n" + "═" * 50)
    print(" SYLO REAL-WORLD TEST — Groq LLM Integration")
    print("═" * 50)
    
    # Clean up any leftover crash flag from a previous interrupted run
    if os.path.exists(".sylo_crash_flag"):
        os.remove(".sylo_crash_flag")

    # ── ATTEMPT 1 ──
    print(f"\n{separator}")
    print(" ATTEMPT 1 — Expect crash at step 2")
    print(separator)
    
    try:
        async with sylo.pipeline("real-world-test") as pipe:
            await summarize_text(pipe.context)
            await analyze_summary(pipe.context)
            await generate_fact(pipe.context)
    except Exception as e:
        print(f"\n  ✗ Pipeline crashed as expected: {e}")
        print("  ✓ Checkpoint saved for step 1")
    
    # ── ATTEMPT 2 ──
    print(f"\n{separator}")
    print(" ATTEMPT 2 — Expect resume from step 2")
    print(separator)
    
    async with sylo.pipeline("real-world-test") as pipe:
        await summarize_text(pipe.context)
        await analyze_summary(pipe.context)
        await generate_fact(pipe.context)
    
    print(f"\n{separator}")
    print(" VERIFICATION CHECKLIST")
    print(separator)
    print(" Did step 1 get SKIPPED on attempt 2? ........... check above")
    print(" Were REAL token counts logged? ................. check above")
    print(" Did step 3 receive step 2 output correctly? .... check above")
    print(" Check ~/.sylo/ for checkpoint JSON files")
    print(separator)
    
    # Show checkpoint files
    checkpoints = glob.glob(
        os.path.expanduser("~/.sylo/**/checkpoints/*.json"), 
        recursive=True
    )
    if checkpoints:
        checkpoints.sort(key=os.path.getmtime)
        print(f"\n Found {len(checkpoints)} checkpoint file(s):")
        for cp in checkpoints[-3:]:
            print(f"   {cp}")
            with open(cp, encoding="utf-8") as f:
                data = json.load(f)
                print(f"   Step: {data.get('step_name')} | Status: {data.get('status')} | Tokens: {data.get('token_usage', {}).get('total_tokens', 'N/A')}")
    
    print("\n✓ Real-world test complete.\n")


if __name__ == "__main__":
    asyncio.run(main())
