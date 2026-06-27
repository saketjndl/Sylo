<div align="center">

# Luro

### Ship AI agents that don't break in production.

Luro is an open-source Python SDK and cloud platform that adds production-grade reliability to your existing AI agent pipelines — **checkpointing, permission enforcement, human approval gates, and immutable audit logging** — without replacing the frameworks you already use.

[![PyPI version](https://img.shields.io/pypi/v/luro-sdk.svg)](https://pypi.org/project/luro-sdk/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Works with** LangGraph · CrewAI · OpenAI Agents SDK

</div>

---

## The Problem

Building AI agents is easy. Running them in production is not.

| Problem | What happens | Cost |
|---|---|---|
| **Pipelines crash mid-run** | 4-step pipeline fails at step 3 — restarts from scratch | Wasted time, duplicated API calls, burned tokens |
| **Agents access too much** | An email-drafting agent reads your calendar, contacts, and files | Silent data leaks, compliance violations |
| **Irreversible actions fire automatically** | Agent deletes 847 customer records without asking | No approval, no audit trail, no undo |

## The Solution

Add three lines of code. Get production guarantees.

```python
import luro
from luro.integrations.langgraph import LuroGraph

luro.init(project="customer-onboarding", api_key="luro_xxx")

# Wrap your existing LangGraph pipeline — everything else stays the same
graph = LuroGraph(StateGraph(MyState), pipeline_name="onboarding")
```

That's it. Your pipeline now has:

- ✅ **Smart Checkpointing** — Resume from failure, not from scratch
- 🔒 **Trust Enforcement** — Agents can only touch what they declared
- ⏸️ **Approval Gates** — Pause before irreversible actions
- 📋 **Full Audit Log** — Replay any execution, know exactly what happened

## Quick Start

### Install

```bash
pip install luro-sdk
```

### Initialize

```python
import luro

# Local mode — no API key needed, data stays on disk
luro.init(project="my-first-project")

# Or connect to Luro Cloud for dashboard, team access, and persistence
luro.init(
    project="my-first-project",
    api_key="luro_xxx",
    environment="production",
    storage="cloud"
)
```

### Wrap Your Pipeline

```python
async with luro.pipeline("email-processor", version="1.0") as pipe:
    # Your existing agent code runs here — unchanged
    emails = await fetch_emails()
    summary = await summarize(emails)
    await send_report(summary)
```

Every run automatically gets:
- A unique execution ID
- Start/end timestamps
- Status tracking (RUNNING → COMPLETED or FAILED)
- Full audit trail

### See the Result

```
✓ Luro execution complete
  Steps: 3 completed, 0 skipped, 0 retried
  Tokens: 2,847 total | Est. cost: $0.043
  Duration: 4.2s
```

When your pipeline fails and restarts:

```
✓ Luro execution complete
  Steps: 1 completed, 2 skipped, 0 retried
  Tokens: 723 total | Est. cost: $0.012
  Resumed from checkpoint: step "summarize" (saved $0.031, 2.8s)
```

## How It Works

```
Your Pipeline
│
├─ Step 1: Fetch emails     ✓ checkpoint saved
├─ Step 2: Summarize        ✓ checkpoint saved
├─ Step 3: Save to Notion   ✗ FAILED
│
Without Luro: restart from Step 1 → cost $0.40, 14 seconds
With Luro:    resume from Step 3 → cost $0.04, 2 seconds
```

## Features

### 🔄 Smart Checkpointing
Every step's output is saved. If your pipeline fails at step 5, it resumes from step 5 — not step 1. You see exactly how much time and money you saved.

```python
@luro.step("fetch-emails", max_retries=3, retry_delay=2.0)
async def fetch_emails(ctx: luro.Context) -> dict:
    result = await call_llm(...)
    return result
```

### 🔒 Trust Broker
Declare what each agent step can access. Luro enforces it at runtime — not just documentation.

```python
@luro.step("send-email")
@luro.trust(
    can_read=["gmail.messages", "gmail.labels"],
    can_write=["gmail.drafts"],
    can_execute=["gmail.send"],
    can_delete=[]
)
async def send_email_step(ctx):
    emails = await ctx.access("gmail.messages", action="read", handler=fetch_fn)
```

### ⏸️ Human Approval Gates
Pause before dangerous actions. Approve from Slack, email, or the dashboard.

```python
@luro.step("delete-records")
@luro.requires_approval(
    title="Delete customer records",
    description="About to permanently delete {record_count} records",
    action_class="destructive",
    timeout_hours=24,
    notify=["email", "slack"]
)
async def delete_records(ctx):
    ...
```

### 📋 Immutable Audit Log
Every execution produces a complete, append-only audit trail. Replay any past execution.

```
Luro Audit Log — customer-onboarding — abc123
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
09:14:00.000  PIPELINE_STARTED
09:14:00.043  STEP_STARTED        fetch-emails
09:14:02.891  STEP_COMPLETED      fetch-emails        2848ms  $0.021
09:14:02.901  STEP_STARTED        summarize-emails
09:14:05.203  STEP_COMPLETED      summarize-emails    2302ms  $0.018
09:14:05.210  APPROVAL_REQUESTED  delete-records      (awaiting)
09:14:47.002  APPROVAL_DECISION   delete-records      APPROVED
09:14:47.041  STEP_STARTED        delete-records
09:14:48.112  STEP_COMPLETED      delete-records      1071ms
09:14:48.120  PIPELINE_COMPLETED

Total: 48.1s | $0.039 | 1,570 tokens | 0 violations | 1 approval
```

## Configuration

### Programmatic

```python
luro.init(
    project="my-project",
    api_key="luro_xxx",
    environment="production",     # "development" | "staging" | "production"
    storage="cloud",              # "local" | "redis" | "cloud"
)
```

### Environment Variables

```bash
LURO_API_KEY=luro_xxx
LURO_PROJECT=my-project
LURO_ENVIRONMENT=production
LURO_STORAGE=redis
LURO_REDIS_URL=redis://localhost:6379
```

## Architecture

```
┌──────────────────────────────────────────┐
│              Your Code                    │
│  (LangGraph / CrewAI / OpenAI Agents)    │
└───────────────┬──────────────────────────┘
                │
┌───────────────▼──────────────────────────┐
│           Luro SDK (Python)              │
│                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │Checkpoint│ │  Trust   │ │ Approval │ │
│  │ Engine   │ │ Broker   │ │  Gates   │ │
│  └──────────┘ └──────────┘ └──────────┘ │
│  ┌──────────────────────────────────────┐│
│  │         Audit & Replay Engine        ││
│  └──────────────────────────────────────┘│
└───────────────┬──────────────────────────┘
                │
┌───────────────▼──────────────────────────┐
│          Storage Backends                │
│  Local (dev) │ Redis (prod) │ Cloud      │
└──────────────────────────────────────────┘
```

## Development

```bash
# Clone the repo
git clone https://github.com/saketjndl/Luro.git
cd Luro/packages/luro-sdk

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Luro** — Production operating layer for AI agent pipelines.

[Documentation](https://docs.luro.dev) · [GitHub](https://github.com/saketjndl/Luro) · [Discord](https://discord.gg/luro)

</div>
