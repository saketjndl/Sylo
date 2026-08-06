# Changelog

All notable changes to the Sylo SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] — 2026-08-06

### Fixed

- **`@sylo.step` silently fell back to passthrough when a pipeline's `ContextVar` was lost across a thread or event-loop boundary** ([#1](https://github.com/saketjndl/Sylo/issues/1)).
  Steps executed via thread-pool dispatches (e.g. `asyncio.to_thread`, `run_in_executor`) ran without checkpointing, trust enforcement, or audit logging — and no warning was raised. A `_managed_by_pipeline` flag on `Context` now distinguishes "no pipeline ever started" (legitimate passthrough) from "pipeline active but context lost," and raises `SyloContextError` with actionable guidance. A new `sylo.propagate_context()` utility lets callers explicitly carry pipeline context across thread and event-loop boundaries.

- **Retry loop in checkpoint engine retried deterministic policy errors** ([#2](https://github.com/saketjndl/Sylo/issues/2)).
  `SyloPermissionError`, `SyloApprovalRejectedError`, and `SyloConfigError` were routed through the full retry-with-backoff loop, wasting time on failures that would be identical on every attempt. These errors are now re-raised immediately without consuming the retry budget.

- **Console cost report printed `[SUCCESS]` unconditionally, even on failed executions** ([#3](https://github.com/saketjndl/Sylo/issues/3)).
  The development-mode cost summary now prints status-appropriate labels: `[SUCCESS]` for completed runs, `[FAILED]` (with the error message) for failures, `[PENDING]` for executions awaiting approval, and `[<STATUS>]` for any other terminal state.

## [0.1.3] — 2026-08-04

Initial public release.
