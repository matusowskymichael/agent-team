# Progress-driven execution

The saved backend evaluation `7501ed9f-479d-44cd-9621-1c4c152681ae` stopped
at the SDK turn budget after discovery, activation, a patch and a check, before
handoff. It also discovered `AuthService` instead of `AuthService.logout` and
attempted its first patch while pending. The frontend evaluation
`dc1bd621-1987-46ef-8f37-6b26cbbc106a` submitted after recovering from the same
activation error, but its local `AccountMenuProps` interface was outside the
old TSX grammar. Its SDK segment then exhausted. The activation denials and
verification transition back to `in_progress` were correct.

## One logical run, many internal segments

`AgentHarness` owns the logical loop. `AgentRunLimits.max_turns` is now an
optional total diagnostic limit, defaulting to `None`; `segment_turns` defaults
to ten. There is no total execution deadline or repair-attempt count.
`OllamaAgentExecutor` invokes the SDK once per segment and translates only
`MaxTurnsExceeded` into `AgentResult.segment_exhausted`. Other failures and
cancellation leave the loop immediately.

All segments share the original request, role, feature, task, workspace,
session and audit run. Explicit total limits clamp the final segment to the
remaining turns and raise `AgentTurnLimitError` when unfinished work reaches
the cap. CLI settings are `--max-turns`, `--segment-turns`, and
`--stall-segments`; Ctrl+C exits with status 130.

## Objective progress and completion

Progress uses audited successful operations and persisted task state, never
model narration. New reads, searches, exact-symbol discoveries, task
transitions, changed patch contents, completed checks, structured handoffs and
verification results can advance a run. Repeated operations with identical
results, denied or failed calls, no-op patches, and submission ID/prose churn
do not establish continued progress. Check fingerprints use the observed
patch revision and check outcome; timing and diagnostic-output churn cannot
keep a stuck run alive. Verification fingerprints likewise exclude volatile
feedback and output hashes. An MCP error result is audited as a failed
operation even when its transport succeeded.

Three consecutive complete segments without new objective progress raise
`AgentStalledError`. The configurable threshold must be at least two. There is
no limit on distinct legitimate discovery operations, and ordinary segment
exhaustion is not itself a stall.

For a bound developer, an authorized mutation or transition into active work
during the logical run starts the completion requirement. A persisted pending
handoff also requires verification. A premature final answer cannot stop that task.
After each segment the harness verifies a pending handoff. Failed verification
returns the task to `in_progress`; fresh feedback directs inspection, repair,
checks and resubmission. Successful verification or a valid explicit `blocked`
transition terminates normally. Infrastructure/configuration failures,
cancellation, diagnostic limits and objective stalls terminate with their own
failure reasons. Read-only advice without active work can finish normally.

## Continuation context and safety

Every continuation rebuilds the existing bounded authoritative feature/task
context and adds a deterministic sanitized operation ledger, successful changed
paths, completed checks, the latest handoff, and verification feedback. Bounded
views report omitted entries; operation fingerprints remain available to the
loop detector. Source bodies, raw audit previews, hidden reasoning and
model-written summaries are not copied into this ledger.

The SDK session callback excludes old conversation items from continued model
inputs while retaining persisted session rows. Initial history remains bounded
by the existing session policy. Successful mutations are not automatically
replayed or retried. Existing feature/task/role/workspace authorization,
trusted attribution, mutation serialization, patch preconditions,
duplicate-submission checks and deterministic verification remain enforced on
every call. A failed provider call stops immediately; a persisted pending
handoff remains available to the existing verification-resume path.

Both developer skills and runtime instructions now order exact discovery,
activation, patch, aggregate check, immediate structured submission, then
verified final output. Skill tool metadata includes submission and remains
non-authoritative for permissions. Exact golden expectations are unchanged.

## TSX and persistence

The closed TSX parser additionally resolves leading local `interface`,
`export interface`, `type`, and `export type` prop declarations. Each must match
the expected string and zero-argument callback prop shape exactly. Imports,
inheritance, generics, unions/intersections, external types, executable
statements and custom components remain unsupported. Existing callback,
enabled/visible control and text checks still apply; JavaScript is never run.

Audit schema v5 adds nullable `total_turn_limit` and `termination_reason`, plus
`segment_count` defaulting to zero. Legacy `max_turns` remains populated with
the internal segment size for new runs. Existing rows, bindings and SDK session
history are preserved. No dependencies or golden datasets change.

## Deterministic reruns

These commands do not run live inference:

```bash
uv run pytest --no-cov tests/unit/application/runtime tests/integration/runtime
uv run pytest --no-cov tests/integration/ollama/test_ollama_segments.py
uv run pytest --no-cov tests/integration/skills tests/integration/persistence
uv run pytest --no-cov tests/unit/infrastructure/evaluation/test_bounded_tsx_probe.py tests/integration/evaluation/task_verification
```

The two targeted live evaluations are left for an explicitly requested local
rerun using the original, already-installed `qwen3.6:27b` model; they were
not run for this change:

```bash
uv run agent-team-eval run --suite backend_developer_development --case-id bd-dev-002 --candidate-model qwen3.6:27b --no-judge
uv run agent-team-eval run --suite frontend_developer_development --case-id fd-dev-002 --candidate-model qwen3.6:27b --no-judge
```
