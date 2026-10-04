# Progress-driven execution with safety watchdogs

The saved backend evaluation `7501ed9f-479d-44cd-9621-1c4c152681ae` stopped
at the SDK turn budget after discovery, activation, a patch and a check, before
handoff. It also discovered `AuthService` instead of `AuthService.logout` and
attempted its first patch while pending. The frontend evaluation
`dc1bd621-1987-46ef-8f37-6b26cbbc106a` submitted after recovering from the same
activation error, but its local `AccountMenuProps` interface was outside the
old TSX grammar. Its SDK segment then exhausted. The activation denials and
verification transition back to `in_progress` were correct.

A later live `fd-dev-002` evaluation remained on its first candidate for more
than ten hours. Removing the total-turn limit left provider calls, complete SDK
segments, and evaluation attempts without finite safety deadlines. The old
detector ran only after a segment returned; novel reads and patch/check
revisions could also reset it without advancing toward task completion.

## One logical run, many internal segments

`AgentHarness` owns the logical loop. `AgentRunLimits.max_turns` is now an
optional total diagnostic limit, defaulting to `None`; `segment_turns` defaults
to ten. There is no fixed total turn count or total repair-attempt count.
Unlimited total turns does not mean unlimited unattended wall time: finite
watchdogs bound provider responses, segments, periods without durable
advancement, and each evaluation candidate attempt.
`OllamaAgentExecutor` invokes the SDK once per segment and translates only
`MaxTurnsExceeded` into `AgentResult.segment_exhausted`. Ordinary segment
exhaustion can continue. Failures, timeouts, and cancellation enter safe
reconciliation instead of blindly restarting the segment.

All segments share the original request, role, feature, task, workspace,
session and audit run. Explicit total limits clamp the final segment to the
remaining turns and raise `AgentTurnLimitError` when unfinished work reaches
the cap. CLI settings are `--max-turns`, `--segment-turns`, and
`--stall-segments`; Ctrl+C exits with status 130.

## Activity, durable advancement, and convergence

Activity includes reads, searches, discoveries, model responses, patches,
checks, and tool results. It supports observability and bounded continuation;
activity alone does not reset convergence watchdogs.

Durable advancement uses audited successful operations and persisted task
state, never model narration. The lifecycle moves through inspected,
activated, implemented, checked, submitted, verified, then completed or
validly blocked. Reaching a later required phase is advancement. Further
patches become evidence of convergence only when a structured check or
verification outcome improves. Fewer failing required checks, or a previous
failure passing without regression of an already passing required check, can
establish improvement.

Stable fingerprints exclude generated IDs, timestamps, narration, output
hashes, timing data, and patch-revision churn. Equivalent failures across
different revisions/submissions and short alternating failure cycles count
toward convergence detection. Failed or denied tool invocations never
establish advancement. A completed check reporting failure can reach the
initial checked phase; repeating its failure cannot advance again. Four
consecutive stagnant segments or three equivalent failed
states raise `AgentStalledError` by default. An injected monotonic clock also
enforces the deadline since durable advancement while a segment is active.
Long discovery remains valid inside configured bounds; distinct reads and
searches cannot keep an unattended run alive indefinitely. Segment exhaustion
alone is not a stall. There is no fixed total repair count.

For a bound developer, an authorized mutation or transition into active work
during the logical run starts the completion requirement. A persisted pending
handoff also requires verification. A premature final answer cannot stop
that task.
After each segment the harness verifies a pending handoff. Failed verification
returns the task to `in_progress`; fresh feedback directs inspection, repair,
checks and resubmission. Successful verification or a valid explicit `blocked`
transition terminates normally. Infrastructure/configuration failures,
cancellation, safety timeouts, diagnostic limits, and objective stalls
terminate with their own failure reasons. Read-only advice without active
work can finish normally.

## Continuation context and safety

Every continuation rebuilds the existing bounded authoritative feature/task
context and adds lifecycle phase, current revision/check state, a deterministic
sanitized operation ledger, successful changed paths, the latest handoff,
verification feedback, and the next required action. Bounded views report
omitted entries; operation fingerprints remain available to the
loop detector. Source bodies, raw audit previews, hidden reasoning and
model-written summaries are not copied into this ledger.

The SDK session callback excludes old conversation items from continued model
inputs while retaining persisted session rows. Initial history remains bounded
by the existing session policy. Successful mutations are not automatically
replayed or retried. Existing feature/task/role/workspace authorization,
trusted attribution, mutation serialization, patch preconditions,
duplicate-submission checks and deterministic verification remain enforced on
every call. A failed provider call ends that segment; a persisted pending
handoff remains available to the existing verification-resume path.
Unexpected workspace infrastructure failures are audited and propagated to
the harness. The SDK tool failure formatter is disabled so cancellation also
stops the logical run. Neither failure starts another segment or replays a
successful patch. Capability denials and failed patch preconditions retain
their existing explicit tool results.

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

## Enabled watchdogs and operator overrides

| Watchdog | Default | CLI override |
| --- | --- | --- |
| Provider response | 900 seconds (15 minutes) | `--provider-timeout-seconds` |
| Internal SDK segment | 1,800 seconds (30 minutes) | `--segment-timeout-seconds` |
| No durable advancement | 1,800 seconds (30 minutes) | `--advancement-timeout-seconds` |
| Evaluation candidate attempt | 2,700 seconds (45 minutes) | `--case-timeout-seconds` |
| Consecutive stagnant segments | 4 | `--stall-segments` |
| Equivalent failure states | 3 | `--equivalent-failure-threshold` |
| Cancellation cleanup grace | At most 10 seconds | `--cleanup-grace-seconds` |

Timeout values must be positive and finite. Thresholds must be positive
integers; the main CLI preserves the existing minimum of two for
`--stall-segments`. Cleanup cannot exceed ten seconds. The evaluation CLI
prints all effective settings before starting and offers no option to disable
every watchdog. Existing MCP and workspace-command timeouts remain enabled.

For a larger task, explicitly raise the relevant bounds:

```bash
uv run agent-team \
  --provider-timeout-seconds 1200 \
  --segment-timeout-seconds 3600 \
  --advancement-timeout-seconds 3600 \
  --stall-segments 6 \
  --equivalent-failure-threshold 5 \
  "Continue the assigned task."
```

The same runtime flags are available to `agent-team-eval run`, together with
its independent `--case-timeout-seconds` deadline for each candidate attempt.

## Uncertain outcomes and cancellation

A timeout can arrive immediately after a side effect. Before finalizing an
uncertain outcome, the harness rereads trusted task, handoff, verification, and
audit state. A completed or validly blocked task can finalize normally. A
durably pending handoff can use the existing safe verification-resume path.
Successful audited patches and checks survive; unknown mutation outcomes are
never safe to replay. If reconciliation cannot complete safely, the typed
failure remains resumable with sanitized diagnostics. Provider and
infrastructure failures never silently start another segment. A file may be
reread when its revision changed or repair requires current contents.

Ctrl+C requests cancellation immediately and exits with status 130. Model and
segment work, heartbeat tasks, and active trusted check processes are stopped.
Cleanup closes MCP subprocesses and SQLite sessions within a grace period of
at most ten seconds, then terminates remaining owned child processes safely.
The user's Ollama service is not terminated. Provider timeout, segment timeout,
objective stall, explicit cancellation, verification failure, and connection
failure remain separate outcomes.
Cleanup timeout also has a distinct reason. MCP transports own their process
groups and can terminate them before awaiting teardown, so a short grace
period cannot interrupt cleanup before child termination is requested.

## Evaluation checkpoints and liveness

The evaluation run ID is allocated before candidate execution. Atomic JSON
checkpoints are saved at run start, after each attempt and completed case, and
on timeout, cancellation, or unrecoverable failure. Saved states distinguish
`running`, `completed`, `interrupted`, `timed_out`, and `failed`. Historical
result files remain readable.

Timeouts cancel candidate activity and retain partial deterministic evidence
before the temporary workspace is destroyed. A timed-out candidate is not
retried as a connection failure. Later requested cases or repetitions can
continue after safe cleanup. Inspect a partial result with:

```bash
uv run agent-team-eval show RUN_ID --verbose
```

Saved diagnostics include case/repetition/attempt, model, duration, segment
and approximate turn counts, task/lifecycle state, the last completed safe
tool name, permitted changed-path metadata, check/verification outcomes and
failure classifications, time since advancement, and typed termination
details. They exclude hidden
reasoning, source bodies, patch contents, secret values, raw absolute paths,
and unbounded outputs.

While a candidate is active, the application emits a heartbeat at least every
thirty seconds through the existing reporting boundary. Interactive output
refreshes in place, showing lifecycle, waiting phase, last operation, time
since advancement, case deadline, and remaining time. For example:

```text
fd-dev-002 | candidate 1/1 | elapsed 00:12:30 | segment 2 |
task in_progress | phase checked | waiting model |
last operation run_check | check failed |
last advancement 00:04:10 ago | case deadline 00:45:00 | remaining 00:32:30
```

Waiting phases distinguish inference, tools, verification, and cleanup.
Heartbeats do not alter workflow state or convergence decisions. They stop
cleanly on every terminal path; `--no-progress` suppresses progress output.

## Deterministic reruns

These commands do not run live inference:

```bash
uv run pytest --no-cov -m "not ollama and not ollama_eval" \
  tests/unit/application/runtime tests/integration/runtime
uv run pytest --no-cov -m "not ollama and not ollama_eval" \
  tests/integration/ollama/test_ollama_segments.py
uv run pytest --no-cov -m "not ollama and not ollama_eval" \
  tests/unit/application/evaluation tests/integration/evaluation
uv run pytest --no-cov -m "not ollama and not ollama_eval" \
  tests/unit/interfaces/cli tests/integration/cli
uv run pytest --no-cov -m "not ollama and not ollama_eval" \
  tests/integration/skills tests/integration/persistence
uv run pytest --no-cov -m "not ollama and not ollama_eval" \
  tests/unit/infrastructure/evaluation/test_bounded_tsx_probe.py \
  tests/integration/evaluation/task_verification
```

No Ollama inference or live evaluation was run for this amendment. The first
post-fix live validation remains a human-operated check using the original,
already-installed `qwen3.6:27b`. The native case watchdog should terminate
first; GNU `timeout` provides temporary additional protection for that check:

```bash
timeout --signal=INT --kill-after=30s 50m \
  uv run agent-team-eval run \
  --suite frontend_developer_development \
  --case-id fd-dev-002 \
  --candidate-model qwen3.6:27b \
  --repetitions 1 \
  --no-judge \
  --infrastructure-retries 1 \
  --case-timeout-seconds 2700
```

```bash
timeout --signal=INT --kill-after=30s 50m \
  uv run agent-team-eval run \
  --suite backend_developer_development \
  --case-id bd-dev-002 \
  --candidate-model qwen3.6:27b \
  --repetitions 1 \
  --no-judge \
  --infrastructure-retries 1 \
  --case-timeout-seconds 2700
```
