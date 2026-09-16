# Verified Task Lifecycle

This document describes the local verified lifecycle for backend and frontend
implementation tasks. It is intentionally small: model agents may implement and
submit work, while deterministic harness code controls completion.

## State Transitions

| From | To | Controller | Notes |
| --- | --- | --- | --- |
| `pending` | `in_progress` | assigned developer or human workflow tool | Starts one task if no same feature and role task is active. |
| `pending` | `blocked` | assigned developer or human workflow tool | Records that work cannot proceed. |
| `blocked` | `in_progress` | assigned developer or human workflow tool | Reopens blocked work. |
| `in_progress` | `blocked` | assigned developer or human workflow tool | Records a blocker during implementation. |
| `in_progress` | `verification_pending` | `submit_task_for_verification` | Persists a structured handoff. |
| `verification_pending` | `in_progress` | deterministic verifier | Verification ran and at least one check failed. |
| `verification_pending` | `blocked` | deterministic verifier | Verification configuration or infrastructure prevented checks. |
| `verification_pending` | `completed` | deterministic verifier | All required trusted checks passed. |
| `completed` | any state | none for model-accessible tools | Completed is terminal for exposed model operations. |

`in_progress` and `verification_pending` consume the one active task slot for
the feature and assigned role. `blocked` and `completed` do not.

## Completion Authority

Backend and frontend developer agents can start, block, patch, run allowed
checks, and submit their own trusted bound task. They cannot directly select
`completed`.

The Delivery Manager remains a human-facing manual coordinator. It cannot use
the generic status tool to bypass deterministic verification completion.

The code boundary is the authority. Runtime instructions explain the policy,
but `WorkflowService`, `CapabilityAuthorizer`, MCP filtering, workspace
authorization, and SQLite compare-and-set operations enforce it.

Workspace mutation requires `in_progress`. SQLite holds `BEGIN IMMEDIATE`
across the authoritative status check and filesystem operation, serializing
it against submission and verification transitions across processes. A
submission that acquires the transaction first prevents later mutation;
otherwise the mutation finishes before submission can transition the task.
Verification finalization atomically checks `verification_pending` and the
same latest submission before persisting evidence and changing status.

## Verification Contract

Implementation tasks carry a minimal verification contract:

```json
{
  "profile_name": "backend",
  "required_checks": [
    "backend:ruff-format",
    "backend:ruff-check",
    "backend:pyright",
    "backend:pytest"
  ]
}
```

Frontend tasks use `profile_name: "frontend"` and required checks
`frontend:lint`, `frontend:typecheck`, `frontend:test`, and `frontend:build`.
Existing backend and frontend SQLite task rows with null contract columns
receive the safe default contract when read.

Backend verification runs `uv run ruff format --check .`,
`uv run ruff check .`, `uv run pyright`, and `uv run pytest` in order.
Frontend verification runs the configured npm scripts `lint`, `typecheck`,
`test`, and `build`. Each executed subcheck produces its own evidence; a
failure prevents completion. The repository's current `package.json` only
configures Allure reporting, so frontend verification fails closed until
the required frontend scripts and tools are configured.

The contract references trusted check names only. It never stores model-written
commands, shell snippets, filesystem roots, or acceptance-criteria copies.

## Handoff Schema

`submit_task_for_verification` persists bounded handoff data:

```json
{
  "id": 1,
  "task_id": 1,
  "agent_run_id": 7,
  "submitted_by": "backend_developer",
  "attribution": "agent:backend_developer",
  "implementation_summary": "Extended AuthService.logout.",
  "changed_paths": ["backend/auth.py"],
  "reused_symbols": ["AuthService.logout"],
  "new_symbols": [],
  "reuse_notes": "Existing logout method was extended.",
  "checks_attempted": ["backend"],
  "limitations": "none",
  "next_action": "run deterministic verification",
  "created_at": "2026-01-01T00:00:00+00:00"
}
```

The model-visible MCP schema omits trusted provenance fields. Runtime injects
the run ID, role, attribution, workspace identity, changed paths, and
`checks_attempted`. Model-supplied provenance fields are rejected. Changed
paths come from successful audited workspace `apply_patch` results in the
same run. Attempted checks are sorted, deduplicated names from completed
audited workspace `run_check` results in that run; nonzero exits and timeouts
still count as attempts. Calls denied or failed before a result is recorded
cannot establish an attempt. Submission requires at least one such result.
Captured check output is hashed in the audit projection and is not copied
into the attempted-check list.

Handoff limits are centralized in `TaskHandoffLimits`: summary 1,200
characters, reuse notes 800, limitations 600, next action 400, each list item
240, changed paths/reused symbols/new symbols 24 items each, attempted checks
12 items, and 5,000 total content characters. Schema and service validation
reject oversized submissions without truncating stored data. Historical
oversized handoffs are rendered within the same total content budget with
deterministic omitted-item counts; their stored contents remain unchanged.

The persisted handoff also contains `workspace_identity_hash`, omitted from
model context. An additive, idempotent SQLite migration adds this nullable
column so historical rows remain readable. Verification hashes the resolved
workspace path and compares it with the submission before running checks.
For legacy handoffs, identity may be recovered from the linked trusted agent
run; absent recoverable identity or a mismatch rejects verification without
running checks or changing status. Canonical paths and symlink aliases for
the same workspace produce the same identity.

## Verification Evidence

The deterministic verifier persists evidence for the latest submission:

```json
{
  "id": 1,
  "task_id": 1,
  "submission_id": 1,
  "verifier_name": "local_workspace_checks",
  "outcome": "passed",
  "failure_classification": "none",
  "feedback": "All required verification checks passed.",
  "checks": [
    {
      "id": 1,
      "verification_id": 1,
      "name": "backend:ruff-format",
      "exit_code": 0,
      "timed_out": false,
      "stdout_hash": "...",
      "stdout_excerpt": "ok",
      "stderr_hash": "...",
      "stderr_excerpt": ""
    }
  ]
}
```

Check output is bounded and sanitized, with hashes for correlation. A failed
check returns the task to `in_progress`. Timeout is classified separately from
ordinary check failure. Configuration or infrastructure errors mark the task
`blocked` with actionable feedback.

## Recovery

Submission reaches SQLite before verification. If the process is interrupted
after submission, the task remains `verification_pending` and can be resumed:

```bash
uv run agent-team verify-task --task-id 1 --workspace-root .
```

Verification is idempotent per submission. If evidence already exists for the
latest handoff, the service returns the existing evidence and does not rerun
developer mutations.

## Developer Sessions

Planning sessions remain feature-and-role scoped. Backend and frontend
developer sessions additionally bind to:

- the trusted task ID;
- a non-reversible hash of the resolved workspace identity.

The raw absolute workspace path is not stored in session or run audit records.
Historical developer sessions without task/workspace scope remain readable but
cannot be reused for new developer execution. Start a new task-scoped session
instead.

## Context Priority

Bound developer context is rebuilt from workflow storage each run and includes:

1. the bound feature;
2. allowed requirements and acceptance criteria;
3. architecture and implementation-plan artifacts;
4. the exact bound task and verification contract;
5. latest handoff;
6. latest verification feedback.

Unrelated same-role tasks are excluded from bound developer context.

## Offline Evaluation Verification

The four backend/frontend submission development cases use reviewed,
runner-only behavior contracts. The verifier records three required results:
the role aggregate for submitted paths, `workspace-diff`, and `behavior`.
The development goldens explicitly require all three persisted evidence rows.
The strict grader still rejects missing or unexpected database effects.
Mutation-only cases require activation from `pending` to `in_progress` before
patching, and explicitly expect that status update.

Backend probes inspect token revocation state across inputs and JSON export
preservation across event lists. Frontend probes inspect rendered message
props and a logout button's callback binding. Small bounded Python and TSX
interpreters support a closed source subset, documented in their modules;
candidate code is never imported or executed. Unsupported syntax fails
closed, so these checks are fixture verification, not a general Python or
React runtime. New submission cases need reviewed runner-side contracts.

Probe values and expected results are never copied into candidate prompts,
context, tools, workspace files, or feedback. Functional API requirements
are stated in the development tasks. Dataset loading remains compatible;
the changed development datasets have version `2026-09-16.0` and hashes
computed from their contents. Holdouts and historical results are unchanged.

## Future Roles

Future QA and code-review agents should consume verified submissions and
verification evidence as authoritative inputs. They should not reinterpret the
developer conversation, replay developer mutations, or weaken deterministic
completion evidence.

To add a future runnable role, update the role profile, instructions, context
policy, tool capabilities, skill visibility, evaluations, and completeness
tests together. A role should remain `placeholder` until all of those pieces
are deliberate.

## CLI Examples

List implementation status and capabilities:

```bash
uv run agent-team --list-agents
```

Run a bound developer task:

```bash
uv run agent-team \
  --role backend_developer \
  --feature-id 1 \
  --task-id 4 \
  --workspace-root . \
  "Implement my assigned task and submit it for verification."
```

Resume verification after a submitted handoff:

```bash
uv run agent-team verify-task --task-id 4 --workspace-root .
```

## Follow-Up Debt

The root `AGENTS.md` is long and should eventually become a shorter routing
page. That rewrite is intentionally deferred to avoid changing broad
instructions during this focused harness change.
