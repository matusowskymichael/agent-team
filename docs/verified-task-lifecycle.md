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

## Verification Contract

Implementation tasks carry a minimal verification contract:

```json
{
  "profile_name": "backend",
  "required_checks": ["backend"]
}
```

Frontend tasks use `profile_name: "frontend"` and `required_checks:
["frontend"]`. Existing backend and frontend SQLite task rows with null
contract columns receive the safe default contract when read.

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
the run ID, role, attribution, and changed paths. Changed paths are reconciled
from successful audited workspace `apply_patch` results in the same run.

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
      "name": "backend",
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
