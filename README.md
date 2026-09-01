# Agent Team

Agent Team is a local Python 3.14 multi-agent application. Runtime inference
uses local Ollama through the OpenAI Agents SDK, workflow tools are exposed by a
local MCP server over stdio, and SQLite stores workflow, session, audit, and
evaluation data.

## Package Tree

```text
src/agent_team/
    domain/
        runtime/
        workflow/
        context/
        sessions/
        audit/
        evaluation/
        skills/
    application/
        runtime/
        workflow/
        context/
        sessions/
        audit/
        evaluation/
        skills/
    infrastructure/
        ollama/
        mcp/
            client/
            server/
                schemas/
        persistence/
            sqlite/
                workflow/
                sessions/
                audit/
        evaluation/
        skills/
        configuration/
    interfaces/
        cli/
skills/
    write-requirements-artifact/
    write-acceptance-criteria/
    review-feature-readiness/
    review-architecture-readiness/
    design-solution-architecture/
    write-implementation-plan/
    decompose-development-tasks/
```

## Dependency Direction

The clean-architecture dependency direction is:

```text
interfaces -> application -> domain
```

Infrastructure implements domain and application ports, then the CLI
composition roots inject those adapters inward. Domain modules stay free of MCP,
Agents SDK, Ollama, SQLite, and CLI imports.

## Local Runtime Assumptions

The project is designed to run locally:

- Python is managed by `uv`.
- Runtime inference uses Ollama through the OpenAI Agents SDK.
- The default model is `qwen3.5:9b`.
- The default Ollama OpenAI-compatible base URL is
  `http://localhost:11434/v1`.
- Workflow, session, and audit data are stored in SQLite.
- Evaluation results are stored under `.agent_team/evals/`.
- Reviewed Agent Skills live under `skills/` and provide local procedural
  knowledge only.

Useful environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible Ollama base URL. |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Default runtime model for `agent-team`. |
| `OLLAMA_MAX_OUTPUT_TOKENS` | `8192` | Local generation output-token limit. |
| `OLLAMA_THINKING_ENABLED` | `false` | Opt-in thinking flag for supported Ollama models. |
| `OLLAMA_JUDGE_MODEL` | none | Default judge model for `agent-team-eval run`. |
| `AGENT_TEAM_DB_PATH` | `.agent_team/workflow.db` | SQLite database for workflow, sessions, and audit. |
| `RUN_OLLAMA_TESTS` | unset | Set to `1` to run opt-in live Ollama tests. |
| `RUN_OLLAMA_EVALS` | unset | Set to `1` to run opt-in live Ollama eval tests. |

## Profiles, Skills, Tools, Harness, And Workflow

- Profiles define the active role, workflow tool allowlist, visible skill
  names, and run limits.
- Skills are local `SKILL.md` knowledge packages. They help the model follow a
  procedure, but they never grant permissions.
- Tools are executable capabilities. Workflow tools come from the local MCP
  server; skill tools only load reviewed local skill text and resources.
- The shared `AgentHarness` selects the profile, prepares session/context,
  opens audit records, and invokes the runtime.
- Workflow data is authoritative SQLite state for features, artifacts, and
  development tasks.

At startup the harness advertises only allowed skill names and descriptions.
The model must call `load_skill` to read full instructions for a matching task.
Skill `allowed-tools` metadata is parsed for compatibility but is never
authoritative; `AgentProfile` and `CapabilityAuthorizer` remain the security
boundary.

## Command Catalog

Run commands through `uv run ...` from the repository root. Example output
below is illustrative where model wording or generated IDs can vary.

### Setup

Install or refresh the local environment:

```bash
uv sync
```

Example output:

```text
Resolved 12 packages in 5ms
Audited 8 packages in 0ms
```

Show the main agent CLI help:

```bash
uv run agent-team --help
```

Example output:

```text
usage: agent-team [-h] [--role {...}] [--feature-id FEATURE_ID]
                  [--session-id SESSION_ID] [--model MODEL]
                  [--list-models] [--list-skills] [prompt]
```

### Main Agent Runtime

Run one prompt through the shared `AgentHarness`, local Ollama model, and
workflow MCP tools:

```bash
uv run agent-team "Explain dependency inversion in one sentence."
```

Example output:

```text
Dependency inversion means high-level code depends on abstractions instead of
concrete low-level implementations.
```

Equivalent package entrypoint:

```bash
uv run python -m agent_team "Explain dependency inversion in one sentence."
```

Create workflow data through the local MCP server:

```bash
uv run agent-team \
  "Create a feature called User Authentication with a secure login and logout flow."
```

Example output:

```text
Created feature 1: User Authentication.
```

Run as a specific development role:

```bash
uv run agent-team \
  --role business_analyst \
  "Add requirements for feature 1 covering password reset."
```

Example output:

```text
Added requirements artifact 3 to feature 1.
```

Available roles:

```text
business_analyst
software_architect
backend_developer
frontend_developer
qa_engineer
code_reviewer
delivery_manager
```

Run with feature-scoped context:

```bash
uv run agent-team \
  --feature-id 1 \
  "Give me the complete details for this feature."
```

Example output:

```text
Feature 1: User Authentication
Status: draft
Artifacts: requirements artifact 3
Tasks: none
```

Run a Software Architect readiness review:

```bash
uv run agent-team \
  --model qwen3.6:27b \
  --role software_architect \
  --feature-id 1 \
  --session-id architect-feature-1 \
  "Review architecture readiness for this feature."
```

Example output:

```text
Feature 1 is not ready for architecture. Blocking questions: ...
```

Draft architecture without saving:

```bash
uv run agent-team \
  --role software_architect \
  --feature-id 1 \
  --session-id architect-feature-1 \
  "Draft an architecture proposal, but do not save it."
```

Example output:

```text
Unsaved architecture proposal:
1. Scope: ...
2. Recommendation: ...
```

Save architecture or implementation-plan artifacts only when explicitly
requested:

```bash
uv run agent-team \
  --role software_architect \
  --feature-id 1 \
  --session-id architect-feature-1 \
  "Save an architecture artifact for this feature."
```

Example output:

```text
Saved architecture artifact 4 for feature 1.
```

Create delivery tasks from an approved plan:

```bash
uv run agent-team \
  --role software_architect \
  --feature-id 1 \
  --session-id architect-feature-1 \
  "Create backend, frontend, QA, and code-review tasks for this feature."
```

Example output:

```text
Created tasks 8, 9, 10, and 11 for feature 1.
```

The Software Architect can read only the bound feature, create
`architecture` and `implementation_plan` artifacts, and create tasks assigned
to `backend_developer`, `frontend_developer`, `qa_engineer`, or
`code_reviewer`. It cannot create features, edit requirements or acceptance
criteria, update task statuses, write source code, use shell/filesystem tools,
or spoof `created_by`.

Continue or bind a local session:

```bash
uv run agent-team \
  --feature-id 1 \
  --session-id auth-planning \
  "Continue from the previous planning turn."
```

Example output:

```text
Continuing feature 1 in session auth-planning.
```

Use a different installed Ollama model for one run:

```bash
uv run agent-team \
  --model qwen3.5:9b \
  "List the workflow features currently stored."
```

Example output:

```text
1. User Authentication - draft
```

List locally installed Ollama models:

```bash
uv run agent-team --list-models
```

Example output:

```text
qwen3.5:9b
qwen3.8:27b
```

List local Agent Skills visible to a role without contacting Ollama:

```bash
uv run agent-team --list-skills --role business_analyst
```

Example output:

```text
NAME                            VERSION  HASH          DESCRIPTION
review-feature-readiness         0.1.0    8f3c2f1a9b0d  Use when asked to review, summarize, assess, or identify gaps in an existing feature.
write-acceptance-criteria        0.1.0    1440fb95b7de  Use for acceptance criteria, completion conditions, expected behavior, or testable feature outcomes.
write-requirements-artifact      0.1.0    5e61b1c99761  Use when asked to create, add, attach, or record requirements for an existing feature.
```

List Software Architect skills:

```bash
uv run agent-team --list-skills --role software_architect
```

Example output:

```text
NAME                         VERSION  HASH          DESCRIPTION
decompose-development-tasks   0.1.0    3b7b9c15f8aa  Use when asked to create, break down, plan, or decompose development tasks.
design-solution-architecture  0.1.0    6a3131433289  Use when asked to propose, draft, review, save, or explain solution architecture.
review-architecture-readiness 0.1.0    c977049e0cf2  Use when asked to review requirements, acceptance criteria, or readiness for architecture work.
write-implementation-plan     0.1.0    5f01e55ed36c  Use when asked to create, draft, review, save, or record an implementation plan.
```

The main agent prints only the final agent response to standard output. Common
local failures, such as unavailable Ollama, MCP startup failure, or database
migration failure, are printed concisely to standard error without a traceback.

### Human Audit CLI

The audit CLI is human-only. It reads already-sanitized audit records from
`AGENT_TEAM_DB_PATH`; it is not exposed as an MCP tool and does not modify
records.

List recent agent runs:

```bash
uv run agent-team-audit list-runs --limit 10
```

Example output:

```text
ID      STATUS      ROLE              MODEL        STARTED                       PROMPT
1       completed   delivery_manager  qwen3.5:9b   2026-08-20T10:00:00+00:00   Create a feature.
```

Show one run and its associated MCP tool calls:

```bash
uv run agent-team-audit show-run 1
```

Example output:

```text
Run 1
Status: completed
Role: delivery_manager
Model: qwen3.5:9b
Feature ID: 1
Session ID: session-1
Started: 2026-08-20T10:00:00+00:00
Ended: 2026-08-20T10:00:03+00:00
Max turns: 6
Prompt: Create a feature.
Output: Created feature 1.
Tool invocations:
- 1 completed development_workflow.create_feature (mutating)
  Arguments: {"title":"Login"}
  Result: {"id":1}
- 2 completed agent_skills.load_skill (read_only)
  Arguments: {"name":"write-requirements-artifact"}
  Result: {"content_hash":"...","loaded":true,"name":"write-requirements-artifact","version":"0.1.0"}
```

Unknown run IDs return a concise error:

```bash
uv run agent-team-audit show-run 404
```

Example output on standard error:

```text
Agent run 404 was not found.
```

### Local Evaluation CLI

The evaluation CLI is also human-only. It runs golden cases from `evals/`,
stores results in `.agent_team/evals/`, and can compare or calibrate previous
runs.

List available suites:

```bash
uv run agent-team-eval list-suites
```

Output:

```text
business_analyst_development
business_analyst_holdout
software_architect_development
software_architect_holdout
```

Run deterministic checks without a semantic judge:

```bash
uv run agent-team-eval run \
  --suite business_analyst_development \
  --case-id ba-dev-001 \
  --candidate-model qwen3.5:9b \
  --no-judge
```

Example output:

```text
Eval run: 6bee2561-7327-4dd2-a6a9-9813645f8d4d
Suite: business_analyst_development
Case filter: ba-dev-001
Candidate model: qwen3.5:9b
Judge model: -
Deterministic:
  passed: 1
  failed: 0
Semantic judge:
  not run: 1
  not applicable: 0
Overall fully evaluated:
  passed: 0
  incomplete: 1
```

Run with a local Ollama judge:

```bash
uv run agent-team-eval run \
  --suite business_analyst_development \
  --case-id ba-dev-001 \
  --candidate-model qwen3.5:9b \
  --judge-model qwen3.8:27b \
  --judge-repetitions 1
```

Example output:

```text
Eval run: 0bf00c80-3ac1-44e5-8e1a-8d29f1f7f112
Suite: business_analyst_development
Case filter: ba-dev-001
Candidate model: qwen3.5:9b
Judge model: qwen3.8:27b
Results:
  passed: 1
  deterministic_failed: 0
  not_judged: 0
  judge_failed: 0
  judge_error: 0
  ambiguous: 0
  total: 1
```

Run an Architect deterministic case:

```bash
uv run agent-team-eval run \
  --suite software_architect_development \
  --case-id sa-dev-004 \
  --candidate-model qwen3.6:27b \
  --no-judge
```

Run the full Architect development suite without a judge:

```bash
uv run agent-team-eval run \
  --suite software_architect_development \
  --candidate-model qwen3.6:27b \
  --no-judge
```

Run one judged Architect case:

```bash
uv run agent-team-eval run \
  --suite software_architect_development \
  --case-id sa-dev-003 \
  --candidate-model qwen3.6:27b \
  --judge-model qwen3.8:27b
```

Inspect a saved evaluation run:

```bash
uv run agent-team-eval show 6bee2561-7327-4dd2-a6a9-9813645f8d4d
```

Example output:

```text
Eval run: 6bee2561-7327-4dd2-a6a9-9813645f8d4d
Suite: business_analyst_development
Case: ba-dev-001
  Final status: passed
  Deterministic checks: passed
  Observed tool calls: [{"name":"get_feature_overview",...}]
```

Inspect full candidate, tool, and judge details:

```bash
uv run agent-team-eval show \
  6bee2561-7327-4dd2-a6a9-9813645f8d4d \
  --verbose
```

Example output:

```text
Candidate final response:
    Feature 1 contains one requirements artifact and no tasks.
Observed tool trajectory: ["get_feature_overview"]
Final status reason: final verdict is passed
```

Compare two saved evaluation runs:

```bash
uv run agent-team-eval compare \
  5d1bd522-0217-4cf4-bf69-55648835f061 \
  6bee2561-7327-4dd2-a6a9-9813645f8d4d
```

Example output:

```text
Baseline: 5d1bd522-0217-4cf4-bf69-55648835f061
Candidate: 6bee2561-7327-4dd2-a6a9-9813645f8d4d
Deterministic improvements: ba-dev-009
Deterministic regressions: -
Semantic improvements: -
Semantic regressions: -
Semantic not comparable: ba-dev-006
```

Use `--allow-non-equivalent` only when you intentionally compare runs with
different candidate models, datasets, instructions, or filters:

```bash
uv run agent-team-eval compare \
  baseline-run-id \
  candidate-run-id \
  --allow-non-equivalent
```

Calibrate a judged eval run against human labels:

```bash
uv run agent-team-eval calibrate \
  --eval-run-id 6bee2561-7327-4dd2-a6a9-9813645f8d4d \
  --human-labels evals/human_labels/example.jsonl
```

Example output:

```text
Verdict agreement: 0.80
Judge ambiguity rate: 0.10
Disagreements: ba-dev-013
```

Human label files are JSONL records documented in
`evals/human_labels/README.md`.

Evaluation exit codes:

| Exit code | Meaning |
| --- | --- |
| `0` | All fully evaluated cases passed, or a read-only eval command succeeded. |
| `1` | A quality gate failed. |
| `2` | A system/configuration error occurred. |

### Workflow MCP Server

Start the local workflow MCP server over stdio:

```bash
uv run agent-team-workflow-mcp
```

This command is not a human-facing shell UI. It waits for MCP JSON-RPC messages
on standard input, writes MCP protocol messages to standard output, and sends
logs to standard error only. The normal `agent-team` command launches this
server automatically when a run needs workflow tools.

Equivalent module entrypoint used by the Agents SDK subprocess:

```bash
uv run python -m agent_team.infrastructure.mcp.server.workflow_mcp_entrypoint
```

Example terminal behavior:

```text
# No prompt is printed.
# The process waits for MCP protocol input.
```

### Verification And Development Checks

Run the full required verification chain from `AGENTS.md`:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest
```

Example final output:

```text
All checks passed!
0 errors, 0 warnings, 0 informations
213 passed, 7 skipped
Required test coverage of 90% reached.
```

Run only the fast test suite selected by pytest configuration:

```bash
uv run pytest
```

Run a warning-focused teardown check:

```bash
uv run pytest --no-cov -W error::ResourceWarning
```

Example output:

```text
213 passed, 7 skipped
```

Run opt-in live Ollama integration tests:

```bash
RUN_OLLAMA_TESTS=1 uv run pytest tests/integration/ollama
```

Example output:

```text
2 passed
```

Run opt-in live Ollama evaluation smoke tests:

```bash
RUN_OLLAMA_EVALS=1 uv run pytest tests/integration/evaluation
```

Example output:

```text
5 passed
```
