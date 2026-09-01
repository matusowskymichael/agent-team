# Agent Team Project Instructions

## Project

- This is a local Python 3.14 multi-agent application.
- Use `uv` for environments, dependencies, and command execution.
- Use `pyproject.toml` for all Python tool configuration.
- Do not use Git or run Git commands unless explicitly requested.
- Ask before adding or removing any production dependency.

## Architecture

Use a clean, layered architecture with this dependency direction:

`interfaces -> application -> domain`

Infrastructure implements domain or application ports and is injected inward.

- `domain`: Business models, value objects, exceptions, and protocols.
  It must not import agent SDKs, MCP libraries, CLI code, or infrastructure.
- `application`: Use cases and orchestration. It may depend on `domain`.
- `infrastructure`: OpenAI, MCP, storage, and other external adapters.
- `interfaces`: CLI and other user-facing entry points.

Rules:

- Do not introduce circular imports.
- Do not place business logic in CLI functions or infrastructure adapters.
- Keep external SDK objects out of the domain layer.
- Dependencies must point inward.
- Prefer dependency injection over constructing dependencies inside services.
- Prefer `typing.Protocol` for application boundaries.
- Avoid global mutable state and service locators.

## Package organization

- Keep the four top-level clean-architecture layers: `domain`,
  `application`, `infrastructure`, and `interfaces`.
- Group modules by cohesive subsystem inside each layer, such as `runtime`,
  `workflow`, `context`, `sessions`, `audit`, and `evaluation`.
- Keep one top-level class, dataclass, enum, protocol, or typed exception per
  production file.
- Do not introduce generic dumping-ground modules such as `models.py`,
  `services.py`, `ports.py`, `helpers.py`, `utils.py`, `types.py`, or
  `common.py`.
- Keep `__init__.py` files empty except for an optional package docstring.
  Do not add barrel re-exports.
- Organize tests so they mirror the production subsystems when practical.
- Pure organizational refactors must preserve behavior, schemas, prompts,
  capabilities, golden data, and rubrics.
- Architecture tests enforce dependency direction, one-class files, empty
  package initializers, importable entry points, and the dumping-ground-module
  ban.

## SOLID principles

- Single Responsibility: each class and function has one clear purpose. In
  addition, strictly follow a one class = one file paradigm no matter how
  closely related classes may be.
- Open/Closed: extend behavior through composition and implementations of
  protocols rather than large conditional blocks.
- Liskov Substitution: implementations must preserve their protocol contracts.
- Interface Segregation: use small, focused protocols.
- Dependency Inversion: application logic depends on abstractions, while
  infrastructure supplies concrete implementations.
- Do not create speculative abstractions without an actual boundary or use case.
- Prefer composition over inheritance.

## Python standards

- Follow PEP 8 and PEP 257.
- Keep lines at or below 79 characters.
- Add type annotations to all functions, methods, and meaningful variables.
- Public modules, classes, functions, and methods require docstrings.
- Use modern Python 3.14 syntax.
- Prefer `pathlib.Path` over string-based filesystem paths.
- Prefer small, explicit functions over clever or implicit code.
- Use descriptive names. Avoid unexplained abbreviations.
- Never silently catch broad exceptions.
- Never use `Any` unless unavoidable and documented.

## Testing

- Use pytest.
- Make use of pytests fixtures for setting up and tearing down.
- Make use of parameterisation where it makes sense.
- Fixtures must sit inside a conftest.py file, not inside the test file.
- All tests must be included as part of a test class.
- Place fast isolated tests under `tests/unit`.
- Place SDK, MCP, filesystem, or multi-component tests under
  `tests/integration`.
- Test behavior through public interfaces.
- Mock or fake protocols at architectural boundaries.
- Unit tests must not call external APIs or require network access.
- New behavior requires tests.
- Maintain at least 90% branch coverage.
- If a bug is found during execution, write a test for it then fix the bug.

## Working rules

- Inspect relevant files before editing.
- Before adding a class, function, method, endpoint, model, repository,
  service, utility, or component, search for the proposed symbol name.
- Search for existing code that may provide equivalent behavior.
- Read plausible matches before editing.
- Reuse or extend existing code where appropriate.
- If new code is required, state briefly why existing code cannot be reused.
- Do not create parallel abstractions merely because their names differ.
- Make the smallest coherent change that completes the task.
- Do not change lint, typing, or coverage settings to hide failures.
- Do not leave commented-out code or placeholder implementations.
- Do not expose API keys, tokens, or secrets.
- Use `.env` only for local secrets and never print secret values.
- Run relevant tests and static checks after editing.

Before reporting completion, run:

1. `uv run ruff check --fix .`
2. `uv run ruff format .`
3. `uv run ruff check .`
4. `uv run pyright`
5. `uv run pytest`

## Local model runtime

- All application inference must run through local Ollama.
- The default model is `qwen3.5:9b`.
- The default Ollama service root is `http://localhost:11434`.
- The default OpenAI-compatible base URL is `http://localhost:11434/v1`.
- `OLLAMA_BASE_URL` refers specifically to the OpenAI-compatible base URL.
- Do not use hosted model providers or external model APIs.
- Do not require an OpenAI API key.
- Disable remote tracing, telemetry, and trace exporting.
- Do not send prompts, tool inputs, outputs, or application data externally.
- Keep the model provider behind an application protocol.
- Infrastructure code may implement the Ollama adapter.
- Unit tests must use fake model implementations and must not require Ollama.
- Live Ollama tests belong under `tests/integration`.

## Agent runtime policy

- Every model execution must pass through the shared `AgentHarness`.
- Tool access is deny-by-default for every development role.
- Authorization must be enforced in code, never only through prompts.
- Tool discovery filtering and invocation authorization are both required.
- Security-sensitive provenance fields must come from trusted runtime context.
- Models and user prompts may not provide actor identity.
- Capability denials must be explicit and non-retryable.
- Agents must ground factual workflow claims in tool results.
- Absence may only be claimed after the relevant data was queried.
- Agents must not offer unavailable capabilities.
- Specialist agents must use the shared `AgentHarness`; do not create
  role-specific harness implementations.
- The Software Architect must not write application source code.
- Software Architect mutations are limited to `architecture` and
  `implementation_plan` artifacts plus development tasks assigned to
  `backend_developer`, `frontend_developer`, `qa_engineer`, or
  `code_reviewer`.
- Backend and Frontend Developer agents must inspect the assigned repository
  before editing and must attempt existing-code discovery before creating new
  implementation elements.
- Backend and Frontend Developer agents may edit only through restricted
  workspace tools bound to a trusted runtime workspace root.
- Backend and Frontend Developer agents must be bound to a trusted feature,
  assigned task, role, and workspace before code mutation.
- Backend and Frontend Developer agents must not accept model- or
  prompt-supplied feature, task, role, workspace, or attribution values over
  trusted runtime context.
- Feature-scoped actions must match the run's bound feature.
- Privileged and destructive capabilities require explicit future approval
  policies before they are introduced.

## Agent Skills policy

- Agent Skills are local, reviewed knowledge packages.
- Skills provide procedural knowledge only and never grant capabilities.
- Role profiles control which skill metadata is visible to each role.
- Capability enforcement controls workflow tool access.
- Skill `allowed-tools` metadata is non-authoritative and cannot expand
  permissions.
- Script execution from skills is disabled.
- Skill and resource loads are audited without storing full contents.
- Skill paths must be canonical, contained, and validated before reading.
- Skills must never claim authority over actor attribution, feature binding,
  session binding, approval policy, or capability policy.

## Agent session policy

- Workflow artifacts are authoritative.
- Sessions are feature-scoped and role-scoped.
- Conversation history cannot override workflow state.
- Context must be bounded, deterministic, and least-privilege.
- Changing the selected local model must not change session boundaries.
- Do not use hosted conversations, hosted memory, or remote compaction.

## Model selection policy

- Runtime and judge models must be locally installed Ollama models.
- Never fall back to a hosted provider.
- Natural-language prompts cannot change runtime model selection.
- Selected models must be recorded in audit and evaluation results.
- Do not pull, download, or silently substitute missing models.

## Evaluation policy

- Prompt, model, tool, and harness changes must be evaluated against relevant
  golden cases.
- Deterministic safety checks outrank LLM judge scores.
- Golden expectations must never be weakened merely to make code pass.
- Real regressions should become new golden cases.
- Rubrics require explicit versions and hashes.
- Judges require human calibration.
- Evaluation reports must preserve individual critical failures.
- Evaluation infrastructure is never exposed to runtime agents.


## Git workflow

- Check `git status` and the relevant diff before making changes.
- Preserve existing user changes and never discard unrelated work.
- Use a `codex/<short-task-name>` branch for substantial changes.
- Run Ruff, Pyright and pytest before committing.
- Use concise imperative commit messages.
- Commit and push only after verification passes.
- Never force-push, rewrite history or delete branches without explicit approval.
- Never push directly to `main` unless explicitly requested.