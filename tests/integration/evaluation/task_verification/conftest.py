"""Golden case fixtures for runner-only workspace verification."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.infrastructure.evaluation.jsonl_golden_dataset_loader import (
    JsonlGoldenDatasetLoader,
)


@pytest.fixture
def evaluation_case(request: pytest.FixtureRequest) -> EvalCase:
    """Load a real development case, retaining its expected trajectory."""
    case_id = str(request.param)
    role = "backend" if case_id.startswith("bd-") else "frontend"
    path = Path(f"evals/datasets/{role}_developer_development.jsonl")
    suite = JsonlGoldenDatasetLoader().load(role, path)
    return next(case for case in suite.cases if case.id == case_id)


@pytest.fixture
def verification_task(evaluation_case: EvalCase) -> DevelopmentTask:
    """Bind verification to the selected case's developer role."""
    now = datetime.now(UTC)
    return DevelopmentTask(
        id=1,
        feature_id=1,
        title="Task",
        description="Description.",
        assigned_role=evaluation_case.active_role,
        status=TaskStatus.VERIFICATION_PENDING,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def submitted_handoff(evaluation_case: EvalCase) -> TaskHandoff:
    """Report expected changed paths without claiming behavior passed."""
    role = evaluation_case.active_role
    paths = tuple(
        str(call.arguments_subset["path"])
        for call in evaluation_case.expected_tool_calls
        if call.name == "apply_patch"
    )
    return TaskHandoff(
        id=1,
        task_id=1,
        agent_run_id=1,
        submitted_by=role,
        attribution=f"agent:{role.value}",
        workspace_identity_hash="workspace-hash",
        implementation_summary="Submitted implementation.",
        changed_paths=paths,
        reused_symbols=(),
        new_symbols=(),
        reuse_notes="Used existing code where relevant.",
        checks_attempted=(role.value.partition("_")[0],),
        limitations="none",
        next_action="verify",
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def seeded_workspace(evaluation_case: EvalCase, tmp_path: Path) -> Path:
    """Write only candidate-visible initial repository files."""
    for file in evaluation_case.workspace_files:
        target = tmp_path / file.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file.content, encoding="utf-8")
    return tmp_path
