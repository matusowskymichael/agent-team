"""Deterministic invocation and task fixtures for runtime progress tests."""

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from agent_team.application.runtime.agent_task_snapshot import (
    AgentTaskSnapshot,
)
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord
from agent_team.domain.audit.tool_invocation_status import ToolInvocationStatus
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_verification_check import (
    TaskVerificationCheck,
)
from agent_team.domain.workflow.task_verification_evidence import (
    TaskVerificationEvidence,
)
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)

FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)


@pytest.fixture
def progress_invocation() -> ToolInvocationRecord:
    """Provide one completed discovery with deliberately private previews."""
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    return ToolInvocationRecord(
        id=1,
        run_id=1,
        server_name="workspace",
        tool_name="read_file",
        classification=ToolClassification.READ_ONLY,
        status=ToolInvocationStatus.COMPLETED,
        arguments_hash="arguments-hash",
        arguments_preview_json='{"content": "private-arguments"}',
        result_hash="result-hash",
        result_preview=json.dumps({"content": "private-source-content"}),
        started_at=instant,
        ended_at=instant,
        error_type=None,
        error_message=None,
    )


@pytest.fixture
def progress_snapshot() -> AgentTaskSnapshot:
    """Provide an authoritative pending task without preexisting handoff."""
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    return AgentTaskSnapshot(
        task=DevelopmentTask(
            id=4,
            feature_id=2,
            title="Implement behavior.",
            description="Private task description.",
            assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
            status=TaskStatus.PENDING,
            created_at=instant,
            updated_at=instant,
        ),
        handoff=None,
        verification=None,
    )


@pytest.fixture
def progress_verified_snapshot(
    progress_snapshot: AgentTaskSnapshot,
) -> AgentTaskSnapshot:
    """Provide a failed verified handoff for semantic progress comparisons."""
    instant = progress_snapshot.task.created_at
    handoff = TaskHandoff(
        id=1,
        task_id=progress_snapshot.task.id,
        agent_run_id=1,
        submitted_by=progress_snapshot.task.assigned_role,
        attribution="agent:backend_developer",
        workspace_identity_hash="workspace-hash",
        implementation_summary="Implemented the behavior.",
        changed_paths=("src/auth.py",),
        reused_symbols=(),
        new_symbols=(),
        reuse_notes="Reused existing code.",
        checks_attempted=("backend",),
        limitations="None.",
        next_action="Await verification.",
        created_at=instant,
    )
    check = TaskVerificationCheck(
        id=1,
        verification_id=1,
        name="backend",
        started_at=instant,
        ended_at=instant,
        exit_code=1,
        timed_out=False,
        stdout_hash="initial-stdout-hash",
        stdout_excerpt="Ran tests in 0.10 seconds.",
        stderr_hash="initial-stderr-hash",
        stderr_excerpt="",
    )
    return replace(
        progress_snapshot,
        task=replace(progress_snapshot.task, status=TaskStatus.IN_PROGRESS),
        handoff=handoff,
        verification=TaskVerificationEvidence(
            id=1,
            task_id=progress_snapshot.task.id,
            submission_id=handoff.id,
            verifier_name="local",
            outcome=TaskVerificationOutcome.FAILED,
            failure_classification=(FailureClassification.CHECK_FAILED),
            feedback="One assertion failed.",
            checks=(check,),
            started_at=instant,
            ended_at=instant,
        ),
    )
