"""Developer golden consistency through persisted verification and grading."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from agent_team.application.evaluation.deterministic_eval_grader import (
    DeterministicEvalGrader,
)
from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.database_effect import DatabaseEffect
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.infrastructure.evaluation.jsonl_golden_dataset_loader import (
    JsonlGoldenDatasetLoader,
)

IMPLEMENTATIONS = (
    (
        "bd-dev-002",
        "class AuthService:\n"
        "    def __init__(self):\n"
        "        self.revoked_tokens = set()\n"
        "    def logout(self, token):\n"
        "        if not token:\n"
        "            return False\n"
        "        self.revoked_tokens.add(token)\n"
        "        return True\n",
    ),
    (
        "bd-dev-004",
        "import json\n"
        "class AuditExportFormatter:\n"
        "    def format(self, events):\n"
        "        return json.dumps(events)\n",
    ),
    (
        "fd-dev-002",
        "export function AccountMenu({onLogout}) {\n"
        "  return <button onClick={onLogout}>Logout</button>\n"
        "}\n",
    ),
    (
        "fd-dev-004",
        "export function EmptyState({message}) {\n"
        "  return <p>{message}</p>\n"
        "}\n",
    ),
)


class TestDeveloperGoldenLifecycle:
    """Strict database effects remain achievable and reject missing checks."""

    @pytest.mark.parametrize(
        ("case_id", "wrong_implementation"),
        (
            ("bd-dev-002", "class AuthService:\n    pass\n"),
            ("bd-dev-004", "class AuditExportFormatter:\n    pass\n"),
            (
                "fd-dev-002",
                "export function AccountMenu({onLogout}) { return null }\n",
            ),
            (
                "fd-dev-004",
                "export function EmptyState({message}) { return null }\n",
            ),
        ),
    )
    @pytest.mark.parametrize("no_op", (False, True))
    def test_incorrect_submission_cannot_pass_golden_grading(
        self,
        golden_lifecycle_runner: Callable[[EvalCase, str], CandidateRunResult],
        case_id: str,
        wrong_implementation: str,
        no_op: bool,
    ) -> None:
        """Preserve hard gates after an incorrect or unchanged submission."""
        case = _case(case_id)
        baseline = case.workspace_files[0].content
        candidate = golden_lifecycle_runner(
            case,
            baseline if no_op else wrong_implementation,
        )

        assert candidate.status == "completed", candidate.error_message
        grade = DeterministicEvalGrader().grade(
            case,
            candidate,
            candidate.model,
        )
        assert grade.hard_gate_failed
        verification = next(
            effect
            for effect in candidate.database_effects
            if effect.table == "task_verifications"
        )
        assert verification.field_values["outcome"] == "failed"
        task = next(
            effect
            for effect in candidate.database_effects
            if effect.table == "development_tasks"
        )
        assert task.field_values["status"] == "in_progress"

    @pytest.mark.parametrize(("case_id", "implementation"), IMPLEMENTATIONS)
    def test_correct_submission_passes_strict_golden_grading(
        self,
        golden_lifecycle_runner: Callable[[EvalCase, str], CandidateRunResult],
        case_id: str,
        implementation: str,
    ) -> None:
        """Match every persisted assertion to an explicit expectation."""
        case = _case(case_id)
        candidate = golden_lifecycle_runner(case, implementation)
        grade = DeterministicEvalGrader().grade(
            case, candidate, candidate.model
        )

        assert candidate.status == "completed", candidate.error_message
        assert grade.passed, grade.reasons
        checks = tuple(
            effect
            for effect in candidate.database_effects
            if effect.table == "task_verification_checks"
        )
        assert len(checks) == 3
        for missing in checks:
            incomplete = replace(
                candidate,
                database_effects=tuple(
                    effect
                    for effect in candidate.database_effects
                    if effect != missing
                ),
            )
            assert (
                DeterministicEvalGrader()
                .grade(
                    case,
                    incomplete,
                    candidate.model,
                )
                .hard_gate_failed
            )
        unexpected = replace(
            candidate,
            database_effects=(
                *candidate.database_effects,
                DatabaseEffect(
                    table="task_verification_checks",
                    operation="insert",
                    field_values={"name": "unexpected", "exit_code": 0},
                ),
            ),
        )
        assert (
            DeterministicEvalGrader()
            .grade(
                case,
                unexpected,
                candidate.model,
            )
            .hard_gate_failed
        )

    @pytest.mark.parametrize(
        ("case_id", "implementation"),
        (
            ("bd-dev-008", "def health():\n    return 'healthy'\n"),
            (
                "fd-dev-008",
                "export function StatusBadge() {\n"
                "  return <span role='status'>OK</span>\n}\n",
            ),
        ),
    )
    def test_mutation_case_has_a_valid_activation_trajectory(
        self,
        golden_lifecycle_runner: Callable[[EvalCase, str], CandidateRunResult],
        case_id: str,
        implementation: str,
    ) -> None:
        """Require activation and its observed effect before real mutation."""
        case = _case(case_id)
        candidate = golden_lifecycle_runner(case, implementation)
        grade = DeterministicEvalGrader().grade(
            case, candidate, candidate.model
        )

        assert candidate.status == "completed", candidate.error_message
        assert grade.passed, grade.reasons
        assert len(candidate.database_effects) == 1
        assert candidate.database_effects[0].field_values["status"] == (
            "in_progress"
        )
        incomplete = replace(candidate, database_effects=())
        assert (
            DeterministicEvalGrader()
            .grade(
                case,
                incomplete,
                candidate.model,
            )
            .hard_gate_failed
        )


def _case(case_id: str) -> EvalCase:
    role = "backend" if case_id.startswith("bd-") else "frontend"
    suite = JsonlGoldenDatasetLoader().load(
        f"{role}_developer_development",
        Path(f"evals/datasets/{role}_developer_development.jsonl"),
    )
    return next(case for case in suite.cases if case.id == case_id)
