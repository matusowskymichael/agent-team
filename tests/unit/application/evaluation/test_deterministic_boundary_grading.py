"""Preserve exact tool sequencing and model/error hard gates."""

from dataclasses import replace

import pytest

from agent_team.application.evaluation.deterministic_eval_grader import (
    DeterministicEvalGrader,
)
from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_error_stage import EvalErrorStage
from agent_team.domain.evaluation.expected_error import ExpectedError
from agent_team.domain.evaluation.expected_tool_call import ExpectedToolCall
from agent_team.domain.evaluation.expected_tool_trajectory import (
    ExpectedToolTrajectory,
)
from agent_team.domain.evaluation.observed_tool_call import ObservedToolCall


class TestDeterministicBoundaryGrading:
    """Require objective evidence for developer workflow claims."""

    @pytest.mark.parametrize(
        ("names", "symbol", "passed"),
        (
            (
                ("read_file", "find_symbol", "apply_patch"),
                "AuthService.logout",
                True,
            ),
            (("read_file", "find_symbol"), "AuthService.logout", False),
            (("find_symbol", "apply_patch"), "AuthService", False),
            (("apply_patch", "find_symbol"), "AuthService.logout", False),
            ((), "AuthService.logout", False),
        ),
    )
    def test_ordered_developer_operations_require_exact_symbols(
        self,
        golden_validation_case: EvalCase,
        names: tuple[str, ...],
        symbol: str,
        passed: bool,
    ) -> None:
        """Optional reads cannot conceal missing or out-of-order operations."""
        case = replace(
            golden_validation_case,
            expected_tool_calls=(),
            acceptable_tool_trajectories=(
                ExpectedToolTrajectory(
                    required_tool_calls=(
                        ExpectedToolCall(
                            "find_symbol", {"name": "AuthService.logout"}
                        ),
                        ExpectedToolCall("apply_patch", {}),
                    ),
                    order_matters=True,
                    optional_read_only_tool_calls=("read_file",),
                ),
            ),
        )
        candidate = CandidateRunResult(
            role=case.active_role,
            model="qwen3.5:9b",
            final_response="",
            tool_calls=tuple(
                ObservedToolCall(name, {"name": symbol}, "completed")
                for name in names
            ),
            database_effects=(),
        )

        grade = DeterministicEvalGrader().grade(
            case, candidate, candidate.model
        )

        assert grade.passed is passed
        if not passed:
            assert "missing acceptable tool trajectory" in grade.reasons

    def test_selected_model_mismatch_is_a_hard_failure(
        self, golden_validation_case: EvalCase
    ) -> None:
        """Natural-language success cannot override the recorded model."""
        case = golden_validation_case
        candidate = CandidateRunResult(
            role=case.active_role,
            model="different-local-model",
            final_response="Task completed.",
            tool_calls=(
                ObservedToolCall(
                    "get_feature", {"feature_id": 1}, "completed"
                ),
            ),
            database_effects=(),
        )

        grade = DeterministicEvalGrader().grade(case, candidate, "qwen3.5:9b")

        assert not grade.passed
        assert grade.hard_gate_failed
        assert "selected model did not match runtime config" in grade.reasons

    def test_expected_stall_requires_the_correct_failure_stage(
        self, golden_validation_case: EvalCase
    ) -> None:
        """A setup failure cannot masquerade as an expected runtime stall."""
        case = replace(
            golden_validation_case,
            expected_tool_calls=(),
            expected_error=ExpectedError(
                "AgentStalledError", EvalErrorStage.CANDIDATE_EXECUTION
            ),
        )
        candidate = CandidateRunResult(
            role=case.active_role,
            model="qwen3.5:9b",
            final_response="",
            tool_calls=(),
            database_effects=(),
            status="failed",
            error_type="AgentStalledError",
            error_stage="infrastructure_setup",
        )

        grade = DeterministicEvalGrader().grade(
            case, candidate, candidate.model
        )

        assert not grade.passed
        assert grade.hard_gate_failed
        assert "wrong error stage infrastructure_setup" in grade.reasons
