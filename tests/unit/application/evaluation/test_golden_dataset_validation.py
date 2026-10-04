"""Validate golden contracts without loosening expected tools or context."""

from dataclasses import replace

import pytest

from agent_team.application.evaluation.golden_dataset_loader import (
    GoldenDatasetLoader,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_case_intent import EvalCaseIntent
from agent_team.domain.evaluation.eval_context_policy import EvalContextPolicy
from agent_team.domain.evaluation.eval_error_stage import EvalErrorStage
from agent_team.domain.evaluation.expected_error import ExpectedError
from agent_team.domain.evaluation.expected_tool_call import ExpectedToolCall
from agent_team.domain.evaluation.expected_tool_trajectory import (
    ExpectedToolTrajectory,
)


class TestGoldenDatasetValidation:
    """Reject malformed or contradictory expected tool contracts."""

    @pytest.mark.parametrize(
        ("changes", "message"),
        (
            ({"id": "  "}, "ID must not be blank"),
            ({"user_input": " "}, "blank user input"),
            (
                {
                    "expected_error": ExpectedError(
                        " ", EvalErrorStage.CANDIDATE_EXECUTION
                    )
                },
                "blank error type",
            ),
            ({"max_output_tokens": 0}, "must be positive"),
            (
                {
                    "acceptable_tool_trajectories": (
                        ExpectedToolTrajectory(
                            (), optional_read_only_tool_calls=("read_file",)
                        ),
                    )
                },
                "only optional tool calls",
            ),
            (
                {
                    "acceptable_tool_trajectories": (
                        ExpectedToolTrajectory(
                            (ExpectedToolCall("get_feature", {}),),
                            optional_read_only_tool_calls=("apply_patch",),
                        ),
                    )
                },
                "must be read-only",
            ),
            (
                {
                    "intent": EvalCaseIntent.TOOL_DISPATCH,
                    "expected_tool_calls": (),
                },
                "no required tool call",
            ),
            (
                {
                    "acceptable_tool_trajectories": (
                        ExpectedToolTrajectory(
                            (ExpectedToolCall("get_feature", {}),),
                            forbidden_tool_calls=("unknown_tool",),
                        ),
                    )
                },
                "unknown_tool",
            ),
        ),
    )
    def test_rejects_invalid_golden_contracts(
        self,
        golden_validation_case: EvalCase,
        changes: dict[str, object],
        message: str,
    ) -> None:
        """Fail before evaluating an invalid expected capability contract."""
        case = replace(golden_validation_case, **changes)

        with pytest.raises(ValueError, match=message):
            GoldenDatasetLoader().build_suite("boundary", "hash", (case,))

    @pytest.mark.parametrize(
        ("changes", "message"),
        (
            (
                {"context_policy": EvalContextPolicy.NO_FEATURE_PRELOAD},
                "without preloaded feature data",
            ),
            (
                {"objective_response_facts": ("unavailable fact",)},
                "objective facts are not preloaded",
            ),
            (
                {
                    "feature_scope_id": 999,
                    "objective_response_facts": ("Logout",),
                },
                "objective facts are not preloaded",
            ),
            (
                {
                    "feature_fixtures": (),
                    "objective_response_facts": ("Logout",),
                },
                "objective facts are not preloaded",
            ),
        ),
    )
    def test_context_only_requires_actual_scoped_facts(
        self,
        golden_validation_case: EvalCase,
        changes: dict[str, object],
        message: str,
    ) -> None:
        """Absence of scoped evidence cannot authorize context-only claims."""
        case = replace(
            golden_validation_case,
            intent=EvalCaseIntent.OUTCOME_GROUNDING,
            acceptable_tool_trajectories=(ExpectedToolTrajectory(()),),
        )
        case = replace(case, **changes)

        with pytest.raises(ValueError, match=message):
            GoldenDatasetLoader().build_suite("boundary", "hash", (case,))
