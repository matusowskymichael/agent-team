"""Pure evaluation fixtures for golden contract boundary tests."""

import pytest

from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_feature_fixture import (
    EvalFeatureFixture,
)
from agent_team.domain.evaluation.eval_suite import EvalSuite
from agent_team.domain.evaluation.expected_tool_call import ExpectedToolCall
from agent_team.domain.evaluation.rubric import Rubric
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.feature_status import FeatureStatus


@pytest.fixture
def eval_safety_suite(golden_validation_case: EvalCase) -> EvalSuite:
    """Provide one pure case for watchdog and interruption regressions."""
    return EvalSuite(
        id="watchdog-suite",
        cases=(golden_validation_case,),
        dataset_hash="reviewed-dataset",
    )


@pytest.fixture
def eval_safety_rubric() -> Rubric:
    """Provide an unused judge rubric for deterministic safety tests."""
    return Rubric(
        id="backend_developer_workflow",
        version="1",
        threshold=0.9,
        dimensions=(),
        content_hash="reviewed-rubric",
        source_text="",
    )


@pytest.fixture
def golden_validation_case() -> EvalCase:
    """Provide a minimal local golden case without filesystem or SDK usage."""
    return EvalCase(
        id="boundary-case",
        name="Boundary case",
        category="grounding",
        severity="critical",
        active_role=DevelopmentRole.BACKEND_DEVELOPER,
        feature_fixtures=(
            EvalFeatureFixture(
                id=1,
                title="Logout",
                description="Implement logout behavior.",
                status=FeatureStatus.DRAFT,
                artifacts=(),
                tasks=(),
            ),
        ),
        prior_session_turns=(),
        user_input="Inspect the assigned feature.",
        expected_tool_calls=(
            ExpectedToolCall("get_feature", {"feature_id": 1}),
        ),
        forbidden_tool_calls=(),
        expected_database_effects=(),
        forbidden_database_effects=(),
        required_response_facts=(),
        forbidden_response_claims=(),
        rubric_id="backend_developer_workflow",
        note="Boundary validation.",
    )
