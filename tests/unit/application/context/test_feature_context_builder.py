"""Tests for authoritative feature context building."""

import pytest

from agent_team.application.context.feature_context_builder import (
    FeatureContextBuilder,
)
from agent_team.domain.context.agent_context_budget_exceeded_error import (
    AgentContextBudgetExceededError,
)
from agent_team.domain.context.agent_context_policy import AgentContextPolicy
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


class TestFeatureContextBuilder:
    """FeatureContextBuilder behavior tests."""

    def test_business_analyst_context_excludes_architecture(self) -> None:
        """Include only analyst-visible artifact kinds."""
        repository = FakeWorkflowRepository()
        feature = repository.create_feature(
            title="Login",
            description="Secure login.",
            status=FeatureStatus.DRAFT,
        )
        repository.add_artifact(
            feature_id=feature.id,
            kind=ArtifactKind.REQUIREMENTS,
            content="Users can log in.",
            created_by="agent:business_analyst",
        )
        repository.add_artifact(
            feature_id=feature.id,
            kind=ArtifactKind.ARCHITECTURE,
            content="Use a layered auth module.",
            created_by="agent:software_architect",
        )

        context = FeatureContextBuilder(repository).build_context(
            feature_id=feature.id,
            role=DevelopmentRole.BUSINESS_ANALYST,
            session_id="session-1",
        )

        assert "Users can log in." in context.authoritative_context
        assert "layered auth module" not in context.authoritative_context
        assert (
            "not requested for this role context policy"
            in context.authoritative_context
        )

    def test_developer_context_includes_only_assigned_tasks(self) -> None:
        """Limit developer task context to the active role."""
        repository = FakeWorkflowRepository()
        feature = repository.create_feature(
            title="Checkout",
            description="Fast checkout.",
            status=FeatureStatus.DRAFT,
        )
        repository.create_task(
            feature_id=feature.id,
            title="Build API",
            description="Backend API.",
            assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
            status=TaskStatus.PENDING,
        )
        repository.create_task(
            feature_id=feature.id,
            title="Build UI",
            description="Frontend UI.",
            assigned_role=DevelopmentRole.FRONTEND_DEVELOPER,
            status=TaskStatus.PENDING,
        )

        context = FeatureContextBuilder(repository).build_context(
            feature_id=feature.id,
            role=DevelopmentRole.BACKEND_DEVELOPER,
            session_id="session-1",
        )

        assert "Build API" in context.authoritative_context
        assert "Build UI" not in context.authoritative_context

    def test_architect_context_includes_design_inputs_and_all_tasks(
        self,
    ) -> None:
        """Provide architect-scoped design facts for the bound feature only."""
        repository = FakeWorkflowRepository()
        feature = repository.create_feature(
            title="Payments",
            description="Collect card payments.",
            status=FeatureStatus.ARCHITECTURE,
        )
        other_feature = repository.create_feature(
            title="Admin",
            description="Admin console.",
            status=FeatureStatus.ANALYSIS,
        )
        repository.add_artifact(
            feature_id=feature.id,
            kind=ArtifactKind.REQUIREMENTS,
            content="Users can pay invoices by card.",
            created_by="agent:business_analyst",
        )
        repository.add_artifact(
            feature_id=feature.id,
            kind=ArtifactKind.ACCEPTANCE_CRITERIA,
            content="Given valid card details, payment succeeds.",
            created_by="agent:business_analyst",
        )
        repository.add_artifact(
            feature_id=feature.id,
            kind=ArtifactKind.ARCHITECTURE,
            content="Use a payment gateway adapter.",
            created_by="agent:software_architect",
        )
        repository.add_artifact(
            feature_id=feature.id,
            kind=ArtifactKind.IMPLEMENTATION_PLAN,
            content="Phase 1: backend contract.",
            created_by="agent:software_architect",
        )
        repository.add_artifact(
            feature_id=feature.id,
            kind=ArtifactKind.TEST_REPORT,
            content="QA-only result.",
            created_by="agent:qa_engineer",
        )
        repository.add_artifact(
            feature_id=other_feature.id,
            kind=ArtifactKind.REQUIREMENTS,
            content="Other feature secret.",
            created_by="agent:business_analyst",
        )
        repository.create_task(
            feature_id=feature.id,
            title="Build gateway adapter",
            description="Backend integration.",
            assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
            status=TaskStatus.PENDING,
        )
        repository.create_task(
            feature_id=feature.id,
            title="Verify payment flow",
            description="QA verification.",
            assigned_role=DevelopmentRole.QA_ENGINEER,
            status=TaskStatus.PENDING,
        )

        context = FeatureContextBuilder(repository).build_context(
            feature_id=feature.id,
            role=DevelopmentRole.SOFTWARE_ARCHITECT,
            session_id="architect-payments",
        )

        text = context.authoritative_context
        assert "Users can pay invoices by card." in text
        assert "Given valid card details, payment succeeds." in text
        assert "Use a payment gateway adapter." in text
        assert "Phase 1: backend contract." in text
        assert "Build gateway adapter" in text
        assert "Verify payment flow" in text
        assert "QA-only result." not in text
        assert "Other feature secret." not in text

    def test_context_refreshes_from_current_workflow_state(self) -> None:
        """Build fresh authoritative context on every run."""
        repository = FakeWorkflowRepository()
        feature = repository.create_feature(
            title="Reports",
            description="Current reports.",
            status=FeatureStatus.DRAFT,
        )
        builder = FeatureContextBuilder(repository)

        first_context = builder.build_context(
            feature_id=feature.id,
            role=DevelopmentRole.BUSINESS_ANALYST,
            session_id="session-1",
        )
        repository.add_artifact(
            feature_id=feature.id,
            kind=ArtifactKind.ACCEPTANCE_CRITERIA,
            content="Reports export as CSV.",
            created_by="agent:business_analyst",
        )
        second_context = builder.build_context(
            feature_id=feature.id,
            role=DevelopmentRole.BUSINESS_ANALYST,
            session_id="session-1",
        )

        assert "Reports export as CSV." not in (
            first_context.authoritative_context
        )
        assert "Reports export as CSV." in (
            second_context.authoritative_context
        )

    def test_oversized_authoritative_context_fails(self) -> None:
        """Raise a typed error instead of truncating workflow artifacts."""
        repository = FakeWorkflowRepository()
        feature = repository.create_feature(
            title="Huge",
            description="Huge feature.",
            status=FeatureStatus.DRAFT,
        )
        repository.add_artifact(
            feature_id=feature.id,
            kind=ArtifactKind.REQUIREMENTS,
            content="x" * 100,
            created_by="agent:business_analyst",
        )
        builder = FeatureContextBuilder(
            repository=repository,
            policies={
                DevelopmentRole.BUSINESS_ANALYST: AgentContextPolicy(
                    artifact_kinds=frozenset({ArtifactKind.REQUIREMENTS}),
                    task_roles=frozenset(),
                    max_authoritative_context_chars=50,
                ),
            },
        )

        with pytest.raises(AgentContextBudgetExceededError):
            builder.build_context(
                feature_id=feature.id,
                role=DevelopmentRole.BUSINESS_ANALYST,
                session_id="session-1",
            )
