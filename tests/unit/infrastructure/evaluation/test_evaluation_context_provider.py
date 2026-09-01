"""Tests for evaluation-only context policies."""

from agent_team.domain.evaluation.eval_context_policy import EvalContextPolicy
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.infrastructure.evaluation.evaluation_context_provider import (
    EvaluationContextProvider,
)
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


class TestEvaluationContextProvider:
    """EvaluationContextProvider behavior tests."""

    def test_standard_policy_preserves_production_context(
        self,
    ) -> None:
        """Delegate standard evaluation context to FeatureContextBuilder."""
        repository = FakeWorkflowRepository()
        feature = repository.create_feature(
            title="Notifications",
            description="User alerts.",
            status=FeatureStatus.ANALYSIS,
        )
        repository.add_artifact(
            feature_id=feature.id,
            kind=ArtifactKind.ACCEPTANCE_CRITERIA,
            content="Given unread alerts, show a badge.",
            created_by="agent:business_analyst",
        )
        repository.create_task(
            feature_id=feature.id,
            title="Build alert worker",
            description="Deliver alert updates.",
            assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
            status=TaskStatus.PENDING,
        )

        context = EvaluationContextProvider(
            repository=repository,
            context_policy=EvalContextPolicy.STANDARD_FEATURE_CONTEXT,
        ).build_context(
            feature_id=feature.id,
            role=DevelopmentRole.BUSINESS_ANALYST,
            session_id="session-1",
        )

        assert "Notifications" in context.authoritative_context
        assert "Given unread alerts, show a badge." in (
            context.authoritative_context
        )
        assert "Build alert worker" not in context.authoritative_context

    def test_metadata_only_policy_hides_artifact_and_task_data(self) -> None:
        """Expose feature metadata without preloading retrieval targets."""
        repository = FakeWorkflowRepository()
        feature = repository.create_feature(
            title="Search",
            description="Search catalog.",
            status=FeatureStatus.ANALYSIS,
        )
        repository.add_artifact(
            feature_id=feature.id,
            kind=ArtifactKind.REQUIREMENTS,
            content="Search by keyword.",
            created_by="agent:business_analyst",
        )
        repository.create_task(
            feature_id=feature.id,
            title="Index catalog",
            description="Create search index.",
            assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
            status=TaskStatus.PENDING,
        )

        context = EvaluationContextProvider(
            repository=repository,
            context_policy=EvalContextPolicy.METADATA_ONLY_FEATURE_CONTEXT,
        ).build_context(
            feature_id=feature.id,
            role=DevelopmentRole.BUSINESS_ANALYST,
            session_id="session-1",
        )

        assert "Feature title: Search" in context.authoritative_context
        assert "Feature description: Search catalog." in (
            context.authoritative_context
        )
        assert "Search by keyword." not in context.authoritative_context
        assert "Index catalog" not in context.authoritative_context
        assert "Artifact contents: not preloaded" in (
            context.authoritative_context
        )

    def test_no_preload_policy_hides_feature_workflow_data(self) -> None:
        """Provide only scoped identity for tool-dispatch cases."""
        repository = FakeWorkflowRepository()
        feature = repository.create_feature(
            title="Billing",
            description="Invoices and receipts.",
            status=FeatureStatus.ANALYSIS,
        )

        context = EvaluationContextProvider(
            repository=repository,
            context_policy=EvalContextPolicy.NO_FEATURE_PRELOAD,
        ).build_context(
            feature_id=feature.id,
            role=DevelopmentRole.BUSINESS_ANALYST,
            session_id="session-1",
        )

        assert "Feature scope ID: 1" in context.authoritative_context
        assert "Billing" not in context.authoritative_context
        assert "Invoices and receipts." not in context.authoritative_context
        assert "Use workflow tools for feature metadata" in (
            context.authoritative_context
        )
