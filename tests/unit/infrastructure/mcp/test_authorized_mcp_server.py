"""Tests for authorized MCP server enforcement."""

import asyncio

import pytest
from mcp.types import TextContent

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.runtime.capability_authorizer import (
    CapabilityAuthorizer,
)
from agent_team.domain.audit.tool_invocation_status import ToolInvocationStatus
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_run_limits import AgentRunLimits
from agent_team.domain.runtime.capability_denied_error import (
    CapabilityDeniedError,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_server_config as workflow_mcp_config,
)
from agent_team.infrastructure.mcp.client.authorized_mcp_server import (
    AuthorizedMCPServer,
)
from tests.reporting.allure_steps import report_step
from tests.unit.fakes.audit.fake_agent_audit_repository import (
    FakeAgentAuditRepository,
)
from tests.unit.fakes.mcp.fake_mcp_server import FakeMCPServer
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


class TestAuthorizedMCPServer:
    """Authorized MCP server behavior tests."""

    @pytest.mark.parametrize(
        ("role", "expected_tool_names"),
        [
            (
                DevelopmentRole.DELIVERY_MANAGER,
                {
                    "create_feature",
                    "get_feature",
                    "get_feature_overview",
                    "list_features",
                    "add_artifact",
                    "list_artifacts",
                    "create_task",
                    "list_tasks",
                    "update_task_status",
                },
            ),
            (
                DevelopmentRole.BUSINESS_ANALYST,
                {
                    "get_feature",
                    "get_feature_overview",
                    "list_features",
                    "list_artifacts",
                    "list_tasks",
                    "add_artifact",
                },
            ),
            (
                DevelopmentRole.SOFTWARE_ARCHITECT,
                {
                    "get_feature",
                    "get_feature_overview",
                    "list_artifacts",
                    "list_tasks",
                    "add_artifact",
                    "create_task",
                },
            ),
            (
                DevelopmentRole.BACKEND_DEVELOPER,
                {
                    "get_feature",
                    "get_feature_overview",
                    "list_features",
                    "list_artifacts",
                    "list_tasks",
                    "update_task_status",
                },
            ),
            (
                DevelopmentRole.FRONTEND_DEVELOPER,
                {
                    "get_feature",
                    "get_feature_overview",
                    "list_features",
                    "list_artifacts",
                    "list_tasks",
                    "update_task_status",
                },
            ),
            (
                DevelopmentRole.QA_ENGINEER,
                {
                    "get_feature",
                    "get_feature_overview",
                    "list_features",
                    "list_artifacts",
                    "list_tasks",
                    "add_artifact",
                    "update_task_status",
                },
            ),
            (
                DevelopmentRole.CODE_REVIEWER,
                {
                    "get_feature",
                    "get_feature_overview",
                    "list_features",
                    "list_artifacts",
                    "list_tasks",
                    "add_artifact",
                    "update_task_status",
                },
            ),
        ],
    )
    def test_each_role_sees_only_allowed_tools(
        self,
        role: DevelopmentRole,
        expected_tool_names: set[str],
    ) -> None:
        """Expose only role-allowed MCP tools."""
        server = _authorized_server(role)

        tools = asyncio.run(server.list_tools())

        assert {tool.name for tool in tools} == expected_tool_names

    def test_add_artifact_schema_hides_created_by(self) -> None:
        """Remove trusted actor fields from the model-visible schema."""
        server = _authorized_server(DevelopmentRole.BUSINESS_ANALYST)

        tools = asyncio.run(server.list_tools())
        add_artifact = next(
            tool for tool in tools if tool.name == "add_artifact"
        )
        properties = add_artifact.input_schema["properties"]
        required = add_artifact.input_schema["required"]

        assert isinstance(properties, dict)
        assert isinstance(required, list)
        assert "created_by" not in properties
        assert "created_by" not in required

    def test_business_analyst_cannot_create_feature(self) -> None:
        """Deny feature creation for business analysts."""
        with report_step("Arrange the business analyst capability profile"):
            server = _authorized_server(DevelopmentRole.BUSINESS_ANALYST)

        with report_step("Attempt a denied feature mutation"):
            denial = _denial_text(
                server,
                "create_feature",
                {
                    "title": "Denied",
                    "description": "Should not reach MCP.",
                },
            )

        with report_step("Verify denial audit and zero delegated MCP calls"):
            assert _fake_delegate(server).call_count == 0
            invocations = list(_fake_audit(server).tool_invocations.values())
            assert len(invocations) == 1
            assert invocations[0].status is ToolInvocationStatus.DENIED
            assert invocations[0].tool_name == "create_feature"
            assert "CAPABILITY_DENIED:" in denial
            assert "cannot call create_feature" in denial

    def test_business_analyst_can_add_requirements_artifact(self) -> None:
        """Allow business analysts to add requirements artifacts."""
        server = _authorized_server(DevelopmentRole.BUSINESS_ANALYST)

        asyncio.run(
            server.call_tool(
                "add_artifact",
                {
                    "feature_id": 1,
                    "kind": "requirements",
                    "content": "Users need secure login.",
                },
            ),
        )

        assert _fake_delegate(server).call_count == 1
        received_arguments = _fake_delegate(server).received_calls[0][1]
        assert received_arguments is not None
        assert received_arguments["created_by"] == "agent:business_analyst"
        invocations = list(_fake_audit(server).tool_invocations.values())
        assert len(invocations) == 1
        assert invocations[0].status is ToolInvocationStatus.COMPLETED
        assert invocations[0].run_id == 1

    def test_architect_artifact_uses_trusted_actor(self) -> None:
        """Inject software architect identity for delegated artifacts."""
        server = _authorized_server(DevelopmentRole.SOFTWARE_ARCHITECT)

        asyncio.run(
            server.call_tool(
                "add_artifact",
                {
                    "feature_id": 1,
                    "kind": "architecture",
                    "content": "Layered architecture.",
                },
            ),
        )

        received_arguments = _fake_delegate(server).received_calls[0][1]
        assert received_arguments is not None
        assert received_arguments["created_by"] == ("agent:software_architect")

    def test_create_task_audit_records_effective_default_status(self) -> None:
        """Audit delegated task arguments after runtime normalization."""
        server = _authorized_server(
            DevelopmentRole.SOFTWARE_ARCHITECT,
            bound_feature_id=1,
        )

        asyncio.run(
            server.call_tool(
                "create_task",
                {
                    "feature_id": 1,
                    "title": "Build API",
                    "description": "Implement the API.",
                    "assigned_role": "backend_developer",
                },
            ),
        )

        received_arguments = _fake_delegate(server).received_calls[0][1]
        assert received_arguments is not None
        assert received_arguments["status"] == TaskStatus.PENDING.value
        invocation = next(
            iter(_fake_audit(server).tool_invocations.values()),
        )
        assert '"status":"pending"' in invocation.arguments_preview_json
        assert '"assigned_role":"backend_developer"' in (
            invocation.arguments_preview_json
        )
        assert '"description_hash"' in invocation.arguments_preview_json
        assert "Implement the API." not in invocation.arguments_preview_json

    def test_business_analyst_cannot_add_architecture_artifact(self) -> None:
        """Deny architecture artifacts for business analysts."""
        server = _authorized_server(DevelopmentRole.BUSINESS_ANALYST)

        denial = _denial_text(
            server,
            "add_artifact",
            {
                "feature_id": 1,
                "kind": "architecture",
                "content": "Layered architecture.",
            },
        )

        assert _fake_delegate(server).call_count == 0
        assert (
            "CAPABILITY_DENIED: The business_analyst role cannot create "
            "architecture artifacts."
        ) in denial
        assert "Do not retry this action with different arguments." in denial
        assert "Error invoking MCP tool" not in denial
        assert "format" not in denial.lower()
        assert "transient" not in denial.lower()

    def test_architect_can_create_tasks_but_not_features(self) -> None:
        """Allow architect tasks and deny architect feature creation."""
        allowed_server = _authorized_server(DevelopmentRole.SOFTWARE_ARCHITECT)

        asyncio.run(
            allowed_server.call_tool(
                "create_task",
                {
                    "feature_id": 1,
                    "title": "Design API",
                    "description": "Define service boundaries.",
                    "assigned_role": "backend_developer",
                },
            ),
        )
        denied_server = _authorized_server(DevelopmentRole.SOFTWARE_ARCHITECT)
        _denial_text(
            denied_server,
            "create_feature",
            {
                "title": "Denied",
                "description": "Architect cannot create features.",
            },
        )

        assert _fake_delegate(allowed_server).call_count == 1
        assert _fake_delegate(denied_server).call_count == 0

    @pytest.mark.parametrize(
        "assigned_role",
        [
            DevelopmentRole.BACKEND_DEVELOPER,
            DevelopmentRole.FRONTEND_DEVELOPER,
            DevelopmentRole.QA_ENGINEER,
            DevelopmentRole.CODE_REVIEWER,
        ],
    )
    def test_architect_can_create_delivery_tasks(
        self,
        assigned_role: DevelopmentRole,
    ) -> None:
        """Allow architect-created tasks only for delivery specialists."""
        server = _authorized_server(
            DevelopmentRole.SOFTWARE_ARCHITECT,
            bound_feature_id=1,
        )

        asyncio.run(
            server.call_tool(
                "create_task",
                {
                    "feature_id": 1,
                    "title": "Implement slice",
                    "description": "Build the scoped deliverable.",
                    "assigned_role": assigned_role.value,
                },
            ),
        )

        assert _fake_delegate(server).call_count == 1
        received_arguments = _fake_delegate(server).received_calls[0][1]
        assert received_arguments is not None
        assert received_arguments["assigned_role"] == assigned_role.value

    @pytest.mark.parametrize(
        "assigned_role",
        [
            DevelopmentRole.BUSINESS_ANALYST,
            DevelopmentRole.SOFTWARE_ARCHITECT,
            DevelopmentRole.DELIVERY_MANAGER,
        ],
    )
    def test_architect_cannot_create_tasks_for_prohibited_roles(
        self,
        assigned_role: DevelopmentRole,
    ) -> None:
        """Deny architect task assignment to planning roles."""
        server = _authorized_server(DevelopmentRole.SOFTWARE_ARCHITECT)

        denial = _denial_text(
            server,
            "create_task",
            {
                "feature_id": 1,
                "title": "Invalid task",
                "description": "Should not reach MCP.",
                "assigned_role": assigned_role.value,
            },
        )

        assert _fake_delegate(server).call_count == 0
        assert "cannot assign tasks" in denial

    def test_architect_task_creation_requires_default_status(self) -> None:
        """Deny architect task creation with non-default status."""
        server = _authorized_server(DevelopmentRole.SOFTWARE_ARCHITECT)

        denial = _denial_text(
            server,
            "create_task",
            {
                "feature_id": 1,
                "title": "Started task",
                "description": "Should not start immediately.",
                "assigned_role": "backend_developer",
                "status": "in_progress",
            },
        )

        assert _fake_delegate(server).call_count == 0
        assert "default pending status" in denial

    def test_architect_task_creation_denies_unknown_arguments(self) -> None:
        """Deny unsupported architect task arguments before MCP."""
        server = _authorized_server(DevelopmentRole.SOFTWARE_ARCHITECT)

        denial = _denial_text(
            server,
            "create_task",
            {
                "feature_id": 1,
                "title": "Task",
                "description": "Task details.",
                "assigned_role": "backend_developer",
                "created_by": "agent:software_architect",
            },
        )

        assert _fake_delegate(server).call_count == 0
        assert "unsupported arguments: created_by" in denial

    def test_bound_feature_mismatch_is_denied_before_mcp(self) -> None:
        """Deny feature-scoped tool calls outside the bound feature."""
        server = _authorized_server(
            DevelopmentRole.SOFTWARE_ARCHITECT,
            bound_feature_id=1,
        )

        denial = _denial_text(
            server,
            "get_feature_overview",
            {"feature_id": 2},
        )

        assert _fake_delegate(server).call_count == 0
        assert "bound to feature 1" in denial
        invocations = list(_fake_audit(server).tool_invocations.values())
        assert invocations[0].status is ToolInvocationStatus.DENIED

    def test_architect_cannot_list_all_features(self) -> None:
        """Deny broad feature discovery for feature-bound architects."""
        server = _authorized_server(DevelopmentRole.SOFTWARE_ARCHITECT)

        denial = _denial_text(server, "list_features", {})

        assert _fake_delegate(server).call_count == 0
        assert "cannot call list_features" in denial

    def test_developer_cannot_update_task_assigned_to_other_role(self) -> None:
        """Deny task updates when assignment does not match the role."""
        repository = FakeWorkflowRepository()
        feature = repository.create_feature(
            title="User Authentication",
            description="Secure login and logout.",
            status=FeatureStatus.DRAFT,
        )
        task = repository.create_task(
            feature_id=feature.id,
            title="Build login UI",
            description="Create frontend flow.",
            assigned_role=DevelopmentRole.FRONTEND_DEVELOPER,
            status=TaskStatus.PENDING,
        )
        server = _authorized_server(
            DevelopmentRole.BACKEND_DEVELOPER,
            repository=repository,
        )

        _denial_text(
            server,
            "update_task_status",
            {"task_id": task.id, "status": "in_progress"},
        )

        assert _fake_delegate(server).call_count == 0

    def test_qa_cannot_add_code_review_artifact(self) -> None:
        """Deny code-review artifacts for QA."""
        server = _authorized_server(DevelopmentRole.QA_ENGINEER)

        _denial_text(
            server,
            "add_artifact",
            {
                "feature_id": 1,
                "kind": "code_review",
                "content": "Looks good.",
            },
        )

        assert _fake_delegate(server).call_count == 0

    def test_reviewer_cannot_add_test_report_artifact(self) -> None:
        """Deny test-report artifacts for code reviewers."""
        server = _authorized_server(DevelopmentRole.CODE_REVIEWER)

        _denial_text(
            server,
            "add_artifact",
            {
                "feature_id": 1,
                "kind": "test_report",
                "content": "All tests passed.",
            },
        )

        assert _fake_delegate(server).call_count == 0

    def test_unknown_tools_are_denied(self) -> None:
        """Deny unknown tools before they reach MCP."""
        server = _authorized_server(DevelopmentRole.DELIVERY_MANAGER)

        _denial_text(server, "delete_feature", {"feature_id": 1})

        assert _fake_delegate(server).call_count == 0

    def test_malformed_arguments_fail_closed(self) -> None:
        """Deny malformed authorization arguments before MCP."""
        server = _authorized_server(DevelopmentRole.BUSINESS_ANALYST)

        _denial_text(
            server,
            "add_artifact",
            {
                "feature_id": "not-an-integer",
                "kind": "requirements",
            },
        )

        assert _fake_delegate(server).call_count == 0

    def test_delivery_manager_can_add_any_known_artifact(self) -> None:
        """Allow delivery managers to add known artifact kinds."""
        server = _authorized_server(DevelopmentRole.DELIVERY_MANAGER)

        asyncio.run(
            server.call_tool(
                "add_artifact",
                {
                    "feature_id": 1,
                    "kind": "architecture",
                    "content": "Use layered architecture.",
                },
            ),
        )

        assert _fake_delegate(server).call_count == 1

    def test_delivery_manager_can_update_any_task(self) -> None:
        """Allow delivery managers to update task status without assignment."""
        server = _authorized_server(DevelopmentRole.DELIVERY_MANAGER)

        asyncio.run(
            server.call_tool(
                "update_task_status",
                {"task_id": 999, "status": "blocked"},
            ),
        )

        assert _fake_delegate(server).call_count == 1

    def test_bound_task_status_update_must_match_feature(self) -> None:
        """Deny task-status updates outside a bound feature."""
        repository = FakeWorkflowRepository()
        feature_one = repository.create_feature(
            title="One",
            description="First.",
            status=FeatureStatus.DRAFT,
        )
        feature_two = repository.create_feature(
            title="Two",
            description="Second.",
            status=FeatureStatus.DRAFT,
        )
        task = repository.create_task(
            feature_id=feature_two.id,
            title="Other task",
            description="Outside the run feature.",
            assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
            status=TaskStatus.PENDING,
        )
        server = _authorized_server(
            DevelopmentRole.DELIVERY_MANAGER,
            repository=repository,
            bound_feature_id=feature_one.id,
        )

        denial = _denial_text(
            server,
            "update_task_status",
            {"task_id": task.id, "status": "completed"},
        )

        assert _fake_delegate(server).call_count == 0
        assert "bound to feature 1" in denial

    def test_bound_task_status_update_must_match_task(self) -> None:
        """Deny developer updates outside the trusted task binding."""
        repository = FakeWorkflowRepository()
        feature = repository.create_feature(
            title="One",
            description="First.",
            status=FeatureStatus.DRAFT,
        )
        first_task = repository.create_task(
            feature_id=feature.id,
            title="Assigned task",
            description="Trusted task.",
            assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
            status=TaskStatus.PENDING,
        )
        second_task = repository.create_task(
            feature_id=feature.id,
            title="Other task",
            description="Another backend task.",
            assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
            status=TaskStatus.PENDING,
        )
        server = _authorized_server(
            DevelopmentRole.BACKEND_DEVELOPER,
            repository=repository,
            bound_feature_id=feature.id,
            bound_task_id=first_task.id,
        )

        denial = _denial_text(
            server,
            "update_task_status",
            {"task_id": second_task.id, "status": "completed"},
        )

        assert _fake_delegate(server).call_count == 0
        assert "cannot update task 2" in denial

    def test_caller_supplied_created_by_is_denied(self) -> None:
        """Reject caller-supplied actor identity before MCP delegation."""
        server = _authorized_server(DevelopmentRole.BUSINESS_ANALYST)

        denial = _denial_text(
            server,
            "add_artifact",
            {
                "feature_id": 1,
                "kind": "requirements",
                "content": "Users need secure login.",
                "created_by": "agent:software_architect",
            },
        )

        assert _fake_delegate(server).call_count == 0
        assert "cannot provide created_by" in denial
        assert "trusted runtime context" in denial

    def test_failed_tool_calls_are_recorded(self) -> None:
        """Finalize delegated MCP failures in the audit log."""
        delegate = FakeMCPServer(
            tool_names=[tool.value for tool in WorkflowToolName],
            fail_call=True,
        )
        server = _authorized_server(
            DevelopmentRole.DELIVERY_MANAGER,
            delegate=delegate,
        )

        with pytest.raises(RuntimeError, match="MCP tool failed"):
            asyncio.run(server.call_tool("list_features", {}))

        invocations = list(_fake_audit(server).tool_invocations.values())
        assert delegate.call_count == 1
        assert len(invocations) == 1
        assert invocations[0].status is ToolInvocationStatus.FAILED
        assert invocations[0].error_type == "RuntimeError"

    def test_audit_recording_failure_prevents_mcp_execution(self) -> None:
        """Do not invoke MCP when allowed calls cannot be audited first."""
        audit_repository = FakeAgentAuditRepository(
            fail_start_tool_invocation=True,
        )
        server = _authorized_server(
            DevelopmentRole.DELIVERY_MANAGER,
            audit_repository=audit_repository,
        )

        with pytest.raises(RuntimeError, match="tool audit start failed"):
            asyncio.run(server.call_tool("list_features", {}))

        assert _fake_delegate(server).call_count == 0

    def test_artifact_content_and_secrets_are_not_stored_in_previews(
        self,
    ) -> None:
        """Store artifact metadata and hashes instead of full content."""
        server = _authorized_server(DevelopmentRole.DELIVERY_MANAGER)

        asyncio.run(
            server.call_tool(
                "add_artifact",
                {
                    "feature_id": 1,
                    "kind": "requirements",
                    "content": "Full artifact content password=secret-value",
                    "api_key": "private-key",
                },
            ),
        )

        invocation = next(iter(_fake_audit(server).tool_invocations.values()))
        preview = invocation.arguments_preview_json
        assert "Full artifact content" not in preview
        assert "secret-value" not in preview
        assert "private-key" not in preview
        assert "content_hash" in preview
        assert "content_length" in preview
        assert "created_by" in preview

    def test_valid_feature_status_filter_is_allowed(self) -> None:
        """Allow valid list feature status filters."""
        server = _authorized_server(DevelopmentRole.BUSINESS_ANALYST)

        asyncio.run(server.call_tool("list_features", {"status": "draft"}))

        assert _fake_delegate(server).call_count == 1

    def test_invalid_feature_status_filter_is_denied(self) -> None:
        """Deny invalid list feature status filters."""
        server = _authorized_server(DevelopmentRole.BUSINESS_ANALYST)

        _denial_text(
            server,
            "list_features",
            {"status": "not-a-status"},
        )

        assert _fake_delegate(server).call_count == 0

    def test_missing_arguments_fail_closed(self) -> None:
        """Deny missing authorization arguments before MCP."""
        server = _authorized_server(DevelopmentRole.BUSINESS_ANALYST)

        _denial_text(server, "add_artifact", None)

        assert _fake_delegate(server).call_count == 0

    def test_missing_string_argument_fails_closed(self) -> None:
        """Deny missing string arguments before MCP."""
        server = _authorized_server(DevelopmentRole.BUSINESS_ANALYST)

        _denial_text(
            server,
            "add_artifact",
            {"feature_id": 1},
        )

        assert _fake_delegate(server).call_count == 0

    def test_missing_task_assignment_fails_closed(self) -> None:
        """Deny task updates when assignment cannot be verified."""
        server = _authorized_server(DevelopmentRole.BACKEND_DEVELOPER)

        _denial_text(
            server,
            "update_task_status",
            {"task_id": 404, "status": "completed"},
        )

        assert _fake_delegate(server).call_count == 0

    def test_role_without_assignment_rights_fails_closed(self) -> None:
        """Deny updates when a profile grants an unsupported role a tool."""
        catalog = AgentProfileCatalog()
        architect = catalog.get_profile(DevelopmentRole.SOFTWARE_ARCHITECT)
        profile = AgentProfile(
            role=architect.role,
            instructions=architect.instructions,
            allowed_tools=architect.allowed_tools
            | frozenset({WorkflowToolName.UPDATE_TASK_STATUS}),
            run_limits=architect.run_limits,
        )
        delegate = FakeMCPServer(
            tool_names=[tool.value for tool in WorkflowToolName],
        )
        audit_repository = FakeAgentAuditRepository()
        run = audit_repository.open_run(role=profile.role)
        server = AuthorizedMCPServer(
            delegate=delegate,
            config=workflow_mcp_config.DevelopmentWorkflowMCPServerConfig(
                profile=profile,
                authorizer=CapabilityAuthorizer(
                    repository=FakeWorkflowRepository(),
                ),
                audit_repository=audit_repository,
                run=run,
            ),
        )

        _denial_text(
            server,
            "update_task_status",
            {"task_id": 1, "status": "completed"},
        )

        assert delegate.call_count == 0

    def test_wrapper_delegates_lifecycle_methods(self) -> None:
        """Delegate MCP lifecycle methods to the wrapped server."""
        server = _authorized_server(DevelopmentRole.DELIVERY_MANAGER)
        delegate = _fake_delegate(server)

        assert server.cached_tools is None
        asyncio.run(server.connect())
        asyncio.run(server.cleanup())

        assert delegate.connected is True
        assert delegate.cleaned_up is True

    def test_prompt_methods_delegate_to_wrapped_server(self) -> None:
        """Delegate prompt methods to the wrapped server."""
        server = _authorized_server(DevelopmentRole.DELIVERY_MANAGER)

        with pytest.raises(NotImplementedError):
            asyncio.run(server.list_prompts())
        with pytest.raises(NotImplementedError):
            asyncio.run(server.get_prompt("missing"))

    def test_agent_run_limits_require_positive_turns(self) -> None:
        """Deny invalid run limits."""
        with pytest.raises(ValueError, match="max_turns"):
            AgentRunLimits(max_turns=0)

    def test_empty_profile_catalog_denies_known_roles(self) -> None:
        """Deny roles missing from the configured catalog."""
        catalog = AgentProfileCatalog(profiles={})

        with pytest.raises(CapabilityDeniedError):
            catalog.get_profile(DevelopmentRole.DELIVERY_MANAGER)


def _authorized_server(  # noqa: PLR0913, PLR0917
    role: DevelopmentRole,
    repository: FakeWorkflowRepository | None = None,
    audit_repository: FakeAgentAuditRepository | None = None,
    delegate: FakeMCPServer | None = None,
    bound_feature_id: int | None = None,
    bound_task_id: int | None = None,
) -> AuthorizedMCPServer:
    catalog = AgentProfileCatalog()
    workflow_repository = repository or FakeWorkflowRepository()
    mcp_delegate = delegate or FakeMCPServer(
        tool_names=[tool.value for tool in WorkflowToolName],
    )
    audit = audit_repository or FakeAgentAuditRepository()
    run = audit.open_run(role=role, feature_id=bound_feature_id)
    return AuthorizedMCPServer(
        delegate=mcp_delegate,
        config=workflow_mcp_config.DevelopmentWorkflowMCPServerConfig(
            profile=catalog.get_profile(role),
            authorizer=CapabilityAuthorizer(repository=workflow_repository),
            audit_repository=audit,
            run=run,
            bound_task_id=bound_task_id,
        ),
    )


def _fake_delegate(server: AuthorizedMCPServer) -> FakeMCPServer:
    assert isinstance(server.delegate, FakeMCPServer)
    return server.delegate


def _fake_audit(server: AuthorizedMCPServer) -> FakeAgentAuditRepository:
    assert isinstance(server.audit_repository, FakeAgentAuditRepository)
    return server.audit_repository


def _denial_text(
    server: AuthorizedMCPServer,
    tool_name: str,
    arguments: dict[str, object] | None,
) -> str:
    result = asyncio.run(server.call_tool(tool_name, arguments))
    assert result.is_error is True
    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, TextContent)
    assert content.text.startswith("CAPABILITY_DENIED:")
    assert "Do not retry this action with different arguments." in content.text
    assert "Error invoking MCP tool" not in content.text
    return content.text
