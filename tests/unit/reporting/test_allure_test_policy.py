"""Tests for centralized Allure test metadata policy."""

from agent_team.domain.runtime.development_role import DevelopmentRole
from tests.reporting.allure_test_policy import (
    behavior_hierarchy,
    metadata_tags,
    owner_for,
    readable_title,
    safe_node_id,
    safe_parameter_value,
    safe_reported_parameter_value,
    severity_for,
    stable_test_id,
    suite_hierarchy,
)


class TestAllureTestPolicy:
    """Allure hierarchy, label, and parameter policy tests."""

    def test_derives_suite_and_behavior_hierarchies(self) -> None:
        """Map a runtime unit test to stable suite and behavior labels."""
        node_id = (
            "tests/unit/application/runtime/test_agent_harness.py::"
            "TestAgentHarness::test_executes"
        )

        assert suite_hierarchy(node_id) == (
            "Unit",
            "Runtime",
            "Agent Harness",
        )
        assert behavior_hierarchy(node_id) == (
            "Agent Team",
            "Agent Runtime",
            "Agent Harness",
        )
        assert owner_for(node_id) == "runtime"

        mcp_node_id = (
            "tests/unit/infrastructure/mcp/test_authorized_mcp_server.py::"
            "TestAuthorizedMCPServer::test_lists_tools"
        )
        assert suite_hierarchy(mcp_node_id) == (
            "Unit",
            "MCP",
            "Authorized MCP Server",
        )

        reporting_node_id = (
            "tests/unit/reporting/test_allure_ci_context.py::"
            "TestAllureCIContext::test_context_link"
        )
        assert suite_hierarchy(reporting_node_id) == (
            "Unit",
            "Reporting",
            "Allure CI Context",
        )

        runtime_collision = (
            "tests/unit/application/runtime/test_context_reporting.py::"
            "TestReportingContext::test_audit_context"
        )
        assert suite_hierarchy(runtime_collision)[1] == "Runtime"

    def test_maps_integration_markers_to_focused_tags(self) -> None:
        """Include layer, subsystem, live-model, and evaluation tags."""
        node_id = (
            "tests/integration/evaluation/test_live_ollama_eval.py::"
            "TestLiveOllamaEval::test_case"
        )

        tags = metadata_tags(node_id, ("ollama_eval",))

        assert tags == (
            "evaluation",
            "integration",
            "live-evaluation",
            "ollama_eval",
        )

    def test_assigns_boundary_and_ordinary_severities(self) -> None:
        """Reserve blocker severity for security boundaries."""
        boundary = (
            "tests/unit/infrastructure/mcp/"
            "test_authorized_mcp_server.py::TestServer::"
            "test_capability_denied"
        )
        ordinary = (
            "tests/unit/application/context/test_feature_context_builder.py::"
            "TestFeatureContextBuilder::test_builds_context"
        )

        assert severity_for(boundary) == "blocker"
        assert severity_for(ordinary) == "normal"

    def test_uses_stable_private_identifiers(self) -> None:
        """Hash node IDs consistently without retaining parameter text."""
        node_id = "tests/unit/test_sample.py::TestSample::test_value[secret]"

        first = stable_test_id(node_id)

        assert first == stable_test_id(node_id)
        assert first.startswith("pytest-")
        assert "secret" not in first
        assert safe_node_id(node_id).startswith(
            "tests/unit/test_sample.py::TestSample::test_value[parameters:",
        )
        assert "secret" not in safe_node_id(node_id)

    def test_builds_readable_titles_with_safe_parameters(self) -> None:
        """Show useful enum values while redacting prompt-like values."""
        title = readable_title(
            "test_role_behavior[parameters]",
            {
                "role": DevelopmentRole.BUSINESS_ANALYST,
                "prompt": "private prompt text",
            },
        )

        assert title == "Role Behavior [role=business_analyst]"
        assert safe_parameter_value("prompt", "private") == "<redacted>"

    def test_redacts_unknown_and_payload_parameters(self) -> None:
        """Deny parameter display unless its name and value are allowlisted."""
        private_value = "unique-private-parameter"

        for name in (
            "arguments",
            "arguments_json",
            "content",
            "description",
            "message",
            "output",
            "password",
            "path",
            "payload",
            "prompt",
            "request",
            "response",
            "secret",
            "token",
            "tool_input",
            "unknown_name",
        ):
            assert safe_parameter_value(name, private_value) == "<redacted>"
            assert private_value not in readable_title(
                "test_private_value",
                {name: private_value},
            )

    def test_preserves_only_valid_allowlisted_values(self) -> None:
        """Retain useful roles, statuses, IDs, booleans, and exceptions."""
        assert (
            safe_parameter_value(
                "role",
                DevelopmentRole.BUSINESS_ANALYST,
            )
            == "business_analyst"
        )
        assert safe_parameter_value("status", "completed") == "completed"
        assert safe_parameter_value("feature_id", 7) == "7"
        assert safe_parameter_value("enabled", True) == "True"
        assert (
            safe_parameter_value("expected_exception", RuntimeError)
            == "RuntimeError"
        )
        assert safe_parameter_value("role", "private-value") == "<redacted>"

    def test_renders_collections_deterministically(self) -> None:
        """Sort sets while preserving intentional list and tuple order."""
        first_set = {"list_tasks", "get_feature", "list_artifacts"}
        second_set: set[str] = set()
        for tool_name in ("list_artifacts", "get_feature", "list_tasks"):
            second_set.add(tool_name)

        expected_set = "[get_feature, list_artifacts, list_tasks]"
        assert safe_parameter_value("tool_names", first_set) == expected_set
        assert safe_parameter_value("tool_names", second_set) == expected_set
        assert (
            safe_parameter_value(
                "tool_names",
                frozenset(second_set),
            )
            == expected_set
        )
        assert (
            safe_parameter_value(
                "tool_names",
                ["list_tasks", "get_feature"],
            )
            == "[list_tasks, get_feature]"
        )
        assert (
            safe_parameter_value(
                "tool_names",
                ("get_feature", "list_tasks"),
            )
            == "[get_feature, list_tasks]"
        )
        assert (
            safe_reported_parameter_value(
                "tool_names",
                "'[get_feature, list_tasks]'",
            )
            == "[get_feature, list_tasks]"
        )

    def test_replaces_unreadable_parameter_objects(self) -> None:
        """Redact complex values instead of exposing their representations."""
        value = object()

        assert safe_parameter_value("value", value) == "<redacted>"
