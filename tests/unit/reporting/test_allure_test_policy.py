"""Tests for centralized Allure test metadata policy."""

from agent_team.domain.runtime.development_role import DevelopmentRole
from tests.reporting.allure_test_policy import (
    behavior_hierarchy,
    metadata_tags,
    owner_for,
    readable_title,
    safe_node_id,
    safe_parameter_value,
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
            "tests/integration/reporting/test_allure_pytest_plugin.py::"
            "TestAllurePytestPlugin::test_plain_run"
        )
        assert suite_hierarchy(reporting_node_id) == (
            "Integration",
            "Reporting",
            "Allure Pytest Plugin",
        )

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

    def test_replaces_unreadable_parameter_objects(self) -> None:
        """Represent complex values by type instead of leaking their repr."""
        value = object()

        assert safe_parameter_value("value", value) == "<object>"
