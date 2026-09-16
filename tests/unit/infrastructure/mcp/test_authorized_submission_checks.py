"""Regression tests for trusted check-attempt submission provenance."""

import asyncio
import json
from dataclasses import replace

import pytest
from mcp.types import TextContent

from agent_team.application.audit.audit_sanitizer import sanitize_tool_result
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.audit.tool_invocation_status import ToolInvocationStatus
from agent_team.infrastructure.mcp.client.authorized_mcp_server import (
    AuthorizedMCPServer,
)
from tests.unit.fakes.audit.fake_agent_audit_repository import (
    FakeAgentAuditRepository,
)
from tests.unit.fakes.mcp.fake_mcp_server import FakeMCPServer


class TestAuthorizedSubmissionChecks:
    """Only audited check results establish attempted checks."""

    @pytest.mark.parametrize(
        "checks", [[], ["backend"], ["passed everything"]]
    )
    def test_model_supplied_checks_are_denied(
        self,
        authorized_submission_server: AuthorizedMCPServer,
        checks: list[str],
    ) -> None:
        """Reject forged check claims under the trusted-argument policy."""
        server = authorized_submission_server

        result = asyncio.run(
            server.call_tool(
                "submit_task_for_verification",
                {"task_id": server.bound_task_id, "checks_attempted": checks},
            ),
        )

        assert result.is_error
        content = result.content[0]
        assert isinstance(content, TextContent)
        assert "cannot provide checks_attempted" in content.text
        assert "Do not retry" in content.text
        assert isinstance(server.delegate, FakeMCPServer)
        assert server.delegate.call_count == 0

    def test_submission_requires_audited_checks(
        self,
        authorized_submission_server: AuthorizedMCPServer,
    ) -> None:
        """Fail closed without any actual current-run check result."""
        server = authorized_submission_server

        result = asyncio.run(
            server.call_tool(
                "submit_task_for_verification",
                {"task_id": server.bound_task_id},
            ),
        )

        assert result.is_error
        content = result.content[0]
        assert isinstance(content, TextContent)
        assert "audited workspace check result" in content.text
        assert isinstance(server.delegate, FakeMCPServer)
        assert server.delegate.call_count == 0

    @pytest.mark.parametrize(
        ("result_preview", "expected"),
        [
            ('{"name":"backend","exit_code":0,"timed_out":false}', True),
            ('{"name":"backend","exit_code":1,"timed_out":false}', True),
            ('{"name":"backend","exit_code":124,"timed_out":true}', True),
            (None, False),
            ("truncated...", False),
            ("[]", False),
            ('{"name":"backend"}', False),
            ('{"name":"","exit_code":0,"timed_out":false}', False),
            ('{"name":42,"exit_code":0,"timed_out":false}', False),
            ('{"name":"backend","exit_code":true,"timed_out":false}', False),
            ('{"name":"backend","exit_code":0,"timed_out":"false"}', False),
        ],
    )
    def test_handoff_uses_actual_check_results(
        self,
        authorized_submission_server: AuthorizedMCPServer,
        result_preview: str | None,
        expected: bool,
    ) -> None:
        """Require complete results and count failures and timeouts."""
        server = authorized_submission_server
        audit = server.audit_repository
        assert isinstance(audit, FakeAgentAuditRepository)
        invocation = audit.start_tool_invocation(
            ToolInvocationStart(
                run_id=server.run.id,
                server_name="workspace",
                tool_name="run_check",
                classification=ToolClassification.READ_ONLY,
                arguments_hash="arguments-hash",
                arguments_preview_json='{"name":"forged-from-arguments"}',
            ),
        )
        audit.tool_invocations[invocation.id] = replace(
            invocation,
            status=ToolInvocationStatus.COMPLETED,
            result_preview=result_preview,
        )

        result = asyncio.run(
            server.call_tool(
                "submit_task_for_verification",
                {"task_id": server.bound_task_id},
            ),
        )

        assert result.is_error is not expected
        assert isinstance(server.delegate, FakeMCPServer)
        assert server.delegate.call_count == int(expected)
        if expected:
            delegated = server.delegate.received_calls[0][1]
            assert delegated is not None
            assert delegated["checks_attempted"] == ["backend"]

    @pytest.mark.parametrize(
        ("source", "status"),
        [
            ("other_run", ToolInvocationStatus.COMPLETED),
            ("other_server", ToolInvocationStatus.COMPLETED),
            ("other_tool", ToolInvocationStatus.COMPLETED),
            ("current", ToolInvocationStatus.ALLOWED),
            ("current", ToolInvocationStatus.FAILED),
            ("current", ToolInvocationStatus.DENIED),
        ],
    )
    def test_unrelated_or_incomplete_check_invocations_are_excluded(
        self,
        authorized_submission_server: AuthorizedMCPServer,
        source: str,
        status: ToolInvocationStatus,
    ) -> None:
        """Exclude other runs, other tools, and calls without results."""
        server = authorized_submission_server
        audit = server.audit_repository
        assert isinstance(audit, FakeAgentAuditRepository)
        other_run = audit.open_run()
        invocation = audit.start_tool_invocation(
            ToolInvocationStart(
                run_id=other_run.id
                if source == "other_run"
                else server.run.id,
                server_name="other"
                if source == "other_server"
                else "workspace",
                tool_name="other" if source == "other_tool" else "run_check",
                classification=ToolClassification.READ_ONLY,
                arguments_hash="arguments-hash",
                arguments_preview_json='{"name":"backend"}',
            ),
        )
        audit.tool_invocations[invocation.id] = replace(
            invocation,
            status=status,
            result_preview=(
                '{"name":"backend","exit_code":0,"timed_out":false}'
            ),
        )

        result = asyncio.run(
            server.call_tool(
                "submit_task_for_verification",
                {"task_id": server.bound_task_id},
            ),
        )

        assert result.is_error
        assert isinstance(server.delegate, FakeMCPServer)
        assert server.delegate.call_count == 0

    def test_checks_are_deduplicated_from_sanitized_results(
        self,
        authorized_submission_server: AuthorizedMCPServer,
    ) -> None:
        """Inject stable unique names without leaking captured output."""
        server = authorized_submission_server
        for name in ("ruff", "backend", "ruff"):
            invocation = server.audit_repository.start_tool_invocation(
                ToolInvocationStart(
                    run_id=server.run.id,
                    server_name="workspace",
                    tool_name="run_check",
                    classification=ToolClassification.READ_ONLY,
                    arguments_hash="arguments-hash",
                    arguments_preview_json=json.dumps({"name": name}),
                ),
            )
            result_hash, preview = sanitize_tool_result(
                "run_check",
                {
                    "name": name,
                    "exit_code": 1,
                    "timed_out": False,
                    "stdout_excerpt": "Private application details. " * 40,
                    "stderr_excerpt": "token=private-secret",
                },
            )
            server.audit_repository.complete_tool_invocation(
                invocation.id,
                result_hash,
                preview,
            )

        result = asyncio.run(
            server.call_tool(
                "submit_task_for_verification",
                {"task_id": server.bound_task_id},
            ),
        )

        assert not result.is_error
        assert isinstance(server.delegate, FakeMCPServer)
        delegated = server.delegate.received_calls[0][1]
        assert delegated is not None
        assert delegated["checks_attempted"] == ["backend", "ruff"]
        assert "Private application details" not in str(delegated)
        assert "private-secret" not in str(delegated)
