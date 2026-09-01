"""Tests for the command-line interface."""

import asyncio
import runpy
import sys
from pathlib import Path

import pytest

from agent_team.application.runtime.agent_harness import AgentHarness
from agent_team.application.runtime.orchestrator import Orchestrator
from agent_team.domain.runtime.agent_output_blank_error import (
    AgentOutputBlankError,
)
from agent_team.domain.runtime.agent_output_incomplete_error import (
    AgentOutputIncompleteError,
)
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.capability_denied_error import (
    CapabilityDeniedError,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.infrastructure.configuration.workflow_database_path import (
    AGENT_TEAM_DB_PATH_ENV,
)
from agent_team.infrastructure.mcp.client import (
    workflow_mcp_unavailable_error,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings
from agent_team.infrastructure.ollama.ollama_unavailable_error import (
    OllamaUnavailableError,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_audit_migration_error,
)
from agent_team.interfaces.cli import agent_cli as cli
from tests.unit.fakes.runtime.fake_agent_executor import FakeAgentExecutor


class TestCli:
    """CLI behavior tests."""

    def test_main_prints_agent_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Print only the final agent response."""

        async def run_prompt(
            task: AgentTask,
            model: str | None = None,
        ) -> AgentResult:
            _assert_task(task, prompt="Explain dependency inversion.")
            assert model is None
            return AgentResult(response="Depend on abstractions.")

        monkeypatch.setattr(cli, "run_prompt", run_prompt)

        exit_code = cli.main(["Explain dependency inversion."])

        captured = capsys.readouterr()

        assert exit_code == 0
        assert captured.out == "Depend on abstractions.\n"
        assert captured.err == ""

    def test_main_returns_error_when_ollama_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Print a concise error without a traceback for Ollama failures."""

        async def run_prompt(
            task: AgentTask,
            model: str | None = None,
        ) -> AgentResult:
            _assert_task(task, prompt="Explain dependency inversion.")
            assert model is None
            raise OllamaUnavailableError("Ollama is unavailable.")

        monkeypatch.setattr(cli, "run_prompt", run_prompt)

        exit_code = cli.main(["Explain dependency inversion."])

        captured = capsys.readouterr()

        assert exit_code == 1
        assert captured.out == ""
        assert captured.err == "Ollama is unavailable.\n"

    def test_main_returns_error_when_workflow_mcp_cannot_start(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Print a concise error without a traceback for MCP failures."""

        async def run_prompt(
            task: AgentTask,
            model: str | None = None,
        ) -> AgentResult:
            _assert_task(task, prompt="Create a feature.")
            assert model is None
            raise workflow_mcp_unavailable_error.WorkflowMCPUnavailableError(
                "Workflow MCP unavailable."
            )

        monkeypatch.setattr(cli, "run_prompt", run_prompt)

        exit_code = cli.main(["Create a feature."])

        captured = capsys.readouterr()

        assert exit_code == 1
        assert captured.out == ""
        assert captured.err == "Workflow MCP unavailable.\n"

    def test_main_returns_error_when_output_is_incomplete(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Print a concise incomplete-output error without a traceback."""

        async def run_prompt(
            task: AgentTask,
            model: str | None = None,
        ) -> AgentResult:
            _assert_task(task, prompt="Write proposal.")
            assert model is None
            raise AgentOutputIncompleteError(
                "The model reached its output limit before completing the "
                "response.",
            )

        monkeypatch.setattr(cli, "run_prompt", run_prompt)

        exit_code = cli.main(["Write proposal."])

        captured = capsys.readouterr()

        assert exit_code == 1
        assert captured.out == ""
        assert captured.err == (
            "The model reached its output limit before completing the "
            "response.\n"
        )

    def test_main_returns_error_when_output_is_blank(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Print a concise blank-output error without a traceback."""

        async def run_prompt(
            task: AgentTask,
            model: str | None = None,
        ) -> AgentResult:
            _assert_task(task, prompt="Write proposal.")
            assert model is None
            raise AgentOutputBlankError("The model returned blank output.")

        monkeypatch.setattr(cli, "run_prompt", run_prompt)

        exit_code = cli.main(["Write proposal."])

        captured = capsys.readouterr()

        assert exit_code == 1
        assert captured.out == ""
        assert captured.err == "The model returned blank output.\n"

    def test_main_returns_error_when_audit_migration_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Print a concise error without a traceback for DB migration."""

        async def run_prompt(
            task: AgentTask,
            model: str | None = None,
        ) -> AgentResult:
            _assert_task(task, prompt="Create a feature.")
            assert model is None
            raise sqlite_audit_migration_error.SQLiteAuditMigrationError(
                "Audit database migration failed."
            )

        monkeypatch.setattr(cli, "run_prompt", run_prompt)

        exit_code = cli.main(["Create a feature."])

        captured = capsys.readouterr()

        assert exit_code == 1
        assert captured.out == ""
        assert captured.err == "Audit database migration failed.\n"

    def test_package_execution_exits_with_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Run the package entry point through the CLI."""

        async def run_prompt(
            task: AgentTask,
            model: str | None = None,
        ) -> AgentResult:
            _assert_task(task, prompt="Explain dependency inversion.")
            assert model is None
            return AgentResult(response="Depend on abstractions.")

        monkeypatch.setattr(cli, "run_prompt", run_prompt)
        monkeypatch.setattr(
            sys,
            "argv",
            ["agent-team", "Explain dependency inversion."],
        )

        with pytest.raises(SystemExit) as exit_error:
            runpy.run_module("agent_team", run_name="__main__")

        captured = capsys.readouterr()

        assert exit_error.value.code == 0
        assert captured.out == "Depend on abstractions.\n"
        assert captured.err == ""

    def test_main_passes_explicit_role(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Pass the optional CLI role into the agent task."""

        async def run_prompt(
            task: AgentTask,
            model: str | None = None,
        ) -> AgentResult:
            _assert_task(
                task,
                prompt="List all features.",
                role=DevelopmentRole.BUSINESS_ANALYST,
            )
            assert model is None
            return AgentResult(response="Features listed.")

        monkeypatch.setattr(cli, "run_prompt", run_prompt)

        exit_code = cli.main(
            [
                "--role",
                "business_analyst",
                "List all features.",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert captured.out == "Features listed.\n"
        assert captured.err == ""

    def test_main_returns_error_when_capability_is_denied(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Print a concise denial without a traceback."""

        async def run_prompt(
            task: AgentTask,
            model: str | None = None,
        ) -> AgentResult:
            _assert_task(
                task,
                prompt="Create a feature.",
                role=DevelopmentRole.BUSINESS_ANALYST,
            )
            assert model is None
            raise CapabilityDeniedError("Capability denied.")

        monkeypatch.setattr(cli, "run_prompt", run_prompt)

        exit_code = cli.main(
            [
                "--role",
                "business_analyst",
                "Create a feature.",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 1
        assert captured.out == ""
        assert captured.err == "Capability denied.\n"

    def test_run_prompt_returns_orchestrator_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pass a parsed prompt to the composed orchestrator."""
        expected_result = AgentResult(response="Depend on abstractions.")
        agent_executor = FakeAgentExecutor(result=expected_result)
        selected_models: list[str] = []

        def build_orchestrator(settings: object) -> Orchestrator:
            assert isinstance(settings, OllamaSettings)
            selected_models.append(settings.model)
            return Orchestrator(agent_executor=agent_executor)

        monkeypatch.setattr(cli, "build_orchestrator", build_orchestrator)

        def ensure_model_ready(_settings: object) -> None:
            return None

        monkeypatch.setattr(
            cli,
            "ensure_ollama_model_ready",
            ensure_model_ready,
        )

        result = asyncio.run(
            cli.run_prompt(AgentTask(prompt="Explain."), model="qwen-cli"),
        )

        assert result == expected_result
        assert selected_models == ["qwen-cli"]
        assert agent_executor.received_task is not None
        assert agent_executor.received_task.prompt == "Explain."
        assert (
            agent_executor.received_task.role
            is DevelopmentRole.DELIVERY_MANAGER
        )

    def test_run_prompt_passes_developer_bindings_to_orchestrator(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pass trusted developer task and workspace through run_prompt."""
        expected_result = AgentResult(response="Patched.")
        agent_executor = FakeAgentExecutor(result=expected_result)

        def build_orchestrator(_settings: object) -> Orchestrator:
            return Orchestrator(agent_executor=agent_executor)

        def ensure_model_ready(_settings: object) -> None:
            return None

        monkeypatch.setattr(cli, "build_orchestrator", build_orchestrator)
        monkeypatch.setattr(
            cli,
            "ensure_ollama_model_ready",
            ensure_model_ready,
        )

        result = asyncio.run(
            cli.run_prompt(
                AgentTask(
                    prompt="Patch.",
                    role=DevelopmentRole.BACKEND_DEVELOPER,
                    feature_id=1,
                    task_id=2,
                    workspace_root=tmp_path,
                ),
            ),
        )

        assert result == expected_result
        assert agent_executor.received_task is not None
        assert agent_executor.received_task.task_id == 2
        assert agent_executor.received_task.workspace_root == tmp_path

    def test_build_orchestrator_uses_agent_harness(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Compose CLI model execution through the shared harness."""
        monkeypatch.setenv(
            AGENT_TEAM_DB_PATH_ENV,
            str(tmp_path / "workflow.db"),
        )

        orchestrator = cli.build_orchestrator()

        assert isinstance(orchestrator.agent_executor, AgentHarness)

    def test_main_passes_feature_session_and_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Pass feature, session, and model flags into the agent task."""

        async def run_prompt(
            task: AgentTask,
            model: str | None = None,
        ) -> AgentResult:
            _assert_task(
                task,
                prompt="Summarize feature.",
                role=DevelopmentRole.BUSINESS_ANALYST,
                feature_id=7,
                session_id="session-7",
            )
            assert model == "qwen3.5:9b"
            return AgentResult(response="Summary.")

        monkeypatch.setattr(cli, "run_prompt", run_prompt)

        exit_code = cli.main(
            [
                "--role",
                "business_analyst",
                "--feature-id",
                "7",
                "--session-id",
                "session-7",
                "--model",
                "qwen3.5:9b",
                "Summarize feature.",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert captured.out == "Summary.\n"
        assert captured.err == ""

    def test_main_passes_developer_task_and_workspace(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Pass trusted developer bindings into the agent task."""

        async def run_prompt(
            task: AgentTask,
            model: str | None = None,
        ) -> AgentResult:
            _assert_task(
                task,
                prompt="Patch backend.",
                role=DevelopmentRole.BACKEND_DEVELOPER,
                feature_id=4,
                task_id=9,
                workspace_root=tmp_path,
            )
            assert model is None
            return AgentResult(response="Patched.")

        monkeypatch.setattr(cli, "run_prompt", run_prompt)

        exit_code = cli.main(
            [
                "--role",
                "backend_developer",
                "--feature-id",
                "4",
                "--task-id",
                "9",
                "--workspace-root",
                str(tmp_path),
                "Patch backend.",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert captured.out == "Patched.\n"
        assert captured.err == ""

    def test_list_models_prints_local_models_without_running_agent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """List installed models without creating an agent run."""

        async def run_prompt(**_kwargs: object) -> AgentResult:
            raise AssertionError("agent run should not start")

        def list_models(_settings: object) -> list[str]:
            return ["qwen3.5:9b", "llama3.2:3b"]

        monkeypatch.setattr(cli, "run_prompt", run_prompt)
        monkeypatch.setattr(cli, "list_installed_ollama_models", list_models)

        exit_code = cli.main(["--list-models"])

        captured = capsys.readouterr()

        assert exit_code == 0
        assert captured.out == "qwen3.5:9b\nllama3.2:3b\n"
        assert captured.err == ""

    def test_list_skills_prints_role_skills_without_ollama(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """List local skills without starting a model run."""

        async def run_prompt(**_kwargs: object) -> AgentResult:
            raise AssertionError("agent run should not start")

        def ensure_model_ready(_settings: object) -> None:
            raise AssertionError("Ollama should not be contacted")

        monkeypatch.setattr(cli, "run_prompt", run_prompt)
        monkeypatch.setattr(
            cli,
            "ensure_ollama_model_ready",
            ensure_model_ready,
        )

        exit_code = cli.main(
            ["--list-skills", "--role", "business_analyst"],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert "NAME\tVERSION\tHASH\tDESCRIPTION" in captured.out
        assert "write-requirements-artifact" in captured.out
        assert "write-acceptance-criteria" in captured.out
        assert "review-feature-readiness" in captured.out
        assert captured.err == ""

    def test_list_skills_prints_backend_developer_skill(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """List the reviewed backend developer skill."""
        exit_code = cli.main(
            ["--list-skills", "--role", "backend_developer"],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert "implement-backend-task" in captured.out
        assert captured.err == ""

    def test_list_skills_prints_frontend_developer_skill(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """List the reviewed frontend developer skill."""
        exit_code = cli.main(
            ["--list-skills", "--role", "frontend_developer"],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert "implement-frontend-task" in captured.out
        assert captured.err == ""

    def test_list_skills_prints_architect_skills(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """List only software architect skills for the architect role."""
        exit_code = cli.main(
            ["--list-skills", "--role", "software_architect"],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert "review-architecture-readiness" in captured.out
        assert "design-solution-architecture" in captured.out
        assert "write-implementation-plan" in captured.out
        assert "decompose-development-tasks" in captured.out
        assert "write-requirements-artifact" not in captured.out
        assert captured.err == ""


def _assert_task(
    task: AgentTask,
    **expected: object,
) -> None:
    values: dict[str, object] = {
        "role": DevelopmentRole.DELIVERY_MANAGER,
        "feature_id": None,
        "session_id": None,
        "task_id": None,
        "workspace_root": None,
    }
    values.update(expected)
    assert task.prompt == values["prompt"]
    assert task.role is values["role"]
    assert task.feature_id == values["feature_id"]
    assert task.session_id == values["session_id"]
    assert task.task_id == values["task_id"]
    assert task.workspace_root == values["workspace_root"]
