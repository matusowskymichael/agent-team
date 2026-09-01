"""Live Ollama integration tests."""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest

from agent_team.application.runtime.agent_harness import AgentHarness
from agent_team.application.runtime.orchestrator import Orchestrator
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.infrastructure.configuration.workflow_database_path import (
    AGENT_TEAM_DB_PATH_ENV,
    load_workflow_database_path,
)
from agent_team.infrastructure.ollama.ollama_agent_executor import (
    OllamaAgentExecutor,
)
from agent_team.infrastructure.ollama.ollama_model_factory import (
    create_ollama_model,
)
from agent_team.infrastructure.ollama.ollama_settings import (
    load_ollama_settings,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository as workflow_repository_module,
)
from agent_team.interfaces.cli import agent_cli as cli


@pytest.mark.ollama
@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 to run live Ollama tests.",
)
class TestLiveOllama:
    """Live local Ollama behavior tests."""

    def test_qwen_prompt_returns_response(self) -> None:
        """Run the full local Ollama vertical slice."""
        settings = load_ollama_settings()
        model = create_ollama_model(settings)
        runtime = OllamaAgentExecutor(model=model, settings=settings)
        audit_repository = audit_repository_module.SQLiteAgentAuditRepository(
            load_workflow_database_path(),
        )
        agent_executor = AgentHarness(
            runtime=runtime,
            audit_repository=audit_repository,
        )
        orchestrator = Orchestrator(agent_executor=agent_executor)

        result = asyncio.run(
            orchestrator.run(
                AgentTask(
                    prompt=("Explain dependency inversion in one sentence."),
                ),
            ),
        )

        assert result.response.strip()

    def test_qwen_creates_feature_through_workflow_mcp(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Run Qwen through MCP and verify persisted workflow data."""
        database_path = tmp_path / "workflow.db"
        title = f"Live MCP Feature {uuid4()}"
        description = "A secure login and logout flow for user authentication."
        prompt = (
            "Create exactly one feature by calling the development workflow "
            f"MCP create_feature tool. Title: {title}. "
            f"Description: {description} Return the created feature ID."
        )
        monkeypatch.setenv(AGENT_TEAM_DB_PATH_ENV, str(database_path))

        result = asyncio.run(cli.run_prompt(AgentTask(prompt=prompt)))

        repository = workflow_repository_module.SQLiteWorkflowRepository(
            database_path
        )
        matching_features = [
            feature
            for feature in repository.list_features()
            if feature.title == title
        ]

        assert result.response.strip()
        assert len(matching_features) == 1
        assert matching_features[0].description == description
