"""Offline SDK scenario state for segment integration fixtures."""

from dataclasses import dataclass

from agents.items import ModelResponse, TResponseInputItem

from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.infrastructure.ollama.ollama_agent_executor import (
    OllamaAgentExecutor,
)
from agent_team.infrastructure.persistence.sqlite.sessions import (
    sqlite_session_factory,
)


@dataclass
class SdkSegmentScenario:
    """Script model responses while retaining real Runner/session behavior."""

    executor: OllamaAgentExecutor
    session_factory: sqlite_session_factory.SQLiteSessionFactory
    profile: AgentProfile
    run: AgentRunRecord
    task: AgentTask
    context: AgentContextEnvelope
    responses: list[ModelResponse | BaseException]
    inputs: list[str | list[TResponseInputItem]]
    operations: list[str]
