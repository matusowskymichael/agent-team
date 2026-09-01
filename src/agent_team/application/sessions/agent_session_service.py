"""Application service for feature-scoped agent sessions."""

import re
from dataclasses import dataclass

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.sessions.agent_session_binding_error import (
    AgentSessionBindingError,
)
from agent_team.domain.sessions.agent_session_id import derive_agent_session_id
from agent_team.domain.sessions.agent_session_metadata import (
    AgentSessionMetadata,
)
from agent_team.domain.sessions.agent_session_repository import (
    AgentSessionRepository,
)
from agent_team.domain.sessions.invalid_agent_session_id_error import (
    InvalidAgentSessionIdError,
)

MAX_SESSION_ID_LENGTH = 128
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


@dataclass(frozen=True, slots=True)
class AgentSessionService:
    """Prepare and validate feature-scoped local agent sessions."""

    repository: AgentSessionRepository

    def prepare_session(
        self,
        feature_id: int | None,
        role: DevelopmentRole,
        requested_session_id: str | None,
    ) -> AgentSessionMetadata | None:
        """Return a verified session binding for a feature-scoped run."""
        if feature_id is None:
            if requested_session_id is not None:
                raise AgentSessionBindingError(
                    "A persisted session requires a feature ID.",
                )
            return None

        if feature_id < 1:
            raise AgentSessionBindingError("Feature ID must be positive.")

        session_id = (
            derive_agent_session_id(role, feature_id)
            if requested_session_id is None
            else requested_session_id
        )
        _validate_session_id(session_id)

        existing_session = self.repository.get_session(session_id)
        if existing_session is None:
            return self.repository.create_session(
                session_id=session_id,
                feature_id=feature_id,
                role=role,
            )

        if (
            existing_session.feature_id != feature_id
            or existing_session.role is not role
        ):
            raise AgentSessionBindingError(
                "Agent session is already bound to another role or feature.",
            )

        return self.repository.touch_session(session_id)


def _validate_session_id(session_id: str) -> None:
    stripped_session_id = session_id.strip()
    if stripped_session_id != session_id or not session_id:
        raise InvalidAgentSessionIdError("Session ID must not be blank.")
    if len(session_id) > MAX_SESSION_ID_LENGTH:
        raise InvalidAgentSessionIdError(
            "Session ID must be at most 128 characters.",
        )
    if SESSION_ID_PATTERN.fullmatch(session_id) is None:
        raise InvalidAgentSessionIdError(
            "Session ID may contain only letters, numbers, dots, "
            "underscores, colons, and hyphens.",
        )
