"""Tests for agent session binding service."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from agent_team.application.sessions.agent_session_service import (
    AgentSessionService,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.sessions.agent_session_binding_error import (
    AgentSessionBindingError,
)
from agent_team.domain.sessions.agent_session_id import derive_agent_session_id
from agent_team.domain.sessions.agent_session_metadata import (
    AgentSessionMetadata,
)
from agent_team.domain.sessions.invalid_agent_session_id_error import (
    InvalidAgentSessionIdError,
)


class _SessionRepository:
    def __init__(self) -> None:
        self.sessions: dict[str, AgentSessionMetadata] = {}

    def get_session(
        self,
        session_id: str,
    ) -> AgentSessionMetadata | None:
        return self.sessions.get(session_id)

    def create_session(
        self,
        session_id: str,
        feature_id: int,
        role: DevelopmentRole,
    ) -> AgentSessionMetadata:
        session = AgentSessionMetadata(
            session_id=session_id,
            feature_id=feature_id,
            role=role,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        self.sessions[session_id] = session
        return session

    def touch_session(self, session_id: str) -> AgentSessionMetadata:
        session = replace(
            self.sessions[session_id],
            updated_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        )
        self.sessions[session_id] = session
        return session


class TestAgentSessionService:
    """AgentSessionService behavior tests."""

    def test_derives_session_id_from_role_and_feature(self) -> None:
        """Create a deterministic binding when session ID is omitted."""
        repository = _SessionRepository()
        service = AgentSessionService(repository)

        session = service.prepare_session(
            feature_id=3,
            role=DevelopmentRole.BUSINESS_ANALYST,
            requested_session_id=None,
        )

        assert session is not None
        assert session.session_id == derive_agent_session_id(
            DevelopmentRole.BUSINESS_ANALYST,
            3,
        )

    def test_session_id_requires_feature_id(self) -> None:
        """Reject persistent sessions without a feature scope."""
        service = AgentSessionService(_SessionRepository())

        with pytest.raises(AgentSessionBindingError):
            service.prepare_session(
                feature_id=None,
                role=DevelopmentRole.BUSINESS_ANALYST,
                requested_session_id="ba-feature-1",
            )

    def test_rejects_session_reuse_for_another_role(self) -> None:
        """Prevent cross-role session reuse."""
        repository = _SessionRepository()
        service = AgentSessionService(repository)
        service.prepare_session(
            feature_id=1,
            role=DevelopmentRole.BUSINESS_ANALYST,
            requested_session_id="feature-1",
        )

        with pytest.raises(AgentSessionBindingError):
            service.prepare_session(
                feature_id=1,
                role=DevelopmentRole.SOFTWARE_ARCHITECT,
                requested_session_id="feature-1",
            )

    def test_rejects_session_reuse_for_another_feature(self) -> None:
        """Prevent cross-feature session reuse."""
        repository = _SessionRepository()
        service = AgentSessionService(repository)
        service.prepare_session(
            feature_id=1,
            role=DevelopmentRole.BUSINESS_ANALYST,
            requested_session_id="feature-1",
        )

        with pytest.raises(AgentSessionBindingError):
            service.prepare_session(
                feature_id=2,
                role=DevelopmentRole.BUSINESS_ANALYST,
                requested_session_id="feature-1",
            )

    @pytest.mark.parametrize("session_id", ["", " unsafe", "../unsafe"])
    def test_rejects_unsafe_session_ids(self, session_id: str) -> None:
        """Reject blank or unsafe user-supplied session IDs."""
        service = AgentSessionService(_SessionRepository())

        with pytest.raises(InvalidAgentSessionIdError):
            service.prepare_session(
                feature_id=1,
                role=DevelopmentRole.BUSINESS_ANALYST,
                requested_session_id=session_id,
            )
