"""Tests for audit query service."""

import pytest

from agent_team.application.audit.audit_query_service import AuditQueryService
from agent_team.domain.audit.agent_run_not_found_error import (
    AgentRunNotFoundError,
)
from tests.unit.fakes.audit.audit_record_factories import make_agent_run_record
from tests.unit.fakes.audit.fake_agent_audit_reader import FakeAgentAuditReader


class TestAuditQueryService:
    """Audit query service behavior tests."""

    def test_lists_runs_with_limit(self) -> None:
        """Return runs from the reader with the requested limit."""
        reader = FakeAgentAuditReader(
            runs=[
                make_agent_run_record(run_id=2),
                make_agent_run_record(run_id=1),
            ],
        )
        service = AuditQueryService(reader=reader)

        runs = service.list_runs(limit=1)

        assert [run.id for run in runs] == [2]
        assert reader.received_limits == [1]

    def test_show_run_rejects_unknown_run_id(self) -> None:
        """Raise a concise domain error for missing runs."""
        service = AuditQueryService(reader=FakeAgentAuditReader())

        with pytest.raises(AgentRunNotFoundError) as error:
            service.show_run(404)

        assert str(error.value) == "Agent run 404 was not found."
