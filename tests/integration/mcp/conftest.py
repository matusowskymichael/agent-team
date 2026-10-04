"""Owned MCP child-process lifecycle fixtures."""

import os
import signal
import sys
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

import pytest

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.runtime.capability_authorizer import (
    CapabilityAuthorizer,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_process_options as process_options,
)
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_server_config as server_config,
)
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_server_factory as server_factory,
)
from agent_team.infrastructure.mcp.client.authorized_mcp_server import (
    AuthorizedMCPServer,
)
from tests.unit.fakes.audit.fake_agent_audit_repository import (
    FakeAgentAuditRepository,
)
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


@pytest.fixture
def stubborn_workflow_mcp(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> Iterator[tuple[AuthorizedMCPServer, Path]]:
    """Create a child that ignores graceful stdin/SIGTERM termination."""
    mode: object = getattr(request, "param", "normal")
    assert isinstance(mode, str)
    executable = tmp_path / "stubborn-mcp"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path('mcp.pid').write_text(str(os.getpid()))\n"
        "from agent_team.infrastructure.mcp.server."
        "workflow_mcp_entrypoint import main\n"
        "mode = os.environ['MCP_TEST_MODE']\n"
        "if mode == 'normal':\n"
        "    main()\n"
        "elif mode == 'malformed':\n"
        "    print('invalid-secret-json', flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    role = DevelopmentRole.BUSINESS_ANALYST
    profile = AgentProfileCatalog().get_profile(role)
    audit = FakeAgentAuditRepository()
    server = server_factory.create_development_workflow_mcp_server(
        server_config.DevelopmentWorkflowMCPServerConfig(
            profile=profile,
            authorizer=CapabilityAuthorizer(FakeWorkflowRepository()),
            audit_repository=audit,
            run=audit.open_run(role=role),
        ),
        process_options.DevelopmentWorkflowMCPProcessOptions(
            python_executable=str(executable),
            cwd=tmp_path,
            environ={
                "PYTHONPATH": str(Path.cwd() / "src"),
                "AGENT_TEAM_DB_PATH": str(tmp_path / "workflow.db"),
                "MCP_TEST_MODE": mode,
            },
        ),
    )
    marker = tmp_path / "mcp.pid"
    yield server, marker
    if marker.exists():
        with suppress(ProcessLookupError):
            os.killpg(int(marker.read_text()), signal.SIGKILL)
