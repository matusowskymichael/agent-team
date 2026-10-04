"""Runtime CLI composition preserves verification cancellation."""

from agent_team.application.runtime.agent_harness import AgentHarness
from agent_team.application.runtime.orchestrator import Orchestrator
from agent_team.infrastructure.workspace.local_task_verifier import (
    LocalTaskVerifier,
)


class TestCliRuntimeComposition:
    """Expose the shared local verifier to runtime watchdog cleanup."""

    def test_runtime_composition_wires_local_verification_cancellation(
        self, local_runtime_orchestrator: Orchestrator
    ) -> None:
        """Cancel the verifier that actually owns active check subprocesses."""
        harness = local_runtime_orchestrator.agent_executor
        assert isinstance(harness, AgentHarness)
        service = harness.task_verification_service
        assert service is not None
        verifier = service.verifier
        assert isinstance(verifier, LocalTaskVerifier)
        assert (
            harness.cancel_verification == verifier.cancel_pending_operations
        )
