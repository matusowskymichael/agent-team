"""MCP server for local development workflow operations."""

import asyncio

from mcp.server import MCPServer

from agent_team.application.workflow.workflow_service import WorkflowService
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus

from .schemas.artifact_mcp_result import (
    ArtifactMcpResult,
)
from .schemas.development_task_mcp_result import (
    DevelopmentTaskMcpResult,
)
from .schemas.feature_mcp_result import (
    FeatureMcpResult,
)
from .schemas.feature_overview_mcp_result import (
    FeatureOverviewMcpResult,
)
from .workflow_result_serializers import (
    serialize_artifact,
    serialize_development_task,
    serialize_feature,
    serialize_feature_overview,
)
from .workflow_tool_parameters import (
    ArtifactContentParameter,
    ArtifactKindParameter,
    CreatedByParameter,
    DescriptionParameter,
    DevelopmentRoleParameter,
    FeatureIdParameter,
    FeatureStatusParameter,
    OptionalFeatureStatusParameter,
    TaskIdParameter,
    TaskStatusParameter,
    TitleParameter,
)

SERVER_NAME = "agent-team-workflow"
SERVER_VERSION = "0.1.0"
EXPECTED_WORKFLOW_TOOL_COUNT = 9


def create_workflow_mcp_server(service: WorkflowService) -> MCPServer:
    """Create an MCP server exposing workflow service tools."""
    server = MCPServer(
        name=SERVER_NAME,
        title="Agent Team Development Workflow",
        description="Stores local development features, artifacts, and tasks.",
        version=SERVER_VERSION,
        log_level="INFO",
    )
    registered_tool_count = (
        _register_feature_tools(server, service)
        + _register_artifact_tools(server, service)
        + _register_task_tools(server, service)
    )
    if registered_tool_count != EXPECTED_WORKFLOW_TOOL_COUNT:
        raise RuntimeError("Workflow MCP tool registration failed.")

    return server


def _register_feature_tools(
    server: MCPServer,
    service: WorkflowService,
) -> int:
    @server.tool(
        name="create_feature",
        title="Create Feature",
        description=(
            "Create a development feature. Returns the created feature with "
            "its ID, status, and UTC timestamps."
        ),
        structured_output=True,
    )
    async def create_feature(
        title: TitleParameter,
        description: DescriptionParameter,
        status: FeatureStatusParameter = FeatureStatus.DRAFT,
    ) -> FeatureMcpResult:
        """Create a feature from required text and an optional status."""
        await _checkpoint()
        feature = service.create_feature(
            title=title,
            description=description,
            status=status,
        )
        return serialize_feature(feature)

    @server.tool(
        name="get_feature",
        title="Get Feature",
        description=(
            "Retrieve one development feature by ID. Metadata only: this "
            "does not include attached artifacts or development tasks."
        ),
        structured_output=True,
    )
    async def get_feature(feature_id: FeatureIdParameter) -> FeatureMcpResult:
        """Get a feature by its ID."""
        await _checkpoint()
        feature = service.get_feature(feature_id)
        return serialize_feature(feature)

    @server.tool(
        name="get_feature_overview",
        title="Get Feature Overview",
        description=(
            "Retrieve complete feature details by ID, including feature "
            "metadata, all attached artifacts, and all development tasks. "
            "Use this when the user asks for complete feature details, full "
            "feature information, a feature overview, or a feature with "
            "artifacts and tasks."
        ),
        structured_output=True,
    )
    async def get_feature_overview(
        feature_id: FeatureIdParameter,
    ) -> FeatureOverviewMcpResult:
        """Get a feature with its artifacts and development tasks."""
        await _checkpoint()
        overview = service.get_feature_overview(feature_id)
        return serialize_feature_overview(overview)

    @server.tool(
        name="list_features",
        title="List Features",
        description=(
            "List development features, optionally filtered by feature status."
        ),
        structured_output=True,
    )
    async def list_features(
        status: OptionalFeatureStatusParameter = None,
    ) -> list[FeatureMcpResult]:
        """List features with an optional status filter."""
        await _checkpoint()
        features = service.list_features(status)
        return [serialize_feature(feature) for feature in features]

    return len(
        (
            create_feature,
            get_feature,
            get_feature_overview,
            list_features,
        ),
    )


def _register_artifact_tools(
    server: MCPServer,
    service: WorkflowService,
) -> int:
    @server.tool(
        name="add_artifact",
        title="Add Artifact",
        description=(
            "Attach a typed artifact to an existing feature. Returns the "
            "created artifact with its ID and UTC timestamp."
        ),
        structured_output=True,
    )
    async def add_artifact(
        feature_id: FeatureIdParameter,
        kind: ArtifactKindParameter,
        content: ArtifactContentParameter,
        created_by: CreatedByParameter,
    ) -> ArtifactMcpResult:
        """Add an artifact to an existing feature."""
        await _checkpoint()
        artifact = service.add_artifact(
            feature_id=feature_id,
            kind=kind,
            content=content,
            created_by=created_by,
        )
        return serialize_artifact(artifact)

    @server.tool(
        name="list_artifacts",
        title="List Artifacts",
        description="List all artifacts attached to an existing feature.",
        structured_output=True,
    )
    async def list_artifacts(
        feature_id: FeatureIdParameter,
    ) -> list[ArtifactMcpResult]:
        """List artifacts for an existing feature."""
        await _checkpoint()
        artifacts = service.list_artifacts(feature_id)
        return [serialize_artifact(artifact) for artifact in artifacts]

    return len((add_artifact, list_artifacts))


def _register_task_tools(
    server: MCPServer,
    service: WorkflowService,
) -> int:
    @server.tool(
        name="create_task",
        title="Create Task",
        description=(
            "Create a development task for an existing feature. Returns the "
            "created task with its ID, assigned role, status, and timestamps."
        ),
        structured_output=True,
    )
    async def create_task(
        feature_id: FeatureIdParameter,
        title: TitleParameter,
        description: DescriptionParameter,
        assigned_role: DevelopmentRoleParameter,
        status: TaskStatusParameter = TaskStatus.PENDING,
    ) -> DevelopmentTaskMcpResult:
        """Create a task for an existing feature."""
        await _checkpoint()
        task = service.create_task(
            feature_id=feature_id,
            title=title,
            description=description,
            assigned_role=assigned_role,
            status=status,
        )
        return serialize_development_task(task)

    @server.tool(
        name="list_tasks",
        title="List Tasks",
        description=(
            "List all development tasks attached to an existing feature."
        ),
        structured_output=True,
    )
    async def list_tasks(
        feature_id: FeatureIdParameter,
    ) -> list[DevelopmentTaskMcpResult]:
        """List tasks for an existing feature."""
        await _checkpoint()
        tasks = service.list_tasks(feature_id)
        return [serialize_development_task(task) for task in tasks]

    @server.tool(
        name="update_task_status",
        title="Update Task Status",
        description=(
            "Update an existing development task status. Returns the updated "
            "task with its current status and timestamps."
        ),
        structured_output=True,
    )
    async def update_task_status(
        task_id: TaskIdParameter,
        status: TaskStatusParameter,
    ) -> DevelopmentTaskMcpResult:
        """Update the status of an existing task."""
        await _checkpoint()
        task = service.update_task_status(task_id=task_id, status=status)
        return serialize_development_task(task)

    return len((create_task, list_tasks, update_task_status))


async def _checkpoint() -> None:
    await asyncio.sleep(0)
