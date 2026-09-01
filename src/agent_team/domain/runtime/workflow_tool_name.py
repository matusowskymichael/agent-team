"""Workflow MCP tool names."""

from enum import StrEnum


class WorkflowToolName(StrEnum):
    """Known development workflow MCP tools."""

    CREATE_FEATURE = "create_feature"
    GET_FEATURE = "get_feature"
    GET_FEATURE_OVERVIEW = "get_feature_overview"
    LIST_FEATURES = "list_features"
    ADD_ARTIFACT = "add_artifact"
    LIST_ARTIFACTS = "list_artifacts"
    CREATE_TASK = "create_task"
    LIST_TASKS = "list_tasks"
    UPDATE_TASK_STATUS = "update_task_status"
