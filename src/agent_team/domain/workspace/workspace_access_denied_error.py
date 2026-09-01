"""Workspace access denial error."""


class WorkspaceAccessDeniedError(PermissionError):
    """Raised when a workspace operation is not authorized."""
