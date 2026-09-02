"""Workspace tool name values."""

from enum import StrEnum


class WorkspaceToolName(StrEnum):
    """Known restricted workspace tool names."""

    LIST_FILES = "list_files"
    SEARCH_CODE = "search_code"
    FIND_SYMBOL = "find_symbol"
    READ_FILE = "read_file"
    APPLY_PATCH = "apply_patch"
    RUN_CHECK = "run_check"
