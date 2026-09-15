"""MCP result schema for a task verification contract."""

from typing import TypedDict


class TaskVerificationContractMcpResult(TypedDict):
    """Structured MCP representation of task verification rules."""

    profile_name: str
    required_checks: list[str]
