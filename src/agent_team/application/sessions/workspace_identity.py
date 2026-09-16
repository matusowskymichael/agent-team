"""Workspace identity hashing for persisted local sessions."""

from pathlib import Path

from agent_team.application.audit.audit_sanitizer import hash_text


def workspace_identity_hash(workspace_root: Path) -> str:
    """Return a non-reversible hash of a resolved workspace identity."""
    return hash_text(str(workspace_root.resolve(strict=False)))
