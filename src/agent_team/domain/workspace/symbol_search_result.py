"""Workspace symbol search result."""

from dataclasses import dataclass

from agent_team.domain.workspace.symbol_definition import SymbolDefinition


@dataclass(frozen=True, slots=True)
class SymbolSearchResult:
    """Bounded source definitions matching one exact symbol name."""

    query_hash: str
    definitions: tuple[SymbolDefinition, ...]
    truncated: bool
