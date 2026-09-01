"""Code search result."""

from dataclasses import dataclass

from agent_team.domain.workspace.code_search_match import CodeSearchMatch


@dataclass(frozen=True, slots=True)
class CodeSearchResult:
    """Bounded code-search results for one query."""

    query_hash: str
    matches: tuple[CodeSearchMatch, ...]
    truncated: bool
