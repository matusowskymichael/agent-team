"""Workspace symbol definition metadata."""

from dataclasses import dataclass
from typing import Literal

SymbolKind = Literal["class", "function", "method"]


@dataclass(frozen=True, slots=True)
class SymbolDefinition:
    """One source definition located in the current workspace."""

    path: str
    line_number: int
    name: str
    qualified_name: str
    kind: SymbolKind
