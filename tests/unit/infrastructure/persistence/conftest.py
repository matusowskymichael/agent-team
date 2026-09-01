"""Persistence unit test fixtures."""

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest


@pytest.fixture
def sqlite_connection() -> Iterator[Callable[[Path], sqlite3.Connection]]:
    """Create SQLite connections and close them after each test."""
    connections: list[sqlite3.Connection] = []

    def connect(database_path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(database_path)
        connections.append(connection)
        return connection

    yield connect

    for connection in connections:
        connection.close()
