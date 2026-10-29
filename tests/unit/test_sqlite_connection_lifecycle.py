"""Verify schema-initialisation connections close on success and failure."""

import sqlite3

import pytest

from rag_eu_law.domain.exceptions import RunStoreError
from rag_eu_law.storage import sqlite_store


@pytest.mark.parametrize("schema_fails", [False, True])
def test_constructor_closes_connection(
    tmp_path, monkeypatch, schema_fails: bool
) -> None:
    connections = []
    original_connect = sqlite_store.SqliteRunStore._connect

    def tracked_connect(self):
        # Use a real connection so the test verifies actual resource cleanup.
        connection = original_connect(self)
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        sqlite_store.SqliteRunStore,
        "_connect",
        tracked_connect,
    )

    if schema_fails:
        monkeypatch.setattr(sqlite_store, "SCHEMA", "INVALID SQL")

        with pytest.raises(RunStoreError):
            sqlite_store.SqliteRunStore(tmp_path / "runs.db")
    else:
        sqlite_store.SqliteRunStore(tmp_path / "runs.db")

    assert len(connections) == 1

    try:
        # A closed SQLite connection must reject further operations.
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[0].execute("SELECT 1")
    finally:
        # Keep the test clean even if the closure assertion fails.
        connections[0].close()
