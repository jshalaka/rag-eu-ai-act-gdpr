"""SQLite run store: a single local file, no server needed."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from rag_eu_law.domain.exceptions import RunStoreError
from rag_eu_law.domain.models import RunRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    user TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations TEXT NOT NULL,
    retrieved_chunk_ids TEXT NOT NULL,
    refused INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    model TEXT NOT NULL,
    index_version TEXT NOT NULL,
    regulation_filter TEXT,
    article_filter TEXT,
    rating INTEGER,
    note TEXT
)
"""
JSON_COLUMNS = ("citations", "retrieved_chunk_ids")


class SqliteRunStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            # Exit the transaction first, then close the connection.
            # Both cleanup steps also run if schema creation fails.
            with closing(self._connect()) as connection, connection:
                connection.execute(SCHEMA)

        except (OSError, sqlite3.Error) as error:
            raise RunStoreError(
                f"Cannot open run history at {path}: {error}"
            ) from error

    def save(self, record: RunRecord) -> None:
        row = _to_row(record)
        columns = ", ".join(row)
        placeholders = ", ".join(f":{name}" for name in row)
        self._execute(f"INSERT INTO runs ({columns}) VALUES ({placeholders})", row)

    def get(self, run_id: str) -> RunRecord | None:
        rows = self._query("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        return rows[0] if rows else None

    def list_runs(self, limit: int, user: str | None = None) -> list[RunRecord]:
        where, params = ("WHERE user = ?", (user,)) if user is not None else ("", ())
        sql = f"SELECT * FROM runs {where} ORDER BY created_at DESC, run_id DESC LIMIT ?"
        return self._query(sql, (*params, limit))

    def rate(self, run_id: str, rating: int, note: str | None) -> None:
        updated = self._execute(
            "UPDATE runs SET rating = ?, note = ? WHERE run_id = ?", (rating, note, run_id)
        )
        if updated == 0:
            raise RunStoreError(f"Unknown run {run_id}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    def _execute(self, sql: str, params: Any) -> int:
        try:
            with closing(self._connect()) as connection, connection:
                return connection.execute(sql, params).rowcount
        except sqlite3.Error as error:
            raise RunStoreError(f"Run history write failed: {error}") from error

    def _query(self, sql: str, params: tuple[Any, ...]) -> list[RunRecord]:
        try:
            with closing(self._connect()) as connection:
                return [_from_row(row) for row in connection.execute(sql, params).fetchall()]
        except sqlite3.Error as error:
            raise RunStoreError(f"Run history read failed: {error}") from error


def _to_row(record: RunRecord) -> dict[str, Any]:
    row = record.model_dump()
    for column in JSON_COLUMNS:
        row[column] = json.dumps(row[column])
    row["refused"] = int(row["refused"])
    return row


def _from_row(row: sqlite3.Row) -> RunRecord:
    values = dict(row)
    for column in JSON_COLUMNS:
        values[column] = json.loads(values[column])
    values["refused"] = bool(values["refused"])
    return RunRecord.model_validate(values)
