from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from rag_eu_law.app.history import index_version
from rag_eu_law.domain.constants import REFUSAL_MESSAGE, UNKNOWN_INDEX_VERSION
from rag_eu_law.domain.exceptions import RunStoreError
from rag_eu_law.domain.models import Regulation, RunRecord, SearchFilter
from rag_eu_law.storage.sqlite_store import SqliteRunStore
from tests.contract.test_run_store_contracts import make_record
from tests.ui_support import build_container


class RefusingLLM:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return REFUSAL_MESSAGE


def test_ask_saves_a_complete_run(tmp_path: Path) -> None:
    container = build_container(tmp_path)
    service = container.history_service
    recorded = service.ask("Is there a right to erasure?", user="ana")
    saved = container.run_store.get(recorded.record.run_id)

    assert saved == recorded.record
    assert saved is not None
    assert saved.user == "ana"
    assert saved.refused is False
    assert saved.citations == [c.format() for c in recorded.answer.citations]
    assert saved.model == "ollama:qwen2.5:7b"
    assert saved.retrieved_chunk_ids
    assert saved.index_version == index_version(
        tmp_path / "processed" / "chunks.jsonl"
    )


def test_refusals_and_filters_are_recorded(tmp_path: Path) -> None:
    container = build_container(tmp_path, llm=RefusingLLM())
    recorded = container.history_service.ask(
        "anything",
        search_filter=SearchFilter(regulation=Regulation.GDPR, article="2"),
    )

    assert recorded.record.refused is True
    assert recorded.record.regulation_filter == "GDPR"
    assert recorded.record.article_filter == "2"
    assert recorded.record.user == "local"


def test_rate_and_recent(tmp_path: Path) -> None:
    service = build_container(tmp_path).history_service
    first = service.ask("right to erasure?").record
    service.rate(first.run_id, 1, "good")
    runs = service.recent(10)

    assert [(r.run_id, r.rating, r.note) for r in runs] == [
        (first.run_id, 1, "good")
    ]


def test_index_version_is_stable_and_changes_with_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chunks.jsonl"
    assert index_version(path) == UNKNOWN_INDEX_VERSION

    path.write_text("a")
    first = index_version(path)
    assert first == index_version(path)

    path.write_text("b")
    assert index_version(path) != first


def test_invalid_rating_is_rejected() -> None:
    values = make_record("r", "2026-10-19T10:00:00+00:00").model_dump()

    with pytest.raises(ValueError, match="rating"):
        RunRecord.model_validate({**values, "rating": 5})


def test_sqlite_store_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "h.sqlite"
    SqliteRunStore(path).save(
        make_record("r1", "2026-10-19T10:00:00+00:00")
    )

    assert SqliteRunStore(path).get("r1") is not None


def test_sqlite_store_reports_unusable_paths(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")

    with pytest.raises(RunStoreError, match="Cannot open"):
        SqliteRunStore(blocker / "sub" / "h.sqlite")


def test_sqlite_errors_are_wrapped(tmp_path: Path) -> None:
    path = tmp_path / "h.sqlite"
    store = SqliteRunStore(path)

    # Commit the table removal, then explicitly close the test connection.
    # SQLite's transaction context alone does not close the connection.
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DROP TABLE runs")

    with pytest.raises(RunStoreError, match="read failed"):
        store.list_runs(5)

    with pytest.raises(RunStoreError, match="write failed"):
        store.save(make_record("r1", "2026-10-19T10:00:00+00:00"))