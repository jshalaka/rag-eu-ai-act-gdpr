"""Contract tests: every RunStore implementation must pass these."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from rag_eu_law.domain.exceptions import RunStoreError
from rag_eu_law.domain.models import RunRecord
from rag_eu_law.domain.protocols import RunStore
from rag_eu_law.storage.memory_store import InMemoryRunStore
from rag_eu_law.storage.sqlite_store import SqliteRunStore
from rag_eu_law.storage.supabase_store import SupabaseRunStore
from tests.fake_supabase import FakeSupabaseClient


def make_record(run_id: str, created_at: str, user: str = "ana") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        created_at=created_at,
        user=user,
        question=f"question {run_id}?",
        answer="An answer [GDPR Art. 17, p. 61].",
        citations=["[GDPR Art. 17, p. 61]"],
        retrieved_chunk_ids=["GDPR:art17:seq0", "GDPR:art18:seq0"],
        refused=False,
        latency_ms=1234.5,
        model="openai:gpt-6-luna",
        index_version="abc123",
        regulation_filter="GDPR",
        article_filter="17",
    )


@pytest.fixture(params=["memory", "sqlite", "supabase-fake"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> RunStore:
    factories: dict[str, Callable[[], RunStore]] = {
        "memory": InMemoryRunStore,
        "sqlite": lambda: SqliteRunStore(tmp_path / "db" / "history.sqlite"),
        "supabase-fake": lambda: SupabaseRunStore(FakeSupabaseClient()),
    }
    return factories[request.param]()


def test_saved_run_round_trips_exactly(store: RunStore) -> None:
    record = make_record("r1", "2026-10-19T10:00:00+00:00")
    store.save(record)
    assert store.get("r1") == record


def test_unknown_run_is_none(store: RunStore) -> None:
    assert store.get("missing") is None


def test_duplicate_run_id_is_rejected(store: RunStore) -> None:
    store.save(make_record("r1", "2026-10-19T10:00:00+00:00"))
    with pytest.raises(RunStoreError):
        store.save(make_record("r1", "2026-10-19T11:00:00+00:00"))


def test_list_is_newest_first_limited_and_filterable_by_user(store: RunStore) -> None:
    store.save(make_record("r1", "2026-10-19T10:00:00+00:00", user="ana"))
    store.save(make_record("r2", "2026-10-19T11:00:00+00:00", user="ben"))
    store.save(make_record("r3", "2026-10-19T12:00:00+00:00", user="ana"))
    assert [r.run_id for r in store.list_runs(10)] == ["r3", "r2", "r1"]
    assert [r.run_id for r in store.list_runs(2)] == ["r3", "r2"]
    assert [r.run_id for r in store.list_runs(10, user="ana")] == ["r3", "r1"]


def test_rating_and_note_are_saved(store: RunStore) -> None:
    store.save(make_record("r1", "2026-10-19T10:00:00+00:00"))
    store.rate("r1", -1, "cited the wrong article")
    saved = store.get("r1")
    assert saved is not None
    assert (saved.rating, saved.note) == (-1, "cited the wrong article")


def test_rating_an_unknown_run_is_an_error(store: RunStore) -> None:
    with pytest.raises(RunStoreError, match="Unknown run"):
        store.rate("missing", 1, None)
