"""In-memory run store, for tests and throwaway sessions."""

from __future__ import annotations

from rag_eu_law.domain.exceptions import RunStoreError
from rag_eu_law.domain.models import RunRecord


class InMemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def save(self, record: RunRecord) -> None:
        if record.run_id in self._runs:
            raise RunStoreError(f"Run {record.run_id} already exists")
        self._runs[record.run_id] = record

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def list_runs(self, limit: int, user: str | None = None) -> list[RunRecord]:
        runs = [r for r in self._runs.values() if user is None or r.user == user]
        return sorted(runs, key=lambda r: (r.created_at, r.run_id), reverse=True)[:limit]

    def rate(self, run_id: str, rating: int, note: str | None) -> None:
        record = self._runs.get(run_id)
        if record is None:
            raise RunStoreError(f"Unknown run {run_id}")
        self._runs[run_id] = record.model_copy(update={"rating": rating, "note": note})
