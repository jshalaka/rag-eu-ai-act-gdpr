"""Run store on Supabase Postgres, through the PostgREST API.

The client must be signed in as a user: row-level security (see
`supabase/migrations/0001_runs.sql`) decides what each user may read and write.
"""

from __future__ import annotations

from typing import Any

from rag_eu_law.domain.constants import RUNS_TABLE
from rag_eu_law.domain.exceptions import RunStoreError
from rag_eu_law.domain.models import RunRecord


class SupabaseRunStore:
    def __init__(self, client: Any) -> None:
        self._client = client

    def save(self, record: RunRecord) -> None:
        self._run("save", lambda table: table.insert(record.model_dump()))

    def get(self, run_id: str) -> RunRecord | None:
        rows = self._run("read", lambda t: t.select("*").eq("run_id", run_id).limit(1))
        return _record(rows[0]) if rows else None

    def list_runs(self, limit: int, user: str | None = None) -> list[RunRecord]:
        def query(table: Any) -> Any:
            selected = table.select("*")
            if user is not None:
                selected = selected.eq("user", user)
            return selected.order("created_at", desc=True).order("run_id", desc=True).limit(limit)

        return [_record(row) for row in self._run("read", query)]

    def rate(self, run_id: str, rating: int, note: str | None) -> None:
        rows = self._run(
            "rate", lambda t: t.update({"rating": rating, "note": note}).eq("run_id", run_id)
        )
        if not rows:
            raise RunStoreError(f"Unknown run {run_id} (or not allowed to rate it)")

    def _run(self, action: str, build: Any) -> list[dict[str, Any]]:
        try:
            response = build(self._client.table(RUNS_TABLE)).execute()
        except Exception as error:  # PostgREST, auth and network errors all surface here
            raise RunStoreError(f"Supabase {action} failed: {error}") from error
        data = response.data
        return list(data) if data else []


def _record(row: dict[str, Any]) -> RunRecord:
    return RunRecord.model_validate(row)
