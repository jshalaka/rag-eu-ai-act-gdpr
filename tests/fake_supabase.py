"""A tiny in-memory stand-in for the parts of supabase-py that the project uses."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass
class _Response:
    data: list[dict[str, Any]]


class _Query:
    def __init__(self, rows: list[dict[str, Any]], fail: Exception | None) -> None:
        self._rows = rows
        self._fail = fail
        self._action = "select"
        self._payload: Any = None
        self._filters: list[tuple[str, Any]] = []
        self._orders: list[tuple[str, bool]] = []
        self._limit: int | None = None

    def select(self, _columns: str) -> _Query:
        return self

    def insert(self, payload: dict[str, Any]) -> _Query:
        self._action, self._payload = "insert", payload
        return self

    def update(self, payload: dict[str, Any]) -> _Query:
        self._action, self._payload = "update", payload
        return self

    def eq(self, column: str, value: Any) -> _Query:
        self._filters.append((column, value))
        return self

    def order(self, column: str, desc: bool = False) -> _Query:
        self._orders.append((column, desc))
        return self

    def limit(self, count: int) -> _Query:
        self._limit = count
        return self

    def execute(self) -> _Response:
        if self._fail is not None:
            raise self._fail
        actions = {"select": self._select, "insert": self._insert, "update": self._update}
        return actions[self._action]()

    def _matching(self) -> list[dict[str, Any]]:
        return [r for r in self._rows if all(r.get(c) == v for c, v in self._filters)]

    def _select(self) -> _Response:
        rows = self._matching()
        for column, desc in reversed(self._orders):
            rows = sorted(rows, key=lambda r, c=column: r[c], reverse=desc)  # type: ignore[misc]
        return _Response([dict(r) for r in rows[: self._limit]])

    def _insert(self) -> _Response:
        if any(r["run_id"] == self._payload["run_id"] for r in self._rows):
            raise RuntimeError('duplicate key value violates unique constraint "runs_pkey"')
        self._rows.append(dict(self._payload))
        return _Response([dict(self._payload)])

    def _update(self) -> _Response:
        matched = self._matching()
        for row in matched:
            row.update(self._payload)
        return _Response([dict(r) for r in matched])


class FakeSupabaseClient:
    """`client.table(name)` plus `client.auth` with email/password sign-in."""

    def __init__(
        self, accounts: dict[str, str] | None = None, fail: Exception | None = None
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        self.signed_out = False
        self._fail = fail
        self._accounts = accounts or {}
        self.auth = SimpleNamespace(sign_in_with_password=self._sign_in, sign_out=self._sign_out)

    def table(self, _name: str) -> _Query:
        return _Query(self.rows, self._fail)

    def _sign_in(self, credentials: dict[str, str]) -> SimpleNamespace:
        email = credentials["email"]
        if self._accounts.get(email) != credentials["password"]:
            raise RuntimeError("Invalid login credentials")
        return SimpleNamespace(user=SimpleNamespace(email=email), session=SimpleNamespace())

    def _sign_out(self) -> None:
        self.signed_out = True
