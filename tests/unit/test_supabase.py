from __future__ import annotations

from pathlib import Path

import pytest

from rag_eu_law.app.auth import SIGN_IN_FAILED, SupabaseAuthenticator
from rag_eu_law.app.config import Settings
from rag_eu_law.app.container import Container
from rag_eu_law.domain.exceptions import AuthenticationError, ConfigurationError, RunStoreError
from rag_eu_law.storage.supabase_store import SupabaseRunStore
from tests.contract.test_run_store_contracts import make_record
from tests.fake_supabase import FakeSupabaseClient

ACCOUNTS = {"ana@team.test": "correct horse"}


def _authenticator(client: FakeSupabaseClient) -> SupabaseAuthenticator:
    return SupabaseAuthenticator("https://x.supabase.test", "anon", lambda _url, _key: client)


def test_sign_in_returns_user_with_a_working_run_store() -> None:
    client = FakeSupabaseClient(ACCOUNTS)
    user = _authenticator(client).sign_in(" ana@team.test ", "correct horse")
    assert user.email == "ana@team.test"
    user.run_store.save(make_record("r1", "2026-10-19T10:00:00+00:00"))
    assert [row["run_id"] for row in client.rows] == ["r1"]


@pytest.mark.parametrize("password", ["wrong", ""])
def test_bad_credentials_give_one_generic_message(password: str) -> None:
    with pytest.raises(AuthenticationError, match=SIGN_IN_FAILED):
        _authenticator(FakeSupabaseClient(ACCOUNTS)).sign_in("ana@team.test", password)


def test_sign_out_calls_the_client_and_never_raises() -> None:
    client = FakeSupabaseClient(ACCOUNTS)
    _authenticator(client).sign_in("ana@team.test", "correct horse").sign_out()
    assert client.signed_out

    def broken() -> None:
        raise RuntimeError("network")

    client.auth.sign_out = broken
    _authenticator(client).sign_in("ana@team.test", "correct horse").sign_out()


def test_backend_failures_become_run_store_errors() -> None:
    store = SupabaseRunStore(
        FakeSupabaseClient(fail=RuntimeError("permission denied for table runs"))
    )
    with pytest.raises(RunStoreError, match="permission denied"):
        store.list_runs(5)


def test_rating_a_row_you_cannot_see_is_reported() -> None:
    with pytest.raises(RunStoreError, match="not allowed"):
        SupabaseRunStore(FakeSupabaseClient()).rate("hidden", 1, None)


def test_settings_require_url_and_key_for_supabase() -> None:
    with pytest.raises(ConfigurationError, match="RAG_SUPABASE_URL"):
        Settings(_env_file=None, run_store="supabase")
    with pytest.raises(ConfigurationError):
        Settings(_env_file=None, run_store="supabase", supabase_url="https://x.supabase.test")
    ok = Settings(
        _env_file=None,
        run_store="supabase",
        supabase_url="https://x.supabase.test",
        supabase_anon_key="anon",
    )
    assert ok.run_store == "supabase"


def test_container_builds_authenticator_only_in_supabase_mode(tmp_path: Path) -> None:
    local = Container(Settings(_env_file=None, processed_dir=tmp_path))
    assert local.authenticator is None
    cloud = Container(
        Settings(
            _env_file=None,
            processed_dir=tmp_path,
            run_store="supabase",
            supabase_url="https://x.supabase.test",
            supabase_anon_key="anon",
        )
    )
    assert isinstance(cloud.authenticator, SupabaseAuthenticator)
    with pytest.raises(ConfigurationError, match="signed-in user"):
        _ = cloud.history_service
