from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest
from streamlit.testing.v1 import AppTest

CLOUD_ENTRY = str(Path(__file__).resolve().parents[2] / "cloud" / "streamlit_app.py")
CLOUD_VARIABLES = (
    "RAG_OPENAI_API_KEY",
    "RAG_LLM_PROVIDER",
    "RAG_RUN_STORE",
    "RAG_SUPABASE_URL",
    "RAG_SUPABASE_ANON_KEY",
)


@pytest.fixture(autouse=True)
def isolated_environment() -> Iterator[None]:
    """The entrypoint writes defaults into os.environ; restore it so other tests are unaffected."""
    with mock.patch.dict(os.environ):
        for name in CLOUD_VARIABLES:
            os.environ.pop(name, None)
        yield


def test_missing_openai_key_is_explained_on_the_page() -> None:
    app = AppTest.from_file(CLOUD_ENTRY, default_timeout=60).run()
    assert not app.exception
    assert any("RAG_OPENAI_API_KEY is required when using openai" in e.value for e in app.error)
    assert any("no secrets at all" in m.value for m in app.markdown)


def test_secret_values_never_appear_on_the_error_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RUN_STORE", "supabase")
    monkeypatch.setenv("RAG_SUPABASE_URL", "https://example.supabase.test")
    app = AppTest.from_file(CLOUD_ENTRY, default_timeout=60).run()
    page = " ".join([*(e.value for e in app.error), *(m.value for m in app.markdown)])
    assert "example.supabase.test" not in page
    assert "RAG_OPENAI_API_KEY is required" in page


def test_error_page_lists_secret_names_but_not_values() -> None:
    app = AppTest.from_file(CLOUD_ENTRY, default_timeout=60)
    app.secrets["OPENAI_API_KEY"] = "sk-this-value-must-not-appear"  # wrong name on purpose
    app.run()
    page = " ".join([*(e.value for e in app.error), *(m.value for m in app.markdown)])
    assert "`OPENAI_API_KEY`" in page
    assert "sk-this-value-must-not-appear" not in page
