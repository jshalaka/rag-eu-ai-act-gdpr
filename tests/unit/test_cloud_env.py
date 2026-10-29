from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rag_eu_law.app.cloud_env import apply_cloud_environment

ROOT = Path("/repo")
WORK = Path("/tmp/work")


def _apply(
    secrets: Mapping[str, object], environment: dict[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    environment = {} if environment is None else environment
    ignored = apply_cloud_environment(secrets, environment, ROOT, WORK)
    return environment, ignored


def test_secrets_become_environment_variables() -> None:
    environment, ignored = _apply({"RAG_OPENAI_API_KEY": "k", "RAG_RUN_STORE": "supabase"})
    assert environment["RAG_OPENAI_API_KEY"] == "k"
    assert environment["RAG_RUN_STORE"] == "supabase"
    assert ignored == []


def test_defaults_apply_but_can_be_overridden() -> None:
    assert _apply({})[0]["RAG_LLM_PROVIDER"] == "openai"
    assert _apply({"RAG_LLM_PROVIDER": "anthropic"})[0]["RAG_LLM_PROVIDER"] == "anthropic"


def test_pasted_dotenv_lines_cannot_break_the_cloud_install() -> None:
    pasted = {
        "RAG_VECTOR_STORE": "chroma",
        "RAG_EMBEDDER": "sentence-transformers",
        "RAG_EMBEDDING_BATCH_SIZE": "64",
        "RAG_PROCESSED_DIR": "data/processed",
        "RAG_OPENAI_API_KEY": "k",
    }
    environment, ignored = _apply(pasted)
    assert environment["RAG_VECTOR_STORE"] == "memory"
    assert environment["RAG_EMBEDDER"] == "fastembed"
    assert environment["RAG_EMBEDDING_BATCH_SIZE"] == "2"
    assert environment["RAG_PROCESSED_DIR"] == str(WORK / "processed")
    assert environment["RAG_RAW_DIR"] == str(ROOT / "data" / "raw")
    assert ignored == [
        "RAG_EMBEDDER",
        "RAG_EMBEDDING_BATCH_SIZE",
        "RAG_PROCESSED_DIR",
        "RAG_VECTOR_STORE",
    ]
    assert environment["RAG_OPENAI_API_KEY"] == "k"


def test_existing_environment_cannot_override_fixed_settings_either() -> None:
    environment, _ = _apply({}, {"RAG_VECTOR_STORE": "chroma"})
    assert environment["RAG_VECTOR_STORE"] == "memory"


def test_nested_and_non_string_secrets_are_skipped() -> None:
    environment, _ = _apply({"section": {"RAG_OPENAI_API_KEY": "k"}, "RAG_TOP_K": 5})
    assert "RAG_OPENAI_API_KEY" not in environment
    assert "RAG_TOP_K" not in environment
