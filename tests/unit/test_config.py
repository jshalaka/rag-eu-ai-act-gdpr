from __future__ import annotations

import pytest

from rag_eu_law.app.config import LLMProvider, Settings
from rag_eu_law.domain.exceptions import ConfigurationError


def test_defaults_use_local_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_LLM_PROVIDER", raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_provider is LLMProvider.OLLAMA
    assert settings.rerank_enabled is False


def test_anthropic_requires_api_key() -> None:
    with pytest.raises(ConfigurationError):
        Settings(_env_file=None, llm_provider=LLMProvider.ANTHROPIC)


def test_anthropic_with_key_is_valid() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider=LLMProvider.ANTHROPIC,
        anthropic_api_key="test-key",
    )
    assert settings.anthropic_api_key is not None


def test_openai_requires_api_key_and_defaults_to_luna() -> None:
    with pytest.raises(ConfigurationError, match="RAG_OPENAI_API_KEY"):
        Settings(_env_file=None, llm_provider=LLMProvider.OPENAI)
    settings = Settings(_env_file=None, llm_provider=LLMProvider.OPENAI, openai_api_key="k")
    assert settings.openai_model == "gpt-6-luna"
    assert settings.openai_reasoning_effort == "low"
