"""Application settings, loaded from environment / `.env` (prefix `RAG_`)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_eu_law.domain import constants
from rag_eu_law.domain.exceptions import ConfigurationError


class LLMProvider(StrEnum):
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    raw_dir: Path = Path(constants.DEFAULT_RAW_DIR)
    processed_dir: Path = Path(constants.DEFAULT_PROCESSED_DIR)
    log_level: str = constants.DEFAULT_LOG_LEVEL
    history_db_path: Path = Path(constants.DEFAULT_HISTORY_DB_PATH)
    run_store: str = constants.RUN_STORE_SQLITE
    supabase_url: str | None = None
    supabase_anon_key: SecretStr | None = None

    vector_store: str = constants.DEFAULT_VECTOR_STORE
    embedder: str = constants.DEFAULT_EMBEDDER
    embedding_model: str = constants.DEFAULT_EMBEDDING_MODEL
    embedding_cache_dir: Path | None = None

    embedding_batch_size: int = Field(default=constants.EMBEDDING_BATCH_SIZE, ge=1)

    top_k: int = Field(default=constants.DEFAULT_TOP_K, ge=1)
    candidates_per_retriever: int = Field(default=constants.DEFAULT_CANDIDATES_PER_RETRIEVER, ge=1)
    rerank_enabled: bool = False
    rerank_model: str = constants.DEFAULT_RERANK_MODEL

    llm_provider: LLMProvider = LLMProvider(constants.DEFAULT_LLM_PROVIDER)
    ollama_base_url: str = constants.DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = constants.DEFAULT_OLLAMA_MODEL
    anthropic_model: str = constants.DEFAULT_ANTHROPIC_MODEL
    anthropic_api_key: SecretStr | None = None
    openai_model: str = constants.DEFAULT_OPENAI_MODEL
    openai_reasoning_effort: str = constants.DEFAULT_OPENAI_REASONING_EFFORT
    openai_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _require_key_for_hosted_provider(self) -> Settings:
        keys = {
            LLMProvider.ANTHROPIC: ("RAG_ANTHROPIC_API_KEY", self.anthropic_api_key),
            LLMProvider.OPENAI: ("RAG_OPENAI_API_KEY", self.openai_api_key),
        }
        if self.llm_provider in keys:
            name, key = keys[self.llm_provider]
            if key is None or not key.get_secret_value():
                raise ConfigurationError(f"{name} is required when using {self.llm_provider.value}")
        return self

    @model_validator(mode="after")
    def _require_supabase_settings(self) -> Settings:
        if self.run_store != constants.RUN_STORE_SUPABASE:
            return self
        key = self.supabase_anon_key
        if not self.supabase_url or key is None or not key.get_secret_value():
            raise ConfigurationError(
                "RAG_SUPABASE_URL and RAG_SUPABASE_ANON_KEY are required "
                "when RAG_RUN_STORE=supabase"
            )
        return self

    @property
    def llm_label(self) -> str:
        """`provider:model`, stored with each run so results can be compared across models."""
        models = {
            LLMProvider.OLLAMA: self.ollama_model,
            LLMProvider.ANTHROPIC: self.anthropic_model,
            LLMProvider.OPENAI: self.openai_model,
        }
        return f"{self.llm_provider.value}:{models[self.llm_provider]}"
