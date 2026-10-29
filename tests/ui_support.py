"""A history service wired with fakes, shared across Streamlit AppTest reruns."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from rag_eu_law.app.auth import AuthenticatedUser
from rag_eu_law.app.config import Settings
from rag_eu_law.app.container import Container
from rag_eu_law.app.history import HistoryService
from rag_eu_law.domain.constants import REFUSAL_MESSAGE
from rag_eu_law.domain.exceptions import AuthenticationError
from rag_eu_law.domain.models import RunRecord
from rag_eu_law.domain.protocols import LLMClient
from rag_eu_law.storage.memory_store import InMemoryRunStore
from tests.fixtures import write_sample_pdf


class CitingLLM:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        tag = re.search(r"cite as: (\[[^\]]+\])", user_prompt)
        return f"As provided {tag.group(1)}." if tag else REFUSAL_MESSAGE


def build_container(directory: Path, llm: LLMClient | None = None) -> Container:
    raw = directory / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    write_sample_pdf(raw / "sample.pdf")
    container = Container(
        Settings(
            _env_file=None,
            raw_dir=raw,
            processed_dir=directory / "processed",
            history_db_path=directory / "history.sqlite",
            embedder="hashing",
            vector_store="memory",
        )
    )
    container.llm = llm or CitingLLM()
    chunks = container.ingestion_pipeline.ingest_directory(raw)
    container.indexing_pipeline.index(chunks)
    return container


_SERVICE: HistoryService | None = None


def shared_service() -> HistoryService:
    global _SERVICE  # noqa: PLW0603 - AppTest reruns the script; state must survive reruns
    if _SERVICE is None:
        _SERVICE = build_container(Path(tempfile.mkdtemp())).history_service
    return _SERVICE


class FakeAuthenticator:
    """Accepts one account and gives it an in-memory run store, like a signed-in Supabase client."""

    def __init__(self) -> None:
        self.run_store = InMemoryRunStore()

    def sign_in(self, email: str, password: str) -> AuthenticatedUser:
        if (email, password) != ("ana@team.test", "correct horse"):
            raise AuthenticationError("Sign-in failed. Check your email and password.")
        return AuthenticatedUser(email=email, run_store=self.run_store, sign_out=lambda: None)


_CONTAINER: Container | None = None
_AUTHENTICATOR = FakeAuthenticator()


def shared_container() -> Container:
    global _CONTAINER  # noqa: PLW0603 - AppTest reruns the script; state must survive reruns
    if _CONTAINER is None:
        _CONTAINER = build_container(Path(tempfile.mkdtemp()))
        _CONTAINER.authenticator = _AUTHENTICATOR
    return _CONTAINER


def signed_in_runs() -> list[RunRecord]:
    """Runs saved through the signed-in user's store."""
    return _AUTHENTICATOR.run_store.list_runs(10)
