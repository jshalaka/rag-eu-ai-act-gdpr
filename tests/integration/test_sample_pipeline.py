"""Integration: ingest a 3-page sample PDF, ask a question, check the cited article."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rag_eu_law.app.config import Settings
from rag_eu_law.app.container import Container
from rag_eu_law.domain.constants import REFUSAL_MESSAGE
from tests.fixtures import write_sample_pdf


class CitingLLM:
    """Stands in for a real model: cites the first source, or refuses if told nothing relevant."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        tag = re.search(r"cite as: (\[[^\]]+\])", user_prompt)
        return f"As provided {tag.group(1)}." if tag else REFUSAL_MESSAGE


@pytest.fixture
def container(tmp_path: Path) -> Container:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_sample_pdf(raw / "sample.pdf")
    built = Container(
        Settings(
            _env_file=None,
            raw_dir=raw,
            processed_dir=tmp_path / "processed",
            embedder="hashing",
            vector_store="memory",
        )
    )
    built.llm = CitingLLM()
    return built


def test_question_about_erasure_cites_the_erasure_article(container: Container) -> None:
    chunks = container.ingestion_pipeline.ingest_directory(container.settings.raw_dir)
    container.indexing_pipeline.index(chunks)
    answer = container.ask_service.ask("Does the data subject have a right to erasure?")
    assert [c.article for c in answer.citations] == ["3"]
    assert answer.citations[0].page == 3
    assert answer.retrieved_chunks[0].chunk.article == "3"


def test_named_article_question_cites_that_article(container: Container) -> None:
    chunks = container.ingestion_pipeline.ingest_directory(container.settings.raw_dir)
    container.indexing_pipeline.index(chunks)
    answer = container.ask_service.ask("What does Article 2 say?")
    assert answer.citations[0].article == "2"
