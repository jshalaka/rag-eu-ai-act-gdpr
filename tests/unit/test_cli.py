from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_eu_law.app.cli import create_app
from rag_eu_law.app.config import Settings
from rag_eu_law.app.container import Container
from rag_eu_law.domain.constants import DISCLAIMER, REFUSAL_MESSAGE
from tests.fixtures import write_sample_pdf

runner = CliRunner()


class EchoLLM:
    """Answers with the first source's text and its own `cite as` tag."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        tag = re.search(r"cite as: (\[[^\]]+\])", user_prompt)
        return f"The sources say so {tag.group(1)}." if tag else REFUSAL_MESSAGE


@pytest.fixture
def container(tmp_path: Path) -> Container:
    raw = tmp_path / "raw"
    raw.mkdir()
    settings = Settings(
        _env_file=None,
        raw_dir=raw,
        processed_dir=tmp_path / "processed",
        embedder="hashing",
        vector_store="memory",
    )
    built = Container(settings)
    built.llm = EchoLLM()
    return built


def test_ingest_reports_counts_and_writes_chunks(container: Container) -> None:
    write_sample_pdf(container.settings.raw_dir / "sample.pdf")
    result = runner.invoke(create_app(lambda: container), ["ingest"])
    assert result.exit_code == 0, result.output
    assert "Ingested" in result.output
    assert "GDPR: " in result.output
    assert (container.settings.processed_dir / "chunks.jsonl").is_file()


def test_ingest_without_pdfs_fails_with_a_clear_message(container: Container) -> None:
    result = runner.invoke(create_app(lambda: container), ["ingest"])
    assert result.exit_code == 1
    assert "No PDF files found" in result.output


def test_ask_before_ingest_tells_the_user_to_ingest(container: Container) -> None:
    result = runner.invoke(create_app(lambda: container), ["ask", "What is erasure?"])
    assert result.exit_code == 1
    assert "Run ingestion first" in result.output


def test_ask_prints_answer_citations_and_disclaimer(container: Container) -> None:
    write_sample_pdf(container.settings.raw_dir / "sample.pdf")
    app = create_app(lambda: container)
    runner.invoke(app, ["ingest"])
    result = runner.invoke(app, ["ask", "What is the right to erasure?", "--show-chunks"])
    assert result.exit_code == 0, result.output
    assert "Citations:" in result.output
    assert "[GDPR Art. 3, p. 3]" in result.output
    assert "Retrieved:" in result.output
    assert DISCLAIMER in result.output


def test_ask_json_output_is_valid_and_filtered(container: Container) -> None:
    write_sample_pdf(container.settings.raw_dir / "sample.pdf")
    app = create_app(lambda: container)
    runner.invoke(app, ["ingest"])
    result = runner.invoke(
        app, ["ask", "What does it say?", "--json", "--regulation", "gdpr", "--article", "2"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {"answer", "citations", "retrieved_chunks", "latency_ms"} <= payload.keys()
    assert {c["chunk"]["article"] for c in payload["retrieved_chunks"]} == {"2"}


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(create_app(), [])
    assert "ingest" in result.output
    assert "ask" in result.output
