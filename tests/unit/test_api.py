from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_eu_law.app.api import create_api
from rag_eu_law.app.config import Settings
from rag_eu_law.app.container import Container
from rag_eu_law.domain.constants import DISCLAIMER, REFUSAL_MESSAGE
from rag_eu_law.domain.exceptions import ConfigurationError, GenerationError
from tests.fixtures import write_sample_pdf


class EchoLLM:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        tag = re.search(r"cite as: (\[[^\]]+\])", user_prompt)
        return f"As provided {tag.group(1)}." if tag else REFUSAL_MESSAGE


class DownLLM:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise GenerationError("model server down")


class MisconfiguredLLM:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise ConfigurationError("bad setting")


def _container(tmp_path: Path, llm: object, ingest: bool = True) -> Container:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    write_sample_pdf(raw / "sample.pdf")
    container = Container(
        Settings(
            _env_file=None,
            raw_dir=raw,
            processed_dir=tmp_path / "processed",
            embedder="hashing",
            vector_store="memory",
        )
    )
    container.llm = llm  # type: ignore[assignment]
    if ingest:
        container.indexing_pipeline.index(container.ingestion_pipeline.ingest_directory(raw))
    return container


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_api(lambda: _container(tmp_path, EchoLLM())))


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_structured_cited_answer(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "Is there a right to erasure?"})
    assert response.status_code == 200
    body = response.json()
    assert {"answer", "citations", "retrieved_chunks", "latency_ms", "disclaimer"} <= body.keys()
    assert body["citations"][0]["article"] == "3"
    assert body["disclaimer"] == DISCLAIMER


def test_ask_accepts_filters_and_top_k(client: TestClient) -> None:
    payload = {"question": "What does it say?", "regulation": "GDPR", "article": "2", "top_k": 1}
    body = client.post("/ask", json=payload).json()
    assert [c["chunk"]["article"] for c in body["retrieved_chunks"]] == ["2"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": "ab"},
        {"question": "x" * 2001},
        {"question": "valid question", "top_k": 0},
        {"question": "valid question", "top_k": 21},
        {"question": "valid question", "regulation": "CCPA"},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/ask", json=payload).status_code == 422


def test_empty_index_is_a_503(tmp_path: Path) -> None:
    api = create_api(lambda: _container(tmp_path, EchoLLM(), ingest=False))
    response = TestClient(api).post("/ask", json={"question": "anything at all"})
    assert response.status_code == 503
    assert "Run ingestion first" in response.json()["detail"]


def test_llm_failure_is_a_502(tmp_path: Path) -> None:
    api = create_api(lambda: _container(tmp_path, DownLLM()))
    response = TestClient(api).post("/ask", json={"question": "right to erasure?"})
    assert response.status_code == 502
    assert "model server down" in response.json()["detail"]


def test_container_is_built_once(tmp_path: Path) -> None:
    calls: list[int] = []

    def factory() -> Container:
        calls.append(1)
        return _container(tmp_path / str(len(calls)), EchoLLM())

    client = TestClient(create_api(factory))
    client.post("/ask", json={"question": "right to erasure?"})
    client.post("/ask", json={"question": "right to erasure?"})
    assert len(calls) == 1


def test_other_project_errors_are_a_500(tmp_path: Path) -> None:
    api = create_api(lambda: _container(tmp_path, MisconfiguredLLM()))
    response = TestClient(api).post("/ask", json={"question": "right to erasure?"})
    assert response.status_code == 500
    assert "bad setting" in response.json()["detail"]
