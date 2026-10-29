"""Contract tests for LLMClient and AnswerGenerator implementations."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import httpx
import pytest

from rag_eu_law.domain.constants import REFUSAL_MESSAGE
from rag_eu_law.domain.exceptions import GenerationError
from rag_eu_law.domain.models import ScoredChunk
from rag_eu_law.domain.protocols import AnswerGenerator, LLMClient
from rag_eu_law.generation.generator import GroundedAnswerGenerator
from rag_eu_law.generation.llm_clients import AnthropicClient, OllamaClient, OpenAIClient
from tests.fixtures import erasure_chunk


def _ollama(ok: bool) -> LLMClient:
    def handle(_: httpx.Request) -> httpx.Response:
        if not ok:
            return httpx.Response(503)
        return httpx.Response(200, json={"message": {"content": "reply"}})

    return OllamaClient("http://x.test", "m", transport=httpx.MockTransport(handle))


def _anthropic(ok: bool) -> LLMClient:
    def create(**_: object) -> SimpleNamespace:
        if not ok:
            raise RuntimeError("down")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="reply")])

    return AnthropicClient(
        "k", "m", client=SimpleNamespace(messages=SimpleNamespace(create=create))
    )


def _openai(ok: bool) -> LLMClient:
    def create(**_: object) -> SimpleNamespace:
        if not ok:
            raise RuntimeError("down")
        message = SimpleNamespace(content="reply")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return OpenAIClient("k", "m", "low", client=fake)


CLIENTS: list[Callable[[bool], LLMClient]] = [_ollama, _anthropic, _openai]


@pytest.mark.parametrize("make_client", CLIENTS)
def test_llm_client_returns_non_empty_text(make_client: Callable[[bool], LLMClient]) -> None:
    assert make_client(True).complete("system", "user").strip() == "reply"


@pytest.mark.parametrize("make_client", CLIENTS)
def test_llm_client_failure_raises_generation_error(
    make_client: Callable[[bool], LLMClient],
) -> None:
    with pytest.raises(GenerationError):
        make_client(False).complete("system", "user")


class _EchoLLM:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "Grounded [GDPR Art. 17(1), p. 61]. Invented [GDPR Art. 99, p. 1]."


GENERATORS: list[Callable[[], AnswerGenerator]] = [lambda: GroundedAnswerGenerator(_EchoLLM())]


@pytest.mark.parametrize("make_generator", GENERATORS)
def test_generator_cites_only_retrieved_chunks(
    make_generator: Callable[[], AnswerGenerator],
) -> None:
    context = [ScoredChunk(chunk=erasure_chunk(), score=1.0)]
    answer = make_generator().generate("q", context)
    retrieved_ids = {item.chunk.chunk_id for item in context}
    assert answer.citations
    assert {c.chunk_id for c in answer.citations} <= retrieved_ids
    assert "Art. 99" not in answer.answer


@pytest.mark.parametrize("make_generator", GENERATORS)
def test_generator_refuses_without_context(make_generator: Callable[[], AnswerGenerator]) -> None:
    answer = make_generator().generate("q", [])
    assert answer.answer == REFUSAL_MESSAGE
    assert answer.citations == []
