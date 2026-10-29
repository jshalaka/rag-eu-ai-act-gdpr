from __future__ import annotations

from rag_eu_law.app.service import AskService
from rag_eu_law.domain.constants import DISCLAIMER, REFUSAL_MESSAGE
from rag_eu_law.domain.models import Answer, ScoredChunk, SearchFilter
from rag_eu_law.generation.generator import GroundedAnswerGenerator
from tests.fixtures import erasure_chunk

CONTEXT = [ScoredChunk(chunk=erasure_chunk(), score=1.0)]


class ScriptedLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.reply


def _generate(reply: str, context: list[ScoredChunk] = CONTEXT) -> tuple[Answer, ScriptedLLM]:
    llm = ScriptedLLM(reply)
    return GroundedAnswerGenerator(llm).generate("What is erasure?", context), llm


def test_cited_answer_is_returned_with_citations_and_chunks() -> None:
    answer, llm = _generate("You may request erasure [GDPR Art. 17(1)(a), p. 61].")
    assert answer.answer == "You may request erasure [GDPR Art. 17(1)(a), p. 61]."
    assert [c.format() for c in answer.citations] == ["[GDPR Art. 17(1)(a), p. 61]"]
    assert answer.retrieved_chunks == CONTEXT
    assert answer.disclaimer == DISCLAIMER
    assert answer.latency_ms >= 0
    assert "What is erasure?" in llm.calls[0][1]


def test_empty_context_refuses_without_calling_the_model() -> None:
    answer, llm = _generate("anything", context=[])
    assert answer.answer == REFUSAL_MESSAGE
    assert llm.calls == []


def test_model_refusal_is_normalised() -> None:
    answer, _ = _generate("Not covered by the provided documents.")
    assert answer.answer == REFUSAL_MESSAGE
    assert answer.citations == []


def test_uncited_answer_is_refused() -> None:
    answer, _ = _generate("Erasure is a right of data subjects.")
    assert answer.answer == REFUSAL_MESSAGE


def test_answer_with_only_invented_citations_is_refused() -> None:
    answer, _ = _generate("Erasure exists [GDPR Art. 99, p. 3].")
    assert answer.answer == REFUSAL_MESSAGE
    assert answer.retrieved_chunks == CONTEXT


def test_ask_service_retrieves_then_generates_and_times_the_request() -> None:
    class StubRetriever:
        def __init__(self) -> None:
            self.args: tuple[str, int, SearchFilter | None] | None = None

        def retrieve(
            self, query: str, top_k: int, search_filter: SearchFilter | None = None
        ) -> list[ScoredChunk]:
            self.args = (query, top_k, search_filter)
            return CONTEXT

    retriever = StubRetriever()
    generator = GroundedAnswerGenerator(ScriptedLLM("Yes [GDPR Art. 17(1), p. 61]."))
    answer = AskService(retriever, generator, top_k=3).ask("Q?", SearchFilter(article="17"))
    assert retriever.args == ("Q?", 3, SearchFilter(article="17"))
    assert answer.citations
