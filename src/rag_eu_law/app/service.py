"""Question answering service: retrieve, then generate, timing the whole request."""

from __future__ import annotations

import time

from rag_eu_law.domain.models import Answer, SearchFilter
from rag_eu_law.domain.protocols import AnswerGenerator, Retriever


class AskService:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator, top_k: int) -> None:
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k

    def ask(
        self,
        question: str,
        search_filter: SearchFilter | None = None,
        top_k: int | None = None,
    ) -> Answer:
        started = time.perf_counter()
        context = self._retriever.retrieve(question, top_k or self._top_k, search_filter)
        answer = self._generator.generate(question, context)
        return answer.model_copy(update={"latency_ms": (time.perf_counter() - started) * 1000.0})
