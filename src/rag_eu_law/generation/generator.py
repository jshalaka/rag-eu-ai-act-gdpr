"""Grounded answer generation with citation verification and refusal."""

from __future__ import annotations

import logging
import time

from rag_eu_law.domain.constants import REFUSAL_MESSAGE
from rag_eu_law.domain.models import Answer, ScoredChunk
from rag_eu_law.domain.protocols import LLMClient
from rag_eu_law.generation.citations import verify_citations
from rag_eu_law.generation.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class GroundedAnswerGenerator:
    """`AnswerGenerator` that refuses unless the answer cites the retrieved sources.

    Refuses when: nothing was retrieved, the model says it cannot answer, or the model's
    answer has no citation that matches a retrieved chunk.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def generate(self, question: str, context: list[ScoredChunk]) -> Answer:
        started = time.perf_counter()
        if not context:
            return _refusal(context, started)
        raw = self._llm.complete(SYSTEM_PROMPT, build_user_prompt(question, context))
        verified = verify_citations(raw, context)
        if verified.removed_count:
            logger.warning("removed %d citations not backed by sources", verified.removed_count)
        if _is_refusal(verified.text) or not verified.citations:
            logger.info("refusing: no grounded citation in the model answer")
            return _refusal(context, started)
        return Answer(
            answer=verified.text,
            citations=verified.citations,
            retrieved_chunks=context,
            latency_ms=_elapsed_ms(started),
        )


def _is_refusal(text: str) -> bool:
    return REFUSAL_MESSAGE.lower() in text.lower()


def _refusal(context: list[ScoredChunk], started: float) -> Answer:
    return Answer(
        answer=REFUSAL_MESSAGE,
        citations=[],
        retrieved_chunks=context,
        latency_ms=_elapsed_ms(started),
    )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
