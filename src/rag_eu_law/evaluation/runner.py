"""Run the evaluation set through retrieval and (optionally) generation."""

from __future__ import annotations
from time import perf_counter

import logging

from rag_eu_law.domain.exceptions import RagError
from rag_eu_law.domain.models import Answer, ScoredChunk
from rag_eu_law.domain.protocols import AnswerGenerator, Retriever
from rag_eu_law.evaluation.models import EvalQuestion, QuestionResult

logger = logging.getLogger(__name__)


class EvalRunner:
    """Retrieves once per question, then generates from that same context."""

    def __init__(self, retriever: Retriever, generator: AnswerGenerator | None, top_k: int) -> None:
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k

    def run(self, questions: list[EvalQuestion]) -> list[QuestionResult]:
        return [self._run_one(question) for question in questions]

    def _run_one(self, question: EvalQuestion) -> QuestionResult:
        retrieval_started = perf_counter()

        try:
            context = self._retriever.retrieve(question.question, self._top_k)
        except RagError as error:
            retrieval_ms = (perf_counter() - retrieval_started) * 1000
            logger.warning("question %s failed: %s", question.id, error)
            return QuestionResult(
                question_id=question.id,
                type=question.type,
                question=question.question,
                retrieved=[],
                retrieval_latency_ms=retrieval_ms,
                error=str(error),
            )

        retrieval_ms = (perf_counter() - retrieval_started) * 1000

        try:
            answer = (
                self._generator.generate(question.question, context)
                if self._generator
                else None
            )
        except RagError as error:
            logger.warning("question %s failed: %s", question.id, error)
            result = score_question(question, context, None)
            return result.model_copy(
                update={
                    "retrieval_latency_ms": retrieval_ms,
                    "error": str(error),
                }
            )

        result = score_question(question, context, answer)
        return result.model_copy(
            update={"retrieval_latency_ms": retrieval_ms}
        )


def score_question(
    question: EvalQuestion, context: list[ScoredChunk], answer: Answer | None
) -> QuestionResult:
    retrieved = [item.chunk for item in context]

    hits = [any(source.matches(chunk) for chunk in retrieved) for source in question.expected]
    missing_sources = (
        [
            source
            for source, found in zip(question.expected, hits, strict=True)
            if not found
        ]
        if question.expected
        else None
        )
    refused = None if answer is None else not answer.citations
    citation_correct = None
    if answer is not None and question.answerable:
        citation_correct = any(
            source.matches(citation)
            for source in question.expected
            for citation in answer.citations
        )
    return QuestionResult(
        missing_sources=missing_sources,
        question_id=question.id,
        type=question.type,
        question=question.question,
        retrieved=[chunk.chunk_id for chunk in retrieved],
        retrieved_context=retrieved,
        retrieval_recall=sum(hits) / len(hits) if hits else None,
        retrieval_hit=any(hits) if hits else None,
        answer=answer.answer if answer else None,
        citations=[citation.format() for citation in answer.citations] if answer else [],
        refused=refused,
        citation_correct=citation_correct,
        latency_ms=answer.latency_ms if answer else None,
    )
