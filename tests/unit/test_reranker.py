from __future__ import annotations

import pytest

from rag_eu_law.domain.exceptions import RetrievalError
from rag_eu_law.domain.models import ScoredChunk
from rag_eu_law.retrieval.reranker import CrossEncoderReranker, NoOpReranker
from tests.fixtures import SAMPLE_CHUNKS

CANDIDATES = [ScoredChunk(chunk=c, score=0.1) for c in SAMPLE_CHUNKS]


class FakeCrossEncoder:
    """Scores passages by position from the end, so the last candidate wins."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(index) for index, _ in enumerate(pairs)]


class BrokenCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        raise RuntimeError("boom")


def test_noop_keeps_order_and_truncates() -> None:
    assert NoOpReranker().rerank("q", CANDIDATES, 2) == CANDIDATES[:2]


def test_cross_encoder_orders_by_model_score() -> None:
    reranker = CrossEncoderReranker("fake", model=FakeCrossEncoder())
    results = reranker.rerank("q", CANDIDATES, 2)
    assert [r.chunk.chunk_id for r in results] == [
        SAMPLE_CHUNKS[3].chunk_id,
        SAMPLE_CHUNKS[2].chunk_id,
    ]
    assert results[0].score == 3.0


def test_cross_encoder_failure_is_wrapped() -> None:
    with pytest.raises(RetrievalError, match="Reranking failed"):
        CrossEncoderReranker("fake", model=BrokenCrossEncoder()).rerank("q", CANDIDATES, 2)
