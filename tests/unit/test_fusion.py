from __future__ import annotations

import pytest

from rag_eu_law.domain.models import ScoredChunk
from rag_eu_law.retrieval.fusion import RankedList, reciprocal_rank_fusion
from tests.fixtures import SAMPLE_CHUNKS


def _ranked(*indexes: int) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=SAMPLE_CHUNKS[i], score=0.0) for i in indexes]


def test_chunk_in_both_lists_beats_chunk_in_one() -> None:
    fused = reciprocal_rank_fusion([RankedList(_ranked(0, 1)), RankedList(_ranked(2, 1))])
    assert fused[0].chunk.chunk_id == SAMPLE_CHUNKS[1].chunk_id


def test_score_is_sum_of_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([RankedList(_ranked(0)), RankedList(_ranked(1, 0))], 60)
    assert fused[0].chunk.chunk_id == SAMPLE_CHUNKS[0].chunk_id
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_weight_scales_a_lists_votes() -> None:
    fused = reciprocal_rank_fusion([RankedList(_ranked(0)), RankedList(_ranked(1), weight=3.0)])
    assert fused[0].chunk.chunk_id == SAMPLE_CHUNKS[1].chunk_id


def test_ties_break_by_chunk_id_and_empty_input_is_empty() -> None:
    fused = reciprocal_rank_fusion([RankedList(_ranked(1)), RankedList(_ranked(0))])
    assert [f.chunk.chunk_id for f in fused] == sorted(f.chunk.chunk_id for f in fused)
    assert reciprocal_rank_fusion([]) == []
