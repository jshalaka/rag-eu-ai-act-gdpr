"""Reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass

from rag_eu_law.domain.constants import RRF_RANK_CONSTANT
from rag_eu_law.domain.models import ScoredChunk


@dataclass(frozen=True)
class RankedList:
    """Results of one retriever, best first, with the weight its votes carry."""

    results: list[ScoredChunk]
    weight: float = 1.0


def reciprocal_rank_fusion(
    ranked_lists: list[RankedList], rank_constant: int = RRF_RANK_CONSTANT
) -> list[ScoredChunk]:
    """Score each chunk by the sum of `weight / (rank_constant + rank)` over the lists."""
    scores: dict[str, float] = {}
    chunks: dict[str, ScoredChunk] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked.results, start=1):
            chunk_id = item.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + ranked.weight / (rank_constant + rank)
            chunks[chunk_id] = item
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [ScoredChunk(chunk=chunks[i].chunk, score=scores[i]) for i in ordered]
