from __future__ import annotations

from rag_eu_law.domain.models import Regulation, ScoredChunk
from rag_eu_law.generation.citations import citation_for_chunk, verify_citations
from tests.fixtures import SAMPLE_CHUNKS, erasure_chunk, make_chunk

ERASURE = erasure_chunk()
CONTEXT = [
    ScoredChunk(chunk=ERASURE, score=1.0),
    ScoredChunk(chunk=SAMPLE_CHUNKS[2], score=0.9),
    ScoredChunk(chunk=SAMPLE_CHUNKS[3], score=0.8),
]


def test_valid_citation_is_kept_and_canonical() -> None:
    result = verify_citations("Erasure applies [GDPR Art. 17(1), p. 61].", CONTEXT)
    assert result.text == "Erasure applies [GDPR Art. 17(1), p. 61]."
    assert [c.chunk_id for c in result.citations] == [ERASURE.chunk_id]
    assert result.removed_count == 0


def test_point_is_kept_only_when_present_in_chunk() -> None:
    kept = verify_citations("A [GDPR Art. 17(1)(a), p. 61].", CONTEXT)
    assert kept.citations[0].point == "a"
    repaired = verify_citations("A [GDPR Art. 17(1)(z), p. 61].", CONTEXT)
    assert repaired.text == "A [GDPR Art. 17(1), p. 61]."
    assert repaired.citations[0].point is None


def test_wrong_page_is_repaired_to_the_chunk_page() -> None:
    result = verify_citations("A [GDPR Art. 17(1), p. 99].", CONTEXT)
    assert result.text == "A [GDPR Art. 17(1), p. 61]."


def test_page_inside_chunk_range_is_kept() -> None:
    assert "p. 62]" in verify_citations("A [GDPR Art. 17(1), p. 62].", CONTEXT).text


def test_unsupported_citations_are_removed() -> None:
    answer = (
        "One [GDPR Art. 99, p. 1]. Two [GDPR Art. 17(2), p. 61]. Three [AI Act Art. 17, p. 61]."
    )
    result = verify_citations(answer, CONTEXT)
    assert result.removed_count == 3
    assert result.citations == []
    assert "[" not in result.text


def test_recital_and_annex_citations() -> None:
    annex = make_chunk("AI_ACT:annexIII:seq0", "List.", regulation=Regulation.AI_ACT)
    annex = annex.model_copy(update={"annex": "III", "page_start": 424, "page_end": 425})
    context = [*CONTEXT, ScoredChunk(chunk=annex, score=0.5)]
    result = verify_citations("R [AI Act Recital 54, p. 1]. A [AI Act Annex III, p. 424].", context)
    assert result.removed_count == 0
    assert result.text.endswith("[AI Act Annex III, p. 424].")
    assert [(c.recital, c.annex) for c in result.citations] == [("54", None), (None, "III")]


def test_whole_article_chunk_accepts_any_paragraph() -> None:
    whole = make_chunk("GDPR:art22:seq0", "1. First.\n2. Second.", article="22")
    context = [ScoredChunk(chunk=whole, score=1.0)]
    result = verify_citations("X [GDPR Art. 22(2), p. 1].", context)
    assert result.text == "X [GDPR Art. 22(2), p. 1]."


def test_duplicate_citations_are_listed_once() -> None:
    answer = "A [GDPR Art. 17(1), p. 61]. B [GDPR Art. 17(1), p. 61]."
    assert len(verify_citations(answer, CONTEXT).citations) == 1


def test_citation_for_chunk_uses_first_page() -> None:
    assert citation_for_chunk(ERASURE).format() == "[GDPR Art. 17(1), p. 61]"
