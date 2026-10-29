from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_eu_law.domain import exceptions
from rag_eu_law.domain.constants import DISCLAIMER
from rag_eu_law.domain.models import (
    Answer,
    Citation,
    Locator,
    Regulation,
    build_chunk_id,
)


def test_chunk_id_is_deterministic() -> None:
    locator = Locator(regulation=Regulation.GDPR, article="17", paragraph="1")
    assert build_chunk_id(locator) == build_chunk_id(locator)
    assert build_chunk_id(locator) == "GDPR:art17:para1:seq0"


def test_chunk_id_differs_by_locator_and_sequence() -> None:
    article = Locator(regulation=Regulation.AI_ACT, article="5")
    recital = Locator(regulation=Regulation.AI_ACT, recital="5")
    assert build_chunk_id(article) != build_chunk_id(recital)
    assert build_chunk_id(article, 0) != build_chunk_id(article, 1)


def test_annex_chunk_id() -> None:
    locator = Locator(regulation=Regulation.AI_ACT, annex="III")
    assert build_chunk_id(locator, 2) == "AI_ACT:annexIII:seq2"


@pytest.mark.parametrize(
    ("citation", "expected"),
    [
        (
            Citation(
                chunk_id="x",
                regulation=Regulation.GDPR,
                page=36,
                article="6",
                paragraph="1",
                point="a",
            ),
            "[GDPR Art. 6(1)(a), p. 36]",
        ),
        (
            Citation(chunk_id="x", regulation=Regulation.GDPR, page=8, recital="47"),
            "[GDPR Recital 47, p. 8]",
        ),
        (
            Citation(chunk_id="x", regulation=Regulation.AI_ACT, page=430, annex="III"),
            "[AI Act Annex III, p. 430]",
        ),
        (
            Citation(chunk_id="x", regulation=Regulation.AI_ACT, page=181, article="5"),
            "[AI Act Art. 5, p. 181]",
        ),
        (
            Citation(chunk_id="x", regulation=Regulation.AI_ACT, page=1),
            "[AI Act unknown location, p. 1]",
        ),
    ],
)
def test_citation_format(citation: Citation, expected: str) -> None:
    assert citation.format() == expected


def test_models_are_frozen() -> None:
    citation = Citation(chunk_id="x", regulation=Regulation.GDPR, page=1)
    with pytest.raises(ValidationError):
        citation.page = 2  # type: ignore[misc]


def test_page_number_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Citation(chunk_id="x", regulation=Regulation.GDPR, page=0)


def test_answer_carries_disclaimer_by_default() -> None:
    answer = Answer(answer="a", citations=[], retrieved_chunks=[], latency_ms=1.0)
    assert answer.disclaimer == DISCLAIMER


def test_exceptions_share_one_base() -> None:
    for error_type in (
        exceptions.IngestionError,
        exceptions.SourceDocumentError,
        exceptions.StructureParseError,
        exceptions.IndexingError,
        exceptions.RetrievalError,
        exceptions.GenerationError,
        exceptions.ConfigurationError,
    ):
        assert issubclass(error_type, exceptions.RagError)
    assert issubclass(exceptions.SourceDocumentError, exceptions.IngestionError)
