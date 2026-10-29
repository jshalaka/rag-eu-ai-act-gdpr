"""Contract tests: every implementation of an ingestion protocol must pass these."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rag_eu_law.domain.models import Page, Regulation
from rag_eu_law.domain.protocols import Chunker, StructureParser, TextCleaner
from rag_eu_law.ingestion.chunker import LegalStructureChunker
from rag_eu_law.ingestion.cleaner import LegalTextCleaner
from rag_eu_law.ingestion.parser import LegalStructureParser
from tests.fixtures import ai_act_pages, gdpr_pages

CLEANERS: list[Callable[[], TextCleaner]] = [LegalTextCleaner]
PARSERS: list[Callable[[], StructureParser]] = [LegalStructureParser]
CHUNKERS: list[Callable[[], Chunker]] = [LegalStructureChunker]
DOCUMENTS: list[tuple[Regulation, Callable[[], list[Page]]]] = [
    (Regulation.GDPR, gdpr_pages),
    (Regulation.AI_ACT, ai_act_pages),
]


@pytest.mark.parametrize("make_cleaner", CLEANERS)
def test_cleaner_keeps_page_count_and_numbers(make_cleaner: Callable[[], TextCleaner]) -> None:
    pages = gdpr_pages()
    cleaned = make_cleaner().clean(pages)
    assert [p.number for p in cleaned] == [p.number for p in pages]
    assert all(p.source_file == pages[0].source_file for p in cleaned)


@pytest.mark.parametrize("make_cleaner", CLEANERS)
def test_cleaner_is_idempotent(make_cleaner: Callable[[], TextCleaner]) -> None:
    cleaner = make_cleaner()
    once = cleaner.clean(ai_act_pages())
    assert cleaner.clean(once) == once


@pytest.mark.parametrize("make_parser", PARSERS)
@pytest.mark.parametrize(("regulation", "make_pages"), DOCUMENTS)
def test_parser_units_carry_valid_pages(
    make_parser: Callable[[], StructureParser],
    regulation: Regulation,
    make_pages: Callable[[], list[Page]],
) -> None:
    units = make_parser().parse(LegalTextCleaner().clean(make_pages()), regulation)
    assert units
    for unit in units:
        assert unit.page_start <= unit.page_end
        assert unit.regulation is regulation
        assert len(unit.line_pages) == len(unit.text.split("\n"))


@pytest.mark.parametrize("make_chunker", CHUNKERS)
@pytest.mark.parametrize(("regulation", "make_pages"), DOCUMENTS)
def test_chunker_output_is_deterministic_and_unique(
    make_chunker: Callable[[], Chunker],
    regulation: Regulation,
    make_pages: Callable[[], list[Page]],
) -> None:
    units = LegalStructureParser().parse(LegalTextCleaner().clean(make_pages()), regulation)
    first, second = make_chunker().chunk(units), make_chunker().chunk(units)
    assert first == second
    assert len({c.chunk_id for c in first}) == len(first)
    assert all(c.text.strip() and c.page_start <= c.page_end for c in first)
