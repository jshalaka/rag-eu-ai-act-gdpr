from __future__ import annotations

import pytest

from rag_eu_law.domain.exceptions import StructureParseError
from rag_eu_law.domain.models import LegalUnit, Regulation, UnitKind
from rag_eu_law.ingestion.cleaner import LegalTextCleaner
from rag_eu_law.ingestion.parser import LegalStructureParser, roman_to_int
from tests.fixtures import ai_act_pages, build_pages, gdpr_pages


def _units(regulation: Regulation) -> list[LegalUnit]:
    raw = gdpr_pages() if regulation is Regulation.GDPR else ai_act_pages()
    return LegalStructureParser().parse(LegalTextCleaner().clean(raw), regulation)


def _by_kind(units: list[LegalUnit], kind: UnitKind) -> list[LegalUnit]:
    return [unit for unit in units if unit.kind is kind]


@pytest.mark.parametrize(("numeral", "value"), [("I", 1), ("IV", 4), ("IX", 9), ("XIII", 13)])
def test_roman_to_int(numeral: str, value: int) -> None:
    assert roman_to_int(numeral) == value


def test_gdpr_recitals_and_articles_are_found() -> None:
    units = _units(Regulation.GDPR)
    assert [u.recital for u in _by_kind(units, UnitKind.RECITAL)] == ["1", "2"]
    assert [u.article for u in _by_kind(units, UnitKind.ARTICLE)] == ["1", "2", "3", "4"]


def test_gdpr_inline_titles_and_wrapped_title() -> None:
    articles = _by_kind(_units(Regulation.GDPR), UnitKind.ARTICLE)
    assert articles[0].article_title == "Subject-matter and objectives"
    assert articles[2].article_title == "Territorial scope of the Union"


def test_cross_reference_line_is_not_a_heading() -> None:
    article_two = _by_kind(_units(Regulation.GDPR), UnitKind.ARTICLE)[1]
    assert "Article 290 TFEU should be delegated" in article_two.text


def test_chapter_is_attached_to_articles() -> None:
    articles = _by_kind(_units(Regulation.GDPR), UnitKind.ARTICLE)
    assert articles[0].chapter == "I General provisions"
    assert articles[3].chapter == "II Principles"


def test_closing_formula_is_cut_from_last_article() -> None:
    last = _by_kind(_units(Regulation.GDPR), UnitKind.ARTICLE)[-1]
    assert "binding in its entirety" not in last.text
    assert "Done at Brussels" not in last.text


def test_pages_are_tracked_per_unit_and_line() -> None:
    articles = _by_kind(_units(Regulation.GDPR), UnitKind.ARTICLE)
    assert (articles[0].page_start, articles[0].page_end) == (1, 1)
    assert articles[1].page_start == 2
    assert len(articles[1].line_pages) == len(articles[1].text.split("\n"))


def test_ai_act_next_line_titles_and_multi_line_chapter() -> None:
    units = _units(Regulation.AI_ACT)
    articles = _by_kind(units, UnitKind.ARTICLE)
    assert [a.article_title for a in articles] == ["Subject matter", "Scope", "Definitions"]
    assert articles[1].chapter == "II PROHIBITED ARTIFICIAL INTELLIGENCE PRACTICES"
    assert articles[1].section == "1 General rules"


def test_ai_act_recitals_annex_and_markers() -> None:
    units = _units(Regulation.AI_ACT)
    recitals = _by_kind(units, UnitKind.RECITAL)
    assert recitals[0].text.startswith("(1) The purpose of this Regulation is")
    annex = _by_kind(units, UnitKind.ANNEX)[0]
    assert annex.annex == "I"
    assert annex.text.splitlines()[0] == "List of Union harmonisation legislation"


def test_document_without_structure_is_rejected() -> None:
    pages = build_pages(["Just some prose without any legal structure."], "x.pdf")
    with pytest.raises(StructureParseError):
        LegalStructureParser().parse(pages, Regulation.GDPR)


def test_missing_profile_is_rejected() -> None:
    with pytest.raises(StructureParseError):
        LegalStructureParser(profiles={}).parse(gdpr_pages(), Regulation.GDPR)
