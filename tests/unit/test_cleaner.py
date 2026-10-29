from __future__ import annotations

from rag_eu_law.domain.models import Page
from rag_eu_law.ingestion.cleaner import LegalTextCleaner
from tests.fixtures import gdpr_pages


def _clean(text: str, number: int = 1) -> str:
    return LegalTextCleaner().clean([Page(source_file="x.pdf", number=number, text=text)])[0].text


def test_ligatures_are_expanded() -> None:
    assert _clean("the ﬁnal oﬃce") == "the final office"


def test_noise_characters_and_lines_are_removed() -> None:
    cleaned = _clean(
        "For ▌ AI systems`\n+\nOJ: please insert the number\n"
        "and complete the corresponding footnote."
    )
    assert cleaned == "For AI systems"


def test_trailing_page_number_is_removed_only_when_it_matches() -> None:
    assert _clean("Some text\n7", number=7) == "Some text"
    assert _clean("Some text\n7", number=8) == "Some text\n7"


def test_wrapping_hyphen_is_joined_when_fused_word_exists() -> None:
    text = "the protection of per-\nsons and personsx persons"
    assert "persons and" in _clean(text).replace("\n", " ")


def test_compound_hyphen_is_kept() -> None:
    text = "a high-risk system\nand a high-\nrisk model"
    assert "high-risk model" in _clean(text)


def test_hyphen_after_single_letter_prefix_is_kept() -> None:
    assert "e-recruiting" in _clean("such as e-\nrecruiting practices")


def test_marker_only_lines_merge_with_next_line() -> None:
    assert (
        _clean("(a)\nthe first point;\n1.\nA paragraph") == "(a) the first point;\n1. A paragraph"
    )


def test_dangling_marker_is_kept() -> None:
    assert _clean("text\n(b)") == "text\n(b)"


def test_gdpr_fixture_end_to_end() -> None:
    pages = LegalTextCleaner().clean(gdpr_pages())
    assert "View: Official source" not in pages[0].text
    assert not pages[0].text.endswith("\n1")
    assert "final rights of persons" in pages[0].text
