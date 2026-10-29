from __future__ import annotations

from rag_eu_law.domain.models import LegalUnit, Regulation, UnitKind
from rag_eu_law.ingestion.chunker import LegalStructureChunker


def _article(text: str, **overrides: str) -> LegalUnit:
    lines = text.split("\n")
    return LegalUnit(
        regulation=Regulation.GDPR,
        kind=UnitKind.ARTICLE,
        text=text,
        page_start=5,
        page_end=6,
        source_file="g.pdf",
        article="7",
        article_title="Conditions for consent",
        line_pages=tuple(5 if i < len(lines) // 2 else 6 for i in range(len(lines))),
        **overrides,
    )


def test_short_article_is_one_chunk() -> None:
    chunks = LegalStructureChunker().chunk([_article("1. First.\n2. Second.")])
    assert len(chunks) == 1
    assert chunks[0].paragraph is None
    assert chunks[0].chunk_id == "GDPR:art7:seq0"
    assert chunks[0].text == "1. First.\n2. Second."


def test_long_article_splits_by_numbered_paragraph() -> None:
    text = "Lead-in.\n1. First paragraph\ncontinues here.\n2. Second.\n3. Third."
    chunks = LegalStructureChunker(max_whole_article_chars=10).chunk([_article(text)])
    assert [c.paragraph for c in chunks] == ["1", "2", "3"]
    assert chunks[0].text.startswith("Lead-in.")
    assert chunks[0].chunk_id == "GDPR:art7:para1:seq0"
    assert chunks[0].heading == "GDPR Article 7 (Conditions for consent), paragraph 1"


def test_definitions_split_by_parenthesised_numbers() -> None:
    text = "Definitions apply:\n(1) ‘a’ means x;\n(2) ‘b’ means y;"
    chunks = LegalStructureChunker(max_whole_article_chars=5).chunk([_article(text)])
    assert [c.paragraph for c in chunks] == ["1", "2"]


def test_single_marker_does_not_split() -> None:
    chunks = LegalStructureChunker(max_whole_article_chars=5).chunk([_article("1. a\n3. b")])
    assert [c.paragraph for c in chunks] == [None]


def test_out_of_sequence_number_stays_inside_its_paragraph() -> None:
    chunks = LegalStructureChunker(max_whole_article_chars=5).chunk([_article("1. a\n3. b\n2. c")])
    assert [c.paragraph for c in chunks] == ["1", "2"]
    assert "3. b" in chunks[0].text


def test_oversized_paragraph_is_packed_into_sequenced_parts() -> None:
    text = "\n".join(f"line number {i} with some words" for i in range(10))
    chunks = LegalStructureChunker(max_chunk_chars=80).chunk([_article(text)])
    assert len(chunks) > 1
    assert [c.chunk_id.rsplit(":", 1)[1] for c in chunks] == [f"seq{i}" for i in range(len(chunks))]


def test_page_range_follows_the_lines_of_each_chunk() -> None:
    text = "1. First\ncontinues\n2. Second\nends"
    chunks = LegalStructureChunker(max_whole_article_chars=5).chunk([_article(text)])
    assert (chunks[0].page_start, chunks[0].page_end) == (5, 5)
    assert (chunks[1].page_start, chunks[1].page_end) == (6, 6)


def test_list_markers_keep_their_own_line() -> None:
    chunks = LegalStructureChunker().chunk(
        [_article("1. Intro:\n(a) first\nwrapped;\n(b) second.")]
    )
    assert chunks[0].text == "1. Intro:\n(a) first wrapped;\n(b) second."


def test_recital_chunk_metadata() -> None:
    unit = LegalUnit(
        regulation=Regulation.AI_ACT,
        kind=UnitKind.RECITAL,
        text="(3) Text.",
        page_start=9,
        page_end=9,
        source_file="a.pdf",
        recital="3",
    )
    chunk = LegalStructureChunker().chunk([unit])[0]
    assert chunk.chunk_id == "AI_ACT:rec3:seq0"
    assert chunk.heading == "AI Act Recital 3"
    assert chunk.page_start == 9


def test_annex_heading() -> None:
    unit = LegalUnit(
        regulation=Regulation.AI_ACT,
        kind=UnitKind.ANNEX,
        text="Title\n1. Item",
        page_start=1,
        page_end=1,
        source_file="a.pdf",
        annex="III",
    )
    assert LegalStructureChunker().chunk([unit])[0].heading == "AI Act Annex III"
