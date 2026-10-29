"""Legal-structure chunking: one article (or paragraph of a long article) per chunk."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_eu_law.domain.constants import MAX_CHUNK_CHARS, MAX_WHOLE_ARTICLE_CHARS
from rag_eu_law.domain.models import Chunk, LegalUnit, Locator, UnitKind, build_chunk_id

NUMBERED_PARAGRAPH = re.compile(r"^(\d+)\.\s")
NUMBERED_DEFINITION = re.compile(r"^\((\d+)\)\s")
LINE_MARKER = re.compile(r"^(?:\(\w{1,4}\)|\d+\.)\s")


@dataclass(frozen=True)
class _Segment:
    """Consecutive lines of a unit that share one paragraph label."""

    paragraph: str | None
    lines: tuple[str, ...]
    pages: tuple[int, ...]


class LegalStructureChunker:
    """`Chunker` that follows the legal structure, never fixed token windows."""

    def __init__(
        self,
        max_whole_article_chars: int = MAX_WHOLE_ARTICLE_CHARS,
        max_chunk_chars: int = MAX_CHUNK_CHARS,
    ) -> None:
        self._max_whole_article_chars = max_whole_article_chars
        self._max_chunk_chars = max_chunk_chars

    def chunk(self, units: list[LegalUnit]) -> list[Chunk]:
        return [chunk for unit in units for chunk in self._chunk_unit(unit)]

    def _chunk_unit(self, unit: LegalUnit) -> list[Chunk]:
        lines = unit.text.split("\n")
        pages = unit.line_pages or (unit.page_start,) * len(lines)
        whole = _Segment(None, tuple(lines), tuple(pages))
        segments = [whole]
        if unit.kind is UnitKind.ARTICLE and len(unit.text) > self._max_whole_article_chars:
            segments = _split_by_paragraph(whole)
        return [
            self._build_chunk(unit, part, sequence)
            for segment in segments
            for sequence, part in enumerate(_pack(segment, self._max_chunk_chars))
        ]

    @staticmethod
    def _build_chunk(unit: LegalUnit, part: _Segment, sequence: int) -> Chunk:
        locator = Locator(
            regulation=unit.regulation,
            article=unit.article,
            paragraph=part.paragraph,
            recital=unit.recital,
            annex=unit.annex,
        )
        return Chunk(
            chunk_id=build_chunk_id(locator, sequence),
            regulation=unit.regulation,
            text=_render(part.lines),
            page_start=min(part.pages),
            page_end=max(part.pages),
            source_file=unit.source_file,
            chapter=unit.chapter,
            section=unit.section,
            article=unit.article,
            article_title=unit.article_title,
            paragraph=part.paragraph,
            recital=unit.recital,
            annex=unit.annex,
        )


def _split_by_paragraph(segment: _Segment) -> list[_Segment]:
    """Split on `1.` paragraphs, else on numbered definitions `(1)`, else keep whole."""
    for pattern in (NUMBERED_PARAGRAPH, NUMBERED_DEFINITION):
        starts = _sequential_starts(segment.lines, pattern)
        if len(starts) >= 2:
            return _cut_at(segment, starts)
    return [segment]


def _sequential_starts(lines: tuple[str, ...], pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    """(line index, label) of markers numbered 1, 2, 3... in order."""
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match and int(match.group(1)) == len(starts) + 1:
            starts.append((index, match.group(1)))
    return starts


def _cut_at(segment: _Segment, starts: list[tuple[int, str]]) -> list[_Segment]:
    """Cut at each marker. Lead-in text before marker 1 stays with the first paragraph."""
    boundaries = [0, *(index for index, _ in starts[1:]), len(segment.lines)]
    return [
        _Segment(label, segment.lines[begin:end], segment.pages[begin:end])
        for (_, label), begin, end in zip(starts, boundaries[:-1], boundaries[1:], strict=True)
    ]


def _pack(segment: _Segment, limit: int) -> list[_Segment]:
    """Greedily pack lines into parts of at most `limit` characters."""
    parts: list[_Segment] = []
    begin, size = 0, 0
    for index, line in enumerate(segment.lines):
        if size and size + len(line) > limit:
            parts.append(_slice(segment, begin, index))
            begin, size = index, 0
        size += len(line) + 1
    parts.append(_slice(segment, begin, len(segment.lines)))
    return parts


def _slice(segment: _Segment, begin: int, end: int) -> _Segment:
    return _Segment(segment.paragraph, segment.lines[begin:end], segment.pages[begin:end])


def _render(lines: tuple[str, ...]) -> str:
    """Join wrapped lines with spaces; keep a newline before list markers."""
    text = ""
    for line in lines:
        if text:
            text += "\n" if LINE_MARKER.match(line) else " "
        text += line
    return text
