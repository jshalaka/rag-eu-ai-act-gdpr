"""PDF loading with PyMuPDF. Drops footnotes and superscripts using font geometry."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from rag_eu_law.domain.constants import (
    FOOTNOTE_TOP_TOLERANCE_PT,
    FOOTNOTE_ZONE_START,
    SMALL_FONT_RATIO,
)
from rag_eu_law.domain.exceptions import SourceDocumentError
from rag_eu_law.domain.models import Page

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Span:
    text: str
    size: float


@dataclass(frozen=True)
class _Line:
    top: float
    spans: tuple[_Span, ...]


class PyMuPdfLoader:
    """`DocumentLoader` that returns one `Page` per PDF page, footnotes removed."""

    def __init__(
        self, small_font_ratio: float = SMALL_FONT_RATIO, footnote_zone: float = FOOTNOTE_ZONE_START
    ) -> None:
        self._small_font_ratio = small_font_ratio
        self._footnote_zone = footnote_zone

    def load(self, path: Path) -> list[Page]:
        document = self._open(path)
        pages = [
            Page(source_file=path.name, number=index, text=self._page_text(page))
            for index, page in enumerate(document, start=1)
        ]
        if not any(page.text.strip() for page in pages):
            raise SourceDocumentError(f"No extractable text in {path} (scanned PDF?)")
        logger.info("loaded %s: %d pages", path.name, len(pages))
        return pages

    @staticmethod
    def _open(path: Path) -> Any:
        if not path.is_file():
            raise SourceDocumentError(f"PDF not found: {path}")
        try:
            return pymupdf.open(path)
        except (RuntimeError, ValueError) as error:
            raise SourceDocumentError(f"Unreadable PDF {path}: {error}") from error

    def _page_text(self, page: pymupdf.Page) -> str:
        lines = _extract_lines(page)
        body_size = _body_font_size(lines)
        small_limit = body_size * self._small_font_ratio
        cutoff = self._footnote_cutoff(lines, small_limit, page.rect.height)
        texts = [
            _line_text(line, small_limit) for line in lines if cutoff is None or line.top < cutoff
        ]
        return "\n".join(text for text in texts if text)

    def _footnote_cutoff(
        self, lines: list[_Line], small_limit: float, page_height: float
    ) -> float | None:
        """Top of the first footnote line (it starts with a small number), if any."""
        zone_top = page_height * self._footnote_zone
        starts = [
            line.top
            for line in lines
            if line.top >= zone_top and line.spans and line.spans[0].size <= small_limit
        ]
        return min(starts) - FOOTNOTE_TOP_TOLERANCE_PT if starts else None


def _extract_lines(page: pymupdf.Page) -> list[_Line]:
    blocks: list[dict[str, Any]] = page.get_text("dict", sort=True)["blocks"]
    return [
        _Line(top=float(line["bbox"][1]), spans=_spans(line))
        for block in blocks
        for line in block.get("lines", [])
    ]


def _spans(line: dict[str, Any]) -> tuple[_Span, ...]:
    return tuple(_Span(text=span["text"], size=float(span["size"])) for span in line["spans"])


def _body_font_size(lines: list[_Line]) -> float:
    sizes: Counter[float] = Counter()
    for line in lines:
        for span in line.spans:
            sizes[round(span.size, 1)] += len(span.text)
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _line_text(line: _Line, small_limit: float) -> str:
    return "".join(span.text for span in line.spans if span.size > small_limit).rstrip()
