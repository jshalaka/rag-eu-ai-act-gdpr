from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from rag_eu_law.domain.exceptions import SourceDocumentError
from rag_eu_law.ingestion.loader import PyMuPdfLoader

BODY_SIZE = 12
FOOTNOTE_MARKER_SIZE = 7


def _make_pdf(path: Path, with_footnote: bool = True) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), "First body line", fontsize=BODY_SIZE)
    page.insert_text((72, 130), "Second body line", fontsize=BODY_SIZE)
    if with_footnote:
        page.insert_text((72, 700), "1", fontsize=FOOTNOTE_MARKER_SIZE)
        page.insert_text((90, 700), "OJ C 229, 31.7.2012, p. 90.", fontsize=BODY_SIZE)
    document.new_page().insert_text((72, 100), "Page two text", fontsize=BODY_SIZE)
    document.save(path)


def test_loads_one_page_per_pdf_page(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    pages = PyMuPdfLoader().load(pdf)
    assert [(p.number, p.source_file) for p in pages] == [(1, "doc.pdf"), (2, "doc.pdf")]
    assert "First body line" in pages[0].text
    assert pages[1].text == "Page two text"


def test_footnote_block_is_dropped(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    text = PyMuPdfLoader().load(pdf)[0].text
    assert "OJ C 229" not in text
    assert text.splitlines() == ["First body line", "Second body line"]


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SourceDocumentError, match="not found"):
        PyMuPdfLoader().load(tmp_path / "absent.pdf")


def test_unreadable_file_is_rejected(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_text("not a pdf")
    with pytest.raises(SourceDocumentError):
        PyMuPdfLoader().load(broken)


def test_pdf_without_text_is_rejected(tmp_path: Path) -> None:
    blank = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(blank)
    with pytest.raises(SourceDocumentError, match="No extractable text"):
        PyMuPdfLoader().load(blank)
