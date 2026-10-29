"""Runs ingestion on the real PDFs in data/raw. Skipped when they are not present."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from rag_eu_law.domain.models import Chunk, Regulation
from rag_eu_law.ingestion.chunker import LegalStructureChunker
from rag_eu_law.ingestion.cleaner import LegalTextCleaner
from rag_eu_law.ingestion.identify import SignatureRegulationDetector
from rag_eu_law.ingestion.loader import PyMuPdfLoader
from rag_eu_law.ingestion.parser import LegalStructureParser
from rag_eu_law.ingestion.pipeline import IngestionPipeline

RAW_DIR = Path("data/raw")
pytestmark = pytest.mark.skipif(
    len(list(RAW_DIR.glob("*.pdf"))) < 2, reason="real PDFs not present in data/raw"
)


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    pipeline = IngestionPipeline(
        PyMuPdfLoader(),
        LegalTextCleaner(),
        SignatureRegulationDetector(),
        LegalStructureParser(),
        LegalStructureChunker(),
    )
    return pipeline.ingest_directory(RAW_DIR)


def _articles(chunks: list[Chunk], regulation: Regulation) -> set[str]:
    return {c.article for c in chunks if c.regulation is regulation and c.article}


def test_every_article_and_recital_is_found(chunks: list[Chunk]) -> None:
    assert len(_articles(chunks, Regulation.GDPR)) == 99
    assert len(_articles(chunks, Regulation.AI_ACT)) == 113
    recitals = Counter(c.regulation for c in chunks if c.recital)
    assert recitals[Regulation.GDPR] >= 173
    assert recitals[Regulation.AI_ACT] >= 180
    annexes = {c.annex for c in chunks if c.annex}
    assert len(annexes) == 13


def test_no_extraction_noise_survives(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        assert not any(mark in chunk.text for mark in ("▌", "ﬁ", "ﬂ", "corresponding footnote"))


def test_known_passages_are_on_the_right_pages(chunks: list[Chunk]) -> None:
    erasure = next(c for c in chunks if c.chunk_id == "GDPR:art17:para1:seq0")
    assert "right to obtain from the controller the erasure" in erasure.text
    assert erasure.page_start == 61
    prohibited = next(c for c in chunks if c.chunk_id == "AI_ACT:art5:para1:seq0")
    assert prohibited.article_title == "Prohibited AI Practices"
    assert prohibited.page_start == 181
