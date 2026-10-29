from __future__ import annotations

from pathlib import Path

import pytest

from rag_eu_law.domain.exceptions import IngestionError, SourceDocumentError
from rag_eu_law.domain.models import Chunk, Page, Regulation
from rag_eu_law.ingestion.chunker import LegalStructureChunker
from rag_eu_law.ingestion.cleaner import LegalTextCleaner
from rag_eu_law.ingestion.identify import SignatureRegulationDetector
from rag_eu_law.ingestion.parser import LegalStructureParser
from rag_eu_law.ingestion.pipeline import (
    IngestionPipeline,
    read_chunks,
    write_chunks,
)
from tests.fixtures import AI_ACT_FILE, GDPR_FILE, ai_act_pages, gdpr_pages


class FakeLoader:
    """Serves fixture pages by filename, so the pipeline runs without real PDFs."""

    def __init__(self) -> None:
        self._pages = {GDPR_FILE: gdpr_pages(), AI_ACT_FILE: ai_act_pages()}

    def load(self, path: Path) -> list[Page]:
        return self._pages[path.name]


def _pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        FakeLoader(),
        LegalTextCleaner(),
        SignatureRegulationDetector(),
        LegalStructureParser(),
        LegalStructureChunker(),
    )


def _raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in (GDPR_FILE, AI_ACT_FILE):
        (raw / name).write_bytes(b"")
    return raw


def test_ingests_every_pdf_found(tmp_path: Path) -> None:
    chunks = _pipeline().ingest_directory(_raw_dir(tmp_path))
    assert {chunk.regulation for chunk in chunks} == {Regulation.GDPR, Regulation.AI_ACT}
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_empty_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SourceDocumentError, match="No PDF files"):
        _pipeline().ingest_directory(tmp_path)


def test_ingestion_is_idempotent(tmp_path: Path) -> None:
    raw = _raw_dir(tmp_path)
    first = write_chunks(_pipeline().ingest_directory(raw), tmp_path / "out").read_bytes()
    second = write_chunks(_pipeline().ingest_directory(raw), tmp_path / "out").read_bytes()
    assert first == second


def test_chunks_round_trip_through_jsonl(tmp_path: Path) -> None:
    chunks = _pipeline().ingest_directory(_raw_dir(tmp_path))
    path = write_chunks(chunks, tmp_path / "out")
    assert sorted(read_chunks(path), key=lambda c: c.chunk_id) == sorted(
        chunks, key=lambda c: c.chunk_id
    )


def test_unreadable_chunks_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        read_chunks(tmp_path / "absent.jsonl")


def test_duplicate_chunk_ids_are_rejected(tmp_path: Path) -> None:
    class DuplicatingChunker:
        def chunk(self, units: list[object]) -> list[Chunk]:
            real = LegalStructureChunker().chunk(units)  # type: ignore[arg-type]
            return [*real, real[0]]

    pipeline = IngestionPipeline(
        FakeLoader(),
        LegalTextCleaner(),
        SignatureRegulationDetector(),
        LegalStructureParser(),
        DuplicatingChunker(),  # type: ignore[arg-type]
    )
    with pytest.raises(IngestionError, match="Duplicate chunk_id"):
        pipeline.ingest_directory(_raw_dir(tmp_path))
