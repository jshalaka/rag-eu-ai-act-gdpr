"""Ingestion pipeline: discover PDFs, then load → clean → parse → chunk."""

from __future__ import annotations

import logging
from pathlib import Path

from rag_eu_law.domain.constants import CHUNKS_FILENAME, PDF_GLOB
from rag_eu_law.domain.exceptions import IngestionError, SourceDocumentError
from rag_eu_law.domain.models import Chunk
from rag_eu_law.domain.protocols import (
    Chunker,
    DocumentLoader,
    RegulationDetector,
    StructureParser,
    TextCleaner,
)

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Depends only on protocols; concrete classes are wired in the composition root."""

    def __init__(
        self,
        loader: DocumentLoader,
        cleaner: TextCleaner,
        detector: RegulationDetector,
        parser: StructureParser,
        chunker: Chunker,
    ) -> None:
        self._loader = loader
        self._cleaner = cleaner
        self._detector = detector
        self._parser = parser
        self._chunker = chunker

    def ingest_directory(self, raw_dir: Path) -> list[Chunk]:
        paths = sorted(raw_dir.glob(PDF_GLOB))
        if not paths:
            raise SourceDocumentError(f"No PDF files found in {raw_dir}")
        chunks = [chunk for path in paths for chunk in self.ingest_file(path)]
        _ensure_unique_ids(chunks)
        return chunks

    def ingest_file(self, path: Path) -> list[Chunk]:
        pages = self._loader.load(path)
        regulation = self._detector.detect(pages)
        units = self._parser.parse(self._cleaner.clean(pages), regulation)
        chunks = self._chunker.chunk(units)
        logger.info("%s → %s: %d units, %d chunks", path.name, regulation, len(units), len(chunks))
        return chunks


def write_chunks(chunks: list[Chunk], processed_dir: Path) -> Path:
    """Write chunks as JSON lines in a stable order, so re-ingestion gives identical bytes."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    target = processed_dir / CHUNKS_FILENAME
    lines = [chunk.model_dump_json() for chunk in sorted(chunks, key=lambda c: c.chunk_id)]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def read_chunks(path: Path) -> list[Chunk]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise IngestionError(f"Cannot read chunks file {path}: {error}") from error
    return [Chunk.model_validate_json(line) for line in text.splitlines() if line.strip()]


def _ensure_unique_ids(chunks: list[Chunk]) -> None:
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise IngestionError(f"Duplicate chunk_id: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
