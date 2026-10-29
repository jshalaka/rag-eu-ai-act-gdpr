from __future__ import annotations

from pathlib import Path

import pytest

from rag_eu_law.app.bootstrap import ensure_index
from rag_eu_law.domain.exceptions import SourceDocumentError
from tests.ui_support import build_container


def test_builds_the_index_when_empty_and_leaves_it_alone_afterwards(tmp_path: Path) -> None:
    container = build_container(tmp_path)  # builds one index already
    assert ensure_index(container) == 0
    fresh = build_container(tmp_path / "other")
    # Start from nothing: a new in-memory store over the same PDFs.
    fresh.__dict__.pop("vector_store")
    fresh.__dict__.pop("indexing_pipeline")
    assert fresh.vector_store.count() == 0
    built = ensure_index(fresh)
    assert built > 0
    assert fresh.vector_store.count() == built
    assert ensure_index(fresh) == 0


def test_missing_pdfs_are_reported(tmp_path: Path) -> None:
    container = build_container(tmp_path)
    container.__dict__.pop("vector_store")
    container.__dict__.pop("indexing_pipeline")
    for pdf in container.settings.raw_dir.glob("*.pdf"):
        pdf.unlink()
    with pytest.raises(SourceDocumentError):
        ensure_index(container)
