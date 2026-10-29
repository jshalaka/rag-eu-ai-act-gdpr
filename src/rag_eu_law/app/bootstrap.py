"""Start-up helper for hosts that begin with no index (for example Streamlit Community Cloud)."""

from __future__ import annotations

import logging

from rag_eu_law.app.container import Container
from rag_eu_law.ingestion.pipeline import write_chunks

logger = logging.getLogger(__name__)


def ensure_index(container: Container) -> int:
    """Build the search index from the PDFs when the vector store is empty.

    Returns the number of chunks indexed, or 0 when an index already existed.
    """
    if container.vector_store.count() > 0:
        return 0
    settings = container.settings
    chunks = container.ingestion_pipeline.ingest_directory(settings.raw_dir)
    write_chunks(chunks, settings.processed_dir)
    container.indexing_pipeline.index(chunks)
    logger.info("built index at start-up: %d chunks", len(chunks))
    return len(chunks)
