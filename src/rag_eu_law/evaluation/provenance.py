"""Capture retrieval inputs for conservative run-compatibility checks."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rag_eu_law.app.config import Settings
from rag_eu_law.domain import constants
from rag_eu_law.evaluation.index_fingerprint import capture_chroma_fingerprint


def _file_hash(path: Path) -> str:
    """Hash file bytes and reject detectable changes during reading."""
    before = path.stat()
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError(f"File changed during provenance capture: {path}")

    return digest.hexdigest()


def _tree_hashes(root: Path, pattern: str = "*") -> dict[str, str]:
    """Fingerprint relative filenames and reject changing file lists."""
    if not root.is_dir():
        raise ValueError(f"Required directory is missing: {root}")

    paths = sorted(
        path for path in root.rglob(pattern)
        if path.is_file()
    )
    if not paths:
        raise ValueError(f"No matching files found in: {root}")

    hashes = {
        path.relative_to(root).as_posix(): _file_hash(path)
        for path in paths
    }

    remaining = {
        path.relative_to(root).as_posix()
        for path in root.rglob(pattern)
        if path.is_file()
    }
    if remaining != set(hashes):
        raise ValueError(f"Directory changed during provenance capture: {root}")

    return hashes


def _package_versions() -> dict[str, str | None]:
    """Record dependencies relevant to retrieval and evaluation."""
    names = (
        "pydantic",
        "pydantic-settings",
        "numpy",
        "chromadb",
        "rank-bm25",
        "sentence-transformers",
        "transformers",
        "torch",
        "fastembed",
    )
    versions = {}

    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None

    return versions


def capture_retrieval_provenance(
    settings: Settings,
    questions_path: Path,
) -> dict:
    """Capture inputs; concurrent writers must be stopped."""
    if settings.vector_store != constants.VECTOR_STORE_CHROMA:
        raise ValueError("Provenance capture currently requires Chroma")

    processed = settings.processed_dir
    vector_directory = processed / constants.CHROMA_DIRNAME

    pdf_paths = sorted(
        path for path in settings.raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        raise ValueError("No source PDFs found")

    # Read existing collection content; opening Chroma can update bookkeeping.
    logical_index = capture_chroma_fingerprint(
        vector_directory,
        constants.CHROMA_COLLECTION,
    )
    physical_files = _tree_hashes(vector_directory)

    # Record SQLite bytes diagnostically. Logical records, metadata and
    # configuration are checked through the collection fingerprint instead.
    sqlite_files = {
        "chroma.sqlite3",
        "chroma.sqlite3-wal",
        "chroma.sqlite3-shm",
        "chroma.sqlite3-journal",
    }

    # Preserve strict checks for persisted search-index artifacts.
    search_index_files = {
        name: digest
        for name, digest in physical_files.items()
        if name not in sqlite_files
    }

    source_root = Path(__file__).resolve().parents[1]

    # Explicitly select public settings, excluding credentials.
    retrieval_settings = {
        "top_k": settings.top_k,
        "candidates_per_retriever": settings.candidates_per_retriever,
        "vector_store": settings.vector_store,
        "embedder": settings.embedder,
        "embedding_model": settings.embedding_model,
        "embedding_batch_size": settings.embedding_batch_size,
        "rerank_enabled": settings.rerank_enabled,
        "rerank_model": (
            settings.rerank_model if settings.rerank_enabled else None
        ),
    }

    identity = {
        "questions_sha256": _file_hash(questions_path),
        "raw_pdfs": {
            path.name: _file_hash(path) for path in pdf_paths
        },
        "chunks_sha256": _file_hash(
            processed / constants.CHUNKS_FILENAME
        ),
        "keyword_index_sha256": _file_hash(
            processed / constants.KEYWORD_INDEX_FILENAME
        ),
        "vector_index": {
            "logical": logical_index,
            "search_index_files": search_index_files,
        },
        "retrieval_settings": retrieval_settings,
        "application_source_files": _tree_hashes(source_root, "*.py"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "machine": platform.machine(),
            "packages": _package_versions(),
        },
    }

    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    return {
        "schema_version": "retrieval-provenance-v2",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "identity_sha256": hashlib.sha256(canonical).hexdigest(),
        "diagnostics": {
            "vector_index_physical_files": physical_files,
        },
        "limitations": [
            "SQLite physical changes are recorded but do not alone determine "
            "logical index compatibility.",
            "Non-SQLite search-index artifacts remain byte-sensitive.",
            "Equal fingerprints do not prove deterministic search behaviour.",
            "Model names and package versions do not pin downloaded weights.",
            "Capture does not establish index-build lineage.",
            "Snapshots are not atomic; concurrent writers must be stopped.",
        ],
    }