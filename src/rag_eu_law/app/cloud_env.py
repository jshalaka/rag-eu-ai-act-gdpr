"""Environment for the Streamlit Community Cloud deployment.

Two kinds of settings: *fixed* ones depend on what the cloud install contains (no PyTorch, no
Chroma) and on the host's memory, so secrets must not change them; *defaults* may be overridden.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path


def fixed_settings(repo_root: Path, work_dir: Path) -> dict[str, str]:
    return {
        "RAG_EMBEDDER": "fastembed",
        "RAG_VECTOR_STORE": "memory",  # 1,154 chunks: exact search in memory, no Chroma needed
        "RAG_EMBEDDING_BATCH_SIZE": "2",  # small batches keep peak memory low on a small host
        "RAG_RAW_DIR": str(repo_root / "data" / "raw"),
        "RAG_PROCESSED_DIR": str(work_dir / "processed"),
        "RAG_EMBEDDING_CACHE_DIR": str(work_dir / "models"),
        "RAG_HISTORY_DB_PATH": str(work_dir / "history.sqlite"),
    }


DEFAULT_SETTINGS = {"RAG_LLM_PROVIDER": "openai"}


def apply_cloud_environment(
    secrets: Mapping[str, object],
    environment: MutableMapping[str, str],
    repo_root: Path,
    work_dir: Path,
) -> list[str]:
    """Put secrets, defaults and fixed settings into `environment`.

    Only top-level string secrets are used. Returns the names of secrets that were ignored
    because they would break the cloud install (so the page can tell the user).
    """
    fixed = fixed_settings(repo_root, work_dir)
    for name, default in DEFAULT_SETTINGS.items():
        environment.setdefault(name, default)
    ignored: list[str] = []
    for key, value in secrets.items():
        if not isinstance(value, str):
            continue
        if key in fixed:
            ignored.append(key)
        else:
            environment[str(key)] = value
    environment.update(fixed)
    return sorted(ignored)
