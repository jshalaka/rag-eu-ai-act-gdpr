"""Text shared by the dense and keyword indexes."""

from __future__ import annotations

import re

from rag_eu_law.domain.models import Chunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def embedding_text(chunk: Chunk) -> str:
    """Chunk text prefixed with its legal location, so "Article 22" matches by meaning too."""
    return f"{chunk.heading}\n{chunk.text}"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())
