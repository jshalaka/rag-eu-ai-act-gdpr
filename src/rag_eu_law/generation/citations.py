"""Parse, verify and canonicalise citations in a generated answer.

The model may only cite what it was shown. Each `[GDPR Art. 6(1)(a), p. 36]` tag is matched
against the retrieved chunks; tags that match nothing are removed, near-misses are repaired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_eu_law.domain.models import Chunk, Citation, Regulation, ScoredChunk

REGULATION_LABELS = "|".join(re.escape(regulation.value) for regulation in Regulation)
CITATION_TAG = re.compile(
    rf"\[(?P<regulation>{REGULATION_LABELS}) "
    r"(?:Art\. ?(?P<article>\d+)(?P<parts>(?:\([^)\s]{1,6}\))*)"
    r"|Recital (?P<recital>\d+)"
    r"|Annex (?P<annex>[IVXLC]+)), "
    r"p\. ?(?P<page>\d+)\]"
)
PART = re.compile(r"\(([^)]+)\)")


@dataclass(frozen=True)
class VerifiedAnswer:
    text: str
    citations: list[Citation]
    removed_count: int


def citation_for_chunk(chunk: Chunk) -> Citation:
    """The citation a reader would use for the whole chunk."""
    return Citation(
        chunk_id=chunk.chunk_id,
        regulation=chunk.regulation,
        page=chunk.page_start,
        article=chunk.article,
        paragraph=chunk.paragraph,
        recital=chunk.recital,
        annex=chunk.annex,
    )


def verify_citations(answer: str, context: list[ScoredChunk]) -> VerifiedAnswer:
    """Rewrite valid tags in canonical form, drop invalid ones, and list the valid citations."""
    chunks = [item.chunk for item in context]
    found: dict[str, Citation] = {}
    removed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal removed
        citation = _resolve(match, chunks)
        if citation is None:
            removed += 1
            return ""
        found.setdefault(citation.format(), citation)
        return citation.format()

    text = CITATION_TAG.sub(replace, answer)
    return VerifiedAnswer(_tidy(text), list(found.values()), removed)


def _resolve(match: re.Match[str], chunks: list[Chunk]) -> Citation | None:
    regulation = Regulation(match["regulation"])
    parts = PART.findall(match["parts"] or "")
    paragraph = parts[0] if parts else None
    point = parts[1] if len(parts) > 1 else None
    for chunk in chunks:
        if chunk.regulation is regulation and _same_location(match, chunk, paragraph):
            page = int(match["page"])
            return Citation(
                chunk_id=chunk.chunk_id,
                regulation=regulation,
                page=page if chunk.page_start <= page <= chunk.page_end else chunk.page_start,
                article=chunk.article,
                paragraph=paragraph if paragraph and _has_paragraph(chunk, paragraph) else None,
                point=point if point and _has_point(chunk, point) else None,
                recital=chunk.recital,
                annex=chunk.annex,
            )
    return None


def _same_location(match: re.Match[str], chunk: Chunk, paragraph: str | None) -> bool:
    if match["article"]:
        return chunk.article == match["article"] and _paragraph_fits(chunk, paragraph)
    if match["recital"]:
        return chunk.recital == match["recital"]
    return chunk.annex == match["annex"]


def _paragraph_fits(chunk: Chunk, paragraph: str | None) -> bool:
    """A chunk that carries a paragraph label must match it; a whole-article chunk fits any."""
    return paragraph is None or chunk.paragraph is None or chunk.paragraph == paragraph


def _has_paragraph(chunk: Chunk, paragraph: str) -> bool:
    return (
        chunk.paragraph == paragraph
        or re.search(rf"(?m)^{re.escape(paragraph)}\.\s", chunk.text) is not None
    )


def _has_point(chunk: Chunk, point: str) -> bool:
    return re.search(rf"(?m)^\({re.escape(point)}\)\s", chunk.text) is not None


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()
