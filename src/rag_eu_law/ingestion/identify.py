"""Identify which regulation a PDF contains from its first pages (no hardcoded filenames)."""

from __future__ import annotations

from rag_eu_law.domain.constants import IDENTIFICATION_PAGE_COUNT, REGULATION_SIGNATURES
from rag_eu_law.domain.exceptions import SourceDocumentError
from rag_eu_law.domain.models import Page, Regulation


class SignatureRegulationDetector:
    """`RegulationDetector` matching signature phrases in the opening pages."""

    def __init__(
        self,
        signatures: dict[str, tuple[str, ...]] = REGULATION_SIGNATURES,
        page_count: int = IDENTIFICATION_PAGE_COUNT,
    ) -> None:
        self._signatures = {Regulation(name): phrases for name, phrases in signatures.items()}
        self._page_count = page_count

    def detect(self, pages: list[Page]) -> Regulation:
        opening = " ".join(page.text for page in pages[: self._page_count]).lower()
        matches = [
            regulation
            for regulation, phrases in self._signatures.items()
            if all(phrase in opening for phrase in phrases)
        ]
        source = pages[0].source_file if pages else "<empty document>"
        if len(matches) != 1:
            raise SourceDocumentError(
                f"Cannot identify the regulation in {source}: matched {[m.value for m in matches]}"
            )
        return matches[0]
