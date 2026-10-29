from __future__ import annotations

import pytest

from rag_eu_law.domain.exceptions import SourceDocumentError
from rag_eu_law.domain.models import Regulation
from rag_eu_law.ingestion.identify import SignatureRegulationDetector
from tests.fixtures import ai_act_pages, build_pages, gdpr_pages


def test_detects_gdpr() -> None:
    assert SignatureRegulationDetector().detect(gdpr_pages()) is Regulation.GDPR


def test_detects_ai_act() -> None:
    assert SignatureRegulationDetector().detect(ai_act_pages()) is Regulation.AI_ACT


def test_unknown_document_is_rejected() -> None:
    with pytest.raises(SourceDocumentError):
        SignatureRegulationDetector().detect(build_pages(["Recipe for soup"], "soup.pdf"))


def test_ambiguous_document_is_rejected() -> None:
    text = "General Data Protection Regulation 2016/679 and the Artificial Intelligence Act"
    with pytest.raises(SourceDocumentError):
        SignatureRegulationDetector().detect(build_pages([text], "both.pdf"))
