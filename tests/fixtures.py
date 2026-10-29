"""Small text fixtures that mimic the two real layouts (not the full PDFs)."""

from __future__ import annotations

from pathlib import Path

from rag_eu_law.domain.models import Chunk, Page, Regulation

GDPR_FILE = "gdpr-sample.pdf"
AI_ACT_FILE = "ai-act-sample.pdf"

GDPR_PAGES = [
    "General Data Protection Regulation (GDPR) -\nfull text\n"
    "REGULATION (EU) 2016/679 OF THE EUROPEAN PARLIAMENT\n"
    "View: Official source\n"
    "(1) The protection of natural persons is a fundamental right.\n"
    "(2) The principles should respect the ﬁnal rights of per-\n"
    "sons and a high-\nrisk approach.\n"
    "HAVE ADOPTED THIS REGULATION:\n"
    "CHAPTER I General provisions\n"
    "Article 1 Subject-matter and objectives\n"
    "1. This Regulation lays down rules relating to protection.\n"
    "2. This Regulation protects fundamental rights.\n"
    "1",
    "Article 2 Material scope\n"
    "1. This Regulation applies to the processing of personal data.\n"
    "Article 290 TFEU should be delegated to the Commission.\n"
    "2. This Regulation does not apply to purely personal activity.\n"
    "Article 3 Territorial scope of the\n"
    "Union\n"
    "For the purposes of this Regulation:\n"
    "(a) processing means any operation;\n"
    "(b) controller means the entity that decides.\n"
    "CHAPTER II Principles\n"
    "Article 4 Definitions\n"
    "For the purposes of this Regulation:\n"
    "1. ‘personal data’ means any information;\n"
    "2. ‘processing’ means any operation.\n"
    "This Regulation shall be binding in its entirety.\n"
    "Done at Brussels, 27 April 2016.\n"
    "2",
]

AI_ACT_PAGES = [
    "Artificial Intelligence Act\nPosition of the European Parliament\n"
    "Whereas:\n"
    "(1)\nThe purpose of this Regulation ▌ is to improve the internal market.\n"
    "(2)\nAI systems can be deployed widely.\n"
    "HAVE ADOPTED THIS REGULATION:\n"
    "CHAPTER I\nGENERAL PROVISIONS\n"
    "Article 1\nSubject matter`\n"
    "1.\nThe purpose of this Regulation is to improve the functioning.\n"
    "2.\nThis Regulation lays down:\n"
    "(a)\nharmonised rules for AI systems;\n"
    "(b)\nprohibitions of certain AI practices.",
    "CHAPTER II\nPROHIBITED ARTIFICIAL INTELLIGENCE\nPRACTICES\n"
    "Section 1\nGeneral rules\n"
    "Article 2\nScope\n"
    "This Regulation applies to providers.\n"
    "Article 3\nDefinitions\n"
    "For the purposes of this Regulation, the following definitions apply:\n"
    "(1)\n‘AI system’ means a machine-based system;\n"
    "(2)\n‘risk’ means the combination of probability and severity;\n"
    "This Regulation shall be binding in its entirety and directly applicable.\n"
    "ANNEX I\nList of Union harmonisation legislation\n"
    "1.\nDirective 2006/42/EC on machinery;\n"
    "2.\nDirective 2009/48/EC on toys.",
]


def build_pages(texts: list[str], source_file: str) -> list[Page]:
    return [Page(source_file=source_file, number=n, text=t) for n, t in enumerate(texts, start=1)]


def gdpr_pages() -> list[Page]:
    return build_pages(GDPR_PAGES, GDPR_FILE)


def ai_act_pages() -> list[Page]:
    return build_pages(AI_ACT_PAGES, AI_ACT_FILE)


def make_chunk(  # noqa: PLR0917 - test helper, keyword use is the norm
    chunk_id: str,
    text: str,
    regulation: Regulation = Regulation.GDPR,
    article: str | None = None,
    recital: str | None = None,
    title: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        regulation=regulation,
        text=text,
        page_start=1,
        page_end=1,
        source_file="sample.pdf",
        article=article,
        article_title=title,
        recital=recital,
    )


SAMPLE_CHUNKS = [
    make_chunk(
        "GDPR:art17:seq0",
        "The data subject shall have the right to obtain erasure of personal data.",
        article="17",
        title="Right to erasure",
    ),
    make_chunk(
        "GDPR:art22:seq0",
        "The data subject shall have the right not to be subject to automated decision-making.",
        article="22",
        title="Automated individual decision-making",
    ),
    make_chunk(
        "AI_ACT:art5:seq0",
        "The following AI practices shall be prohibited: subliminal manipulation.",
        regulation=Regulation.AI_ACT,
        article="5",
        title="Prohibited AI Practices",
    ),
    make_chunk(
        "AI_ACT:rec54:seq0",
        "Biometric data processing must respect data protection rules.",
        regulation=Regulation.AI_ACT,
        recital="54",
    ),
]


def erasure_chunk() -> Chunk:
    """GDPR Art. 17(1) with sub-points, spanning two pages."""
    return Chunk(
        chunk_id="GDPR:art17:para1:seq0",
        regulation=Regulation.GDPR,
        text=(
            "1. The data subject shall have the right to erasure where:\n"
            "(a) the data are no longer necessary;\n(b) consent is withdrawn."
        ),
        page_start=61,
        page_end=62,
        source_file="gdpr.pdf",
        article="17",
        article_title="Right to erasure",
        paragraph="1",
    )


SAMPLE_PDF_PAGES = [
    [
        "General Data Protection Regulation (GDPR)",
        "REGULATION (EU) 2016/679 OF THE EUROPEAN PARLIAMENT",
        "(1) The protection of natural persons is a fundamental right.",
        "(2) The processing of personal data should serve mankind.",
        "HAVE ADOPTED THIS REGULATION:",
        "CHAPTER I General provisions",
        "Article 1 Subject-matter and objectives",
        "1. This Regulation lays down rules relating to the protection of personal data.",
        "2. This Regulation protects fundamental rights and freedoms.",
    ],
    [
        "Article 2 Material scope",
        "1. This Regulation applies to the processing of personal data by automated means.",
        "2. This Regulation does not apply to purely personal or household activity.",
    ],
    [
        "CHAPTER II Rights of the data subject",
        "Article 3 Right to erasure",
        "1. The data subject shall have the right to obtain the erasure of personal data",
        "without undue delay where the data are no longer necessary.",
        "This Regulation shall be binding in its entirety.",
    ],
]


def write_sample_pdf(path: Path) -> None:
    """A real 3-page PDF in the GDPR layout, small enough to ingest in milliseconds."""
    import pymupdf  # noqa: PLC0415 - only this helper needs it

    document = pymupdf.open()
    for number, lines in enumerate(SAMPLE_PDF_PAGES, start=1):
        page = document.new_page()
        for row, line in enumerate(lines):
            page.insert_text((72, 80 + 16 * row), line, fontsize=10)
        page.insert_text((290, 800), str(number), fontsize=10)
    document.save(path)
