"""Legal structure parser: recitals, chapters, sections, articles and annexes.

Headings are accepted only when numbered in sequence, which rejects cross-references that
happen to start a line (for example "Article 290 TFEU should be delegated...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto

from rag_eu_law.domain.exceptions import StructureParseError
from rag_eu_law.domain.models import LegalUnit, Page, Regulation, UnitKind

ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
RECITAL_LINE = re.compile(r"^\((\d+)\)\s+\S")
ENACTING_FORMULA = re.compile(r"^HAVE ADOPTED THIS REGULATION")
TITLE_CONNECTORS = frozenset({"the", "of", "and", "or", "to", "for", "in", "on", "by", "with", "a"})
CLOSING_FORMULA = re.compile(r"^This Regulation shall be binding in its entirety")


@dataclass(frozen=True)
class LayoutProfile:
    """How a document lays out its headings."""

    title_inline: bool  # "Article 17 Right to erasure" vs "Article 17" + title on next line


DEFAULT_PROFILES: dict[Regulation, LayoutProfile] = {
    Regulation.GDPR: LayoutProfile(title_inline=True),
    Regulation.AI_ACT: LayoutProfile(title_inline=False),
}


class _Mode(Enum):
    PREAMBLE = auto()
    RECITALS = auto()
    ARTICLES = auto()
    ANNEXES = auto()
    CLOSED = auto()


@dataclass
class _Draft:
    """A unit under construction."""

    kind: UnitKind
    start_page: int
    meta: dict[str, str | None]
    lines: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class _HeadingSpec:
    keyword: str
    identifier_pattern: str
    upper_case_title: bool = False


CHAPTER_SPEC = _HeadingSpec("CHAPTER", r"[IVXLC]+", upper_case_title=True)
SECTION_SPEC = _HeadingSpec("Section", r"\d+")
ARTICLE_SPEC = _HeadingSpec("Article", r"\d+")


@dataclass(frozen=True)
class _Heading:
    identifier: str
    title: str | None
    lines_used: int


def roman_to_int(numeral: str) -> int:
    total = 0
    for current, following in zip(numeral, numeral[1:] + " ", strict=False):
        value = ROMAN_VALUES[current]
        total += -value if ROMAN_VALUES.get(following, 0) > value else value
    return total


class LegalStructureParser:
    """`StructureParser` producing one `LegalUnit` per recital, article and annex."""

    def __init__(self, profiles: dict[Regulation, LayoutProfile] | None = None) -> None:
        self._profiles = DEFAULT_PROFILES if profiles is None else profiles

    def parse(self, pages: list[Page], regulation: Regulation) -> list[LegalUnit]:
        profile = self._profiles.get(regulation)
        if profile is None:
            raise StructureParseError(f"No layout profile for {regulation.value}")
        run = _ParseRun(regulation, profile, pages)
        units = run.execute()
        _validate(units, regulation)
        return units


class _ParseRun:
    """One pass over a document. Holds the mutable state, so the parser itself stays stateless."""

    def __init__(self, regulation: Regulation, profile: LayoutProfile, pages: list[Page]) -> None:
        self._regulation = regulation
        self._profile = profile
        self._source_file = pages[0].source_file if pages else ""
        self._lines = [(page.number, text) for page in pages for text in page.text.split("\n")]
        self._mode = _Mode.PREAMBLE
        self._units: list[LegalUnit] = []
        self._draft: _Draft | None = None
        self._chapter: str | None = None
        self._section: str | None = None
        self._last = {"recital": 0, "chapter": 0, "section": 0, "article": 0, "annex": 0}

    def execute(self) -> list[LegalUnit]:
        index = 0
        while index < len(self._lines):
            index += self._handle_line(index)
        self._close_draft()
        return self._units

    def _handle_line(self, index: int) -> int:
        used = self._try_structural_line(index)
        if used:
            return used
        page, line = self._lines[index]
        self._append(page, line)
        return 1

    def _try_structural_line(self, index: int) -> int:
        page, line = self._lines[index]
        if ENACTING_FORMULA.match(line):
            self._close_draft()
            self._mode = _Mode.ARTICLES
            return 1
        if CLOSING_FORMULA.match(line) and self._mode is _Mode.ARTICLES:
            self._close_draft()
            self._mode = _Mode.CLOSED
            return 1
        return (
            self._try_recital(page, line)
            or self._try_chapter(index)
            or self._try_section(index)
            or self._try_article(index)
            or self._try_annex(index)
        )

    def _try_recital(self, page: int, line: str) -> int:
        if self._mode not in (_Mode.PREAMBLE, _Mode.RECITALS):
            return 0
        match = RECITAL_LINE.match(line)
        if not match or int(match.group(1)) != self._last["recital"] + 1:
            return 0
        self._close_draft()
        self._mode = _Mode.RECITALS
        self._last["recital"] += 1
        self._draft = _Draft(UnitKind.RECITAL, page, {"recital": match.group(1)})
        self._append(page, line.split(None, 1)[0] + " " + line.split(None, 1)[1])
        return 1

    def _try_chapter(self, index: int) -> int:
        heading = _read_heading(CHAPTER_SPEC, self._lines, index, self._profile)
        if heading is None or roman_to_int(heading.identifier) != self._last["chapter"] + 1:
            return 0
        self._close_draft()
        self._mode = _Mode.ARTICLES
        self._last["chapter"] += 1
        self._last["section"] = 0
        self._section = None
        self._chapter = f"{heading.identifier} {heading.title}".strip()
        return heading.lines_used

    def _try_section(self, index: int) -> int:
        if self._mode is not _Mode.ARTICLES:
            return 0
        heading = _read_heading(SECTION_SPEC, self._lines, index, self._profile)
        if heading is None or int(heading.identifier) != self._last["section"] + 1:
            return 0
        self._close_draft()
        self._last["section"] += 1
        self._section = f"{heading.identifier} {heading.title}".strip()
        return heading.lines_used

    def _try_article(self, index: int) -> int:
        if self._mode is not _Mode.ARTICLES:
            return 0
        page = self._lines[index][0]
        heading = _read_heading(ARTICLE_SPEC, self._lines, index, self._profile)
        if heading is None or int(heading.identifier) != self._last["article"] + 1:
            return 0
        self._close_draft()
        self._last["article"] += 1
        meta = {
            "article": heading.identifier,
            "article_title": heading.title,
            "chapter": self._chapter,
            "section": self._section,
        }
        self._draft = _Draft(UnitKind.ARTICLE, page, meta)
        return heading.lines_used

    def _try_annex(self, index: int) -> int:
        if self._mode in (_Mode.PREAMBLE, _Mode.RECITALS):
            return 0
        page, line = self._lines[index]
        match = re.match(r"^ANNEX ([IVXLC]+)$", line)
        if not match or roman_to_int(match.group(1)) <= self._last["annex"]:
            return 0
        self._close_draft()
        self._mode = _Mode.ANNEXES
        self._last["annex"] = roman_to_int(match.group(1))
        self._draft = _Draft(UnitKind.ANNEX, page, {"annex": match.group(1)})
        return 1

    def _append(self, page: int, line: str) -> None:
        if self._draft is not None and self._mode is not _Mode.CLOSED:
            self._draft.lines.append(line)
            self._draft.pages.append(page)

    def _close_draft(self) -> None:
        draft, self._draft = self._draft, None
        if draft is None or not draft.lines:
            return
        self._units.append(
            LegalUnit(
                regulation=self._regulation,
                kind=draft.kind,
                text="\n".join(draft.lines),
                page_start=draft.start_page,
                page_end=draft.pages[-1],
                source_file=self._source_file,
                chapter=draft.meta.get("chapter"),
                section=draft.meta.get("section"),
                article=draft.meta.get("article"),
                article_title=draft.meta.get("article_title"),
                recital=draft.meta.get("recital"),
                annex=draft.meta.get("annex"),
                line_pages=tuple(draft.pages),
            )
        )


def _read_heading(
    spec: _HeadingSpec, lines: list[tuple[int, str]], index: int, profile: LayoutProfile
) -> _Heading | None:
    """Read `KEYWORD ID [title]` with the title inline or on the following line(s)."""
    line = lines[index][1]
    if profile.title_inline:
        match = re.match(rf"^{spec.keyword} ({spec.identifier_pattern})\s+(\S.*)$", line)
        first_title, first_offset = (match.group(2), 1) if match else ("", 1)
    else:
        match = re.match(rf"^{spec.keyword} ({spec.identifier_pattern})$", line)
        first_title, first_offset = _following_line(lines, index + 1), 2
    if not match:
        return None
    title, extra = _extend_title(first_title, lines, index + first_offset, spec.upper_case_title)
    used = first_offset - (0 if first_title else 1) + extra
    return _Heading(match.group(1), title or None, used)


def _following_line(lines: list[tuple[int, str]], index: int) -> str:
    return lines[index][1] if index < len(lines) else ""


def _extend_title(
    title: str, lines: list[tuple[int, str]], index: int, upper_case_title: bool
) -> tuple[str, int]:
    """Append wrapped title lines that follow `title`. Returns the title and lines consumed."""
    consumed = 0
    while index + consumed < len(lines):
        candidate = lines[index + consumed][1]
        if not _continues_title(title, candidate, upper_case_title):
            break
        title = f"{title} {candidate}"
        consumed += 1
    return title, consumed


def _continues_title(title: str, candidate: str, upper_case_title: bool) -> bool:
    if not title or not candidate:
        return False
    if upper_case_title and candidate.isupper():
        return True
    ends_with_connector = title.rsplit(maxsplit=1)[-1].lower().rstrip(",") in TITLE_CONNECTORS
    return candidate[0].islower() or ends_with_connector


def _validate(units: list[LegalUnit], regulation: Regulation) -> None:
    articles = [unit for unit in units if unit.kind is UnitKind.ARTICLE]
    recitals = [unit for unit in units if unit.kind is UnitKind.RECITAL]
    if not articles or not recitals:
        raise StructureParseError(
            f"{regulation.value}: found {len(articles)} articles and {len(recitals)} recitals"
        )
