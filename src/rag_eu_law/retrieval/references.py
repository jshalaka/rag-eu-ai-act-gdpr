"""Find explicit legal references ("Article 22", "Recital 47", "Annex III") in a question."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_eu_law.domain.models import Regulation, SearchFilter

NUMBER_LIST = r"(\d+(?:\s*(?:,|and|or|&)\s*\d+)*)"
ARTICLE_PATTERN = re.compile(rf"\b(?:Articles?|Arts?\.?)\s*{NUMBER_LIST}", re.IGNORECASE)
RECITAL_PATTERN = re.compile(rf"\bRecitals?\s*{NUMBER_LIST}", re.IGNORECASE)
ANNEX_PATTERN = re.compile(r"\b[Aa]nnex(?:es)?\s+([IVXLC]+)\b")
DIGITS = re.compile(r"\d+")
REGULATION_NAMES: dict[Regulation, re.Pattern[str]] = {
    Regulation.GDPR: re.compile(r"\bGDPR\b|general data protection regulation", re.IGNORECASE),
    Regulation.AI_ACT: re.compile(r"\bAI Act\b|\bAIA\b|artificial intelligence act", re.IGNORECASE),
}


@dataclass(frozen=True)
class QueryReferences:
    regulations: frozenset[Regulation]
    articles: tuple[str, ...]
    recitals: tuple[str, ...]
    annexes: tuple[str, ...]

    @property
    def single_regulation(self) -> Regulation | None:
        """The regulation if the question names exactly one."""
        return next(iter(self.regulations)) if len(self.regulations) == 1 else None

    def filters(self) -> list[SearchFilter]:
        """One filter per explicitly referenced article, recital or annex."""
        return [
            *(SearchFilter(article=value) for value in self.articles),
            *(SearchFilter(recital=value) for value in self.recitals),
            *(SearchFilter(annex=value) for value in self.annexes),
        ]


def parse_references(question: str) -> QueryReferences:
    return QueryReferences(
        regulations=frozenset(
            regulation
            for regulation, pattern in REGULATION_NAMES.items()
            if pattern.search(question)
        ),
        articles=_numbers(ARTICLE_PATTERN, question),
        recitals=_numbers(RECITAL_PATTERN, question),
        annexes=tuple(dict.fromkeys(ANNEX_PATTERN.findall(question))),
    )


def _numbers(pattern: re.Pattern[str], question: str) -> tuple[str, ...]:
    found = [number for match in pattern.findall(question) for number in DIGITS.findall(match)]
    return tuple(dict.fromkeys(found))
