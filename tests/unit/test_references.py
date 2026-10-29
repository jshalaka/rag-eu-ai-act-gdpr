from __future__ import annotations

import pytest

from rag_eu_law.domain.models import Regulation, SearchFilter
from rag_eu_law.retrieval.references import parse_references


@pytest.mark.parametrize(
    ("question", "articles"),
    [
        ("What does GDPR Article 17 say?", ("17",)),
        ("What does Art. 22 say", ("22",)),
        ("Compare Articles 13 and 14", ("13", "14")),
        ("Articles 6, 9 or 10 apply", ("6", "9", "10")),
        ("Is Article 6(1)(a) relevant?", ("6",)),
        ("See Article 5 and article 5 again", ("5",)),
        ("When is a DPIA required?", ()),
        ("The 17 chapters of the regulation", ()),
    ],
)
def test_article_references(question: str, articles: tuple[str, ...]) -> None:
    assert parse_references(question).articles == articles


def test_recital_and_annex_references() -> None:
    refs = parse_references("What do Recital 47 and Annex III say, and Annex I?")
    assert refs.recitals == ("47",)
    assert refs.annexes == ("III", "I")
    assert refs.filters() == [
        SearchFilter(recital="47"),
        SearchFilter(annex="III"),
        SearchFilter(annex="I"),
    ]


def test_lowercase_annex_word_is_not_a_reference() -> None:
    assert parse_references("an annex is attached").annexes == ()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Under the GDPR, what is consent?", Regulation.GDPR),
        ("Under the General Data Protection Regulation", Regulation.GDPR),
        ("What does the AI Act prohibit?", Regulation.AI_ACT),
        ("What does the Artificial Intelligence Act prohibit?", Regulation.AI_ACT),
        ("How does the AI Act refer to the GDPR?", None),
        ("What is consent?", None),
    ],
)
def test_single_regulation(question: str, expected: Regulation | None) -> None:
    assert parse_references(question).single_regulation is expected
