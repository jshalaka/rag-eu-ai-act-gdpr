"""Text cleaning: noise removal, ligatures, page numbers, hyphenation, marker lines."""

from __future__ import annotations

import re

from rag_eu_law.domain.models import Page

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
PRIVATE_USE_MARK = chr(0xF02A)  # footnote asterisk glyph in the AI Act title
NOISE_CHARACTERS = ("\u258c", PRIVATE_USE_MARK, "`")
NOISE_LINE = re.compile(r"^(?:\+|View: Official source|OJ: please.*|.*corresponding footnote\.?)$")
MARKER_ONLY_LINE = re.compile(r"^(?:\(\d+\)|\([a-z]{1,4}\)|\d+\.)$")
WORD_PATTERN = re.compile(r"[a-z]+(?:-[a-z]+)*")
TRAILING_FRAGMENT = re.compile(r"([A-Za-z]+)-$")
LEADING_FRAGMENT = re.compile(r"^([a-z]+)")
HYPHENATED_PREFIXES = frozenset({"e", "pre", "re", "self", "non", "anti", "semi", "multi", "start"})
SPACE_RUN = re.compile(r"[ \t]{2,}")


class LegalTextCleaner:
    """`TextCleaner` for EU legal PDFs. Keeps the line structure the parser relies on."""

    def clean(self, pages: list[Page]) -> list[Page]:
        stripped = [page.model_copy(update={"text": _strip_noise(page)}) for page in pages]
        vocabulary = _build_vocabulary(stripped)
        return [
            page.model_copy(update={"text": _restructure(page.text, vocabulary)})
            for page in stripped
        ]


def _strip_noise(page: Page) -> str:
    text = page.text
    for ligature, replacement in LIGATURES.items():
        text = text.replace(ligature, replacement)
    for character in NOISE_CHARACTERS:
        text = text.replace(character, " ")
    lines = [SPACE_RUN.sub(" ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line and not NOISE_LINE.match(line)]
    return "\n".join(_drop_page_number(lines, page.number))


def _drop_page_number(lines: list[str], page_number: int) -> list[str]:
    if lines and lines[-1] == str(page_number):
        return lines[:-1]
    return lines


def _build_vocabulary(pages: list[Page]) -> set[str]:
    return {word for page in pages for word in WORD_PATTERN.findall(page.text.lower())}


def _restructure(text: str, vocabulary: set[str]) -> str:
    lines = text.split("\n")
    return "\n".join(_merge_marker_lines(_repair_hyphenation(lines, vocabulary)))


def _repair_hyphenation(lines: list[str], vocabulary: set[str]) -> list[str]:
    repaired: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        following = lines[index + 1] if index + 1 < len(lines) else None
        if following is not None and _is_wrapped_word(line, following):
            line = _join_wrapped_word(line, following, vocabulary)
            index += 1
        repaired.append(line)
        index += 1
    return repaired


def _is_wrapped_word(line: str, following: str) -> bool:
    return bool(TRAILING_FRAGMENT.search(line) and LEADING_FRAGMENT.match(following))


def _join_wrapped_word(line: str, following: str, vocabulary: set[str]) -> str:
    left = TRAILING_FRAGMENT.search(line)
    right = LEADING_FRAGMENT.match(following)
    assert left is not None and right is not None  # guaranteed by _is_wrapped_word
    left_word, right_word = left.group(1).lower(), right.group(1)
    if _keeps_hyphen(left_word, right_word.lower(), vocabulary):
        return line + following
    return line[:-1] + following


def _keeps_hyphen(left: str, right: str, vocabulary: set[str]) -> bool:
    """Decide whether a line-end hyphen belongs to a compound ("high-risk") or is wrapping."""
    if left + right in vocabulary:
        return False
    return (
        f"{left}-{right}" in vocabulary
        or left in HYPHENATED_PREFIXES
        or (left in vocabulary and right in vocabulary)
    )


def _merge_marker_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    pending_marker: str | None = None
    for line in lines:
        text = line if pending_marker is None else f"{pending_marker} {line}"
        pending_marker = None
        if MARKER_ONLY_LINE.match(text):
            pending_marker = text
        else:
            merged.append(text)
    if pending_marker is not None:
        merged.append(pending_marker)
    return merged
