from __future__ import annotations

from rag_eu_law.domain.constants import REFUSAL_MESSAGE
from rag_eu_law.domain.models import ScoredChunk
from rag_eu_law.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from tests.fixtures import erasure_chunk


def test_system_prompt_states_the_grounding_and_refusal_rules() -> None:
    assert "ONLY" in SYSTEM_PROMPT
    assert REFUSAL_MESSAGE in SYSTEM_PROMPT
    assert "[GDPR Art. 6(1), p. 36]" in SYSTEM_PROMPT


def test_user_prompt_labels_each_source_with_its_citation_tag() -> None:
    prompt = build_user_prompt(
        "  What is erasure?  ", [ScoredChunk(chunk=erasure_chunk(), score=1)]
    )
    assert "[SOURCE 1] GDPR Article 17 (Right to erasure), paragraph 1" in prompt
    assert "cite as: [GDPR Art. 17(1), p. 61]" in prompt
    assert prompt.endswith("QUESTION\nWhat is erasure?")
