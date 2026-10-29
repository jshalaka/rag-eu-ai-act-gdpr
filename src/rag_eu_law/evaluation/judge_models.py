"""Structured outputs for semantic evaluation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClaimAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str = Field(min_length=1)
    verdict: Literal[
        "supported",
        "contradicted",
        "insufficient_evidence",
        "uncertain",
    ]
    evidence_chunk_ids: list[str]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_evidence_for_decisive_verdict(self) -> "ClaimAssessment":
        if self.verdict in {"supported", "contradicted"}:
            if not self.evidence_chunk_ids:
                raise ValueError(
                    "Supported or contradicted claims need evidence chunk IDs"
                )
        return self


class JudgeAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer_behavior: Literal[
        "answered",
        "partially_answered",
        "insufficient_evidence",
        "clarification_required",
    ]
    relevance: Literal["relevant", "partially_relevant", "irrelevant", "uncertain"]
    claims: list[ClaimAssessment]
    reason: str = Field(min_length=1)