"""Ask a question and keep the run, so answers can be reviewed, rated and reused as tests."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rag_eu_law.app.service import AskService
from rag_eu_law.domain.constants import (
    DEFAULT_USER,
    INDEX_VERSION_LENGTH,
    UNKNOWN_INDEX_VERSION,
)
from rag_eu_law.domain.models import Answer, RunRecord, SearchFilter
from rag_eu_law.domain.protocols import RunStore


@dataclass(frozen=True)
class RecordedAnswer:
    record: RunRecord
    answer: Answer


def index_version(chunks_path: Path) -> str:
    """Short fingerprint of the ingested chunks, so runs can be tied to the data they used."""
    try:
        digest = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    except OSError:
        return UNKNOWN_INDEX_VERSION
    return digest[:INDEX_VERSION_LENGTH]


class HistoryService:
    def __init__(
        self, ask_service: AskService, run_store: RunStore, model: str, index_version_id: str
    ) -> None:
        self._ask_service = ask_service
        self._run_store = run_store
        self._model = model
        self._index_version = index_version_id

    @property
    def model(self) -> str:
        return self._model

    @property
    def index_version(self) -> str:
        return self._index_version

    def ask(
        self,
        question: str,
        user: str = DEFAULT_USER,
        search_filter: SearchFilter | None = None,
    ) -> RecordedAnswer:
        answer = self._ask_service.ask(question, search_filter)
        record = RunRecord(
            run_id=uuid.uuid4().hex,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            user=user or DEFAULT_USER,
            question=question,
            answer=answer.answer,
            citations=[citation.format() for citation in answer.citations],
            retrieved_chunk_ids=[item.chunk.chunk_id for item in answer.retrieved_chunks],
            refused=not answer.citations,
            latency_ms=answer.latency_ms,
            model=self._model,
            index_version=self._index_version,
            regulation_filter=search_filter.regulation.value
            if search_filter and search_filter.regulation
            else None,
            article_filter=search_filter.article if search_filter else None,
        )
        self._run_store.save(record)
        return RecordedAnswer(record=record, answer=answer)

    def rate(self, run_id: str, rating: int, note: str | None = None) -> None:
        self._run_store.rate(run_id, rating, note or None)

    def recent(self, limit: int, user: str | None = None) -> list[RunRecord]:
        return self._run_store.list_runs(limit, user)
