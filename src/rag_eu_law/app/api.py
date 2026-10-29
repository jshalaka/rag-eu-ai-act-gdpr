"""HTTP API: `POST /ask`. Run with `uvicorn rag_eu_law.app.api:app`."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_eu_law.app.cli import default_container
from rag_eu_law.app.container import Container
from rag_eu_law.domain.constants import (
    API_MAX_QUESTION_CHARS,
    API_MAX_TOP_K,
    API_MIN_QUESTION_CHARS,
)
from rag_eu_law.domain.exceptions import GenerationError, RagError, RetrievalError
from rag_eu_law.domain.models import Answer, Regulation, SearchFilter

logger = logging.getLogger(__name__)

HTTP_BAD_GATEWAY = 502
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_INTERNAL_ERROR = 500


class AskRequest(BaseModel):
    question: str = Field(min_length=API_MIN_QUESTION_CHARS, max_length=API_MAX_QUESTION_CHARS)
    regulation: Regulation | None = Field(default=None, description='"GDPR" or "AI Act"')
    article: str | None = Field(default=None, description="Only this article number")
    top_k: int | None = Field(default=None, ge=1, le=API_MAX_TOP_K)


def create_api(container_factory: Callable[[], Container] = default_container) -> FastAPI:
    """Build the API. The container is created on first use, not at import time."""
    api = FastAPI(title="EU law RAG", description="Cited answers about the GDPR and the AI Act.")
    get_container = lru_cache(maxsize=1)(container_factory)

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.post("/ask")
    def ask(request: AskRequest) -> Answer:
        search_filter = SearchFilter(regulation=request.regulation, article=request.article)
        try:
            return get_container().ask_service.ask(request.question, search_filter, request.top_k)
        except RetrievalError as error:
            raise HTTPException(HTTP_SERVICE_UNAVAILABLE, detail=str(error)) from error
        except GenerationError as error:
            raise HTTPException(HTTP_BAD_GATEWAY, detail=str(error)) from error
        except RagError as error:
            logger.exception("unexpected project error")
            raise HTTPException(HTTP_INTERNAL_ERROR, detail=str(error)) from error

    return api


app = create_api()
