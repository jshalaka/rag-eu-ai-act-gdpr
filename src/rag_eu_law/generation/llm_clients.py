"""LLM clients: Ollama (default, local) and Anthropic."""

from __future__ import annotations

from typing import Any

import httpx

from rag_eu_law.domain.constants import (
    LLM_MAX_OUTPUT_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    OPENAI_MAX_COMPLETION_TOKENS,
)
from rag_eu_law.domain.exceptions import GenerationError


class OllamaClient:
    """Calls a local Ollama server's chat endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = LLM_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self._model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_OUTPUT_TOKENS},
        }
        try:
            response = self._http.post("/api/chat", json=payload)
            response.raise_for_status()
            content = str(response.json()["message"]["content"]).strip()
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise GenerationError(f"Ollama request failed ({self._model}): {error}") from error
        if not content:
            raise GenerationError(f"Ollama returned an empty answer ({self._model})")
        return content


class AnthropicClient:
    """Calls the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str, client: Any = None) -> None:
        self._model = model
        self._client: Any = client or self._create(api_key)

    @staticmethod
    def _create(api_key: str) -> Any:
        import anthropic  # noqa: PLC0415 - optional 'llm' extra

        return anthropic.Anthropic(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            message = self._client.messages.create(
                model=self._model,
                system=system_prompt,
                max_tokens=LLM_MAX_OUTPUT_TOKENS,
                temperature=LLM_TEMPERATURE,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = "".join(block.text for block in message.content if block.type == "text")
        except Exception as error:
            raise GenerationError(f"Anthropic request failed ({self._model}): {error}") from error
        if not text.strip():
            raise GenerationError(f"Anthropic returned an empty answer ({self._model})")
        return text.strip()


class OpenAIClient:
    """Calls the OpenAI Chat Completions API (reasoning models such as gpt-6-luna).

    Reasoning models take `reasoning_effort` and `max_completion_tokens`, and do not accept a
    custom temperature, so none is sent.
    """

    def __init__(self, api_key: str, model: str, reasoning_effort: str, client: Any = None) -> None:
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._client: Any = client or self._create(api_key)

    @staticmethod
    def _create(api_key: str) -> Any:
        import openai  # noqa: PLC0415 - optional 'llm' extra

        return openai.OpenAI(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                reasoning_effort=self._reasoning_effort,
                max_completion_tokens=OPENAI_MAX_COMPLETION_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
        except Exception as error:
            raise GenerationError(f"OpenAI request failed ({self._model}): {error}") from error
        if not text:
            reason = getattr(choice, "finish_reason", "unknown")
            raise GenerationError(f"OpenAI returned an empty answer ({self._model}, {reason})")
        return text
