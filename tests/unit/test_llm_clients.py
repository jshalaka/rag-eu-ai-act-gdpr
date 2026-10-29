from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from rag_eu_law.domain.exceptions import GenerationError
from rag_eu_law.generation.llm_clients import AnthropicClient, OllamaClient, OpenAIClient


def _ollama(handler: httpx.MockTransport) -> OllamaClient:
    return OllamaClient("http://ollama.test", "qwen-test", transport=handler)


def test_ollama_sends_chat_request_and_returns_content() -> None:
    seen: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"message": {"content": "  hello  "}})

    assert _ollama(httpx.MockTransport(handle)).complete("sys", "usr") == "hello"
    assert seen["url"] == "http://ollama.test/api/chat"
    assert '"model":"qwen-test"' in str(seen["body"]).replace(" ", "")


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, json={"unexpected": True}),
        httpx.Response(200, json={"message": {"content": "   "}}),
        httpx.Response(200, text="not json"),
    ],
)
def test_ollama_failures_become_generation_errors(response: httpx.Response) -> None:
    with pytest.raises(GenerationError):
        _ollama(httpx.MockTransport(lambda _: response)).complete("s", "u")


def test_ollama_connection_error_is_wrapped() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(GenerationError, match="Ollama request failed"):
        _ollama(httpx.MockTransport(refuse)).complete("s", "u")


class FakeMessages:
    def __init__(self, blocks: list[SimpleNamespace] | Exception) -> None:
        self._blocks = blocks
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        if isinstance(self._blocks, Exception):
            raise self._blocks
        return SimpleNamespace(content=self._blocks)


def _anthropic(blocks: list[SimpleNamespace] | Exception) -> tuple[AnthropicClient, FakeMessages]:
    messages = FakeMessages(blocks)
    return AnthropicClient(
        "key", "claude-test", client=SimpleNamespace(messages=messages)
    ), messages


def test_anthropic_joins_text_blocks() -> None:
    blocks = [
        SimpleNamespace(type="text", text="Hello "),
        SimpleNamespace(type="tool_use", text="ignored"),
        SimpleNamespace(type="text", text="world"),
    ]
    client, messages = _anthropic(blocks)
    assert client.complete("sys", "usr") == "Hello world"
    assert messages.kwargs["system"] == "sys"
    assert messages.kwargs["model"] == "claude-test"


def test_anthropic_errors_and_empty_answers_are_wrapped() -> None:
    with pytest.raises(GenerationError, match="Anthropic request failed"):
        _anthropic(RuntimeError("rate limited"))[0].complete("s", "u")
    with pytest.raises(GenerationError, match="empty"):
        _anthropic([SimpleNamespace(type="text", text=" ")])[0].complete("s", "u")


class FakeCompletions:
    def __init__(self, content: str | None, finish_reason: str = "stop") -> None:
        self._content = content
        self._finish_reason = finish_reason
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=self._finish_reason)]
        )


def _openai(
    content: str | None, finish_reason: str = "stop"
) -> tuple[OpenAIClient, FakeCompletions]:
    completions = FakeCompletions(content, finish_reason)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIClient("key", "gpt-test", "low", client=fake), completions


def test_openai_sends_reasoning_parameters_and_returns_text() -> None:
    client, completions = _openai("  answer  ")
    assert client.complete("sys", "usr") == "answer"
    assert completions.kwargs["model"] == "gpt-test"
    assert completions.kwargs["reasoning_effort"] == "low"
    assert "max_completion_tokens" in completions.kwargs
    assert "temperature" not in completions.kwargs
    assert completions.kwargs["messages"][0] == {"role": "system", "content": "sys"}  # type: ignore[index]


def test_openai_empty_answer_reports_finish_reason() -> None:
    with pytest.raises(GenerationError, match="length"):
        _openai(None, finish_reason="length")[0].complete("s", "u")


def test_openai_errors_are_wrapped() -> None:
    class Broken:
        def create(self, **_: object) -> None:
            raise RuntimeError("quota exceeded")

    fake = SimpleNamespace(chat=SimpleNamespace(completions=Broken()))
    client = OpenAIClient("key", "gpt-test", "low", client=fake)
    with pytest.raises(GenerationError, match="quota exceeded"):
        client.complete("s", "u")
