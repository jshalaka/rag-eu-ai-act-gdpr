from __future__ import annotations

import json
import logging
import sys

import pytest

from rag_eu_law.app.logging_config import JsonFormatter, configure_logging


def test_formatter_emits_json_with_extra_fields() -> None:
    record = logging.LogRecord("demo", logging.INFO, __file__, 1, "indexed %d", (3,), None)
    record.chunks = 3
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "indexed 3"
    assert payload["level"] == "INFO"
    assert payload["chunks"] == 3


def test_formatter_includes_exception_text() -> None:
    try:
        raise ValueError("bad")
    except ValueError:
        record = logging.LogRecord("demo", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    assert "ValueError: bad" in json.loads(JsonFormatter().format(record))["exception"]


def test_configure_logging_writes_json_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("info")
    logging.getLogger("demo").info("hello")
    assert json.loads(capsys.readouterr().err.splitlines()[-1])["message"] == "hello"
    logging.getLogger().handlers.clear()
