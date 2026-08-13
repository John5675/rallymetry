import io
import json
import logging

from pickleball_vision.config import LogFormat
from pickleball_vision.logging import configure_logging


def test_json_logging_retains_event_and_context() -> None:
    stream = io.StringIO()
    logger = configure_logging(level="INFO", log_format=LogFormat.JSON, stream=stream)

    logger.info("test_event", extra={"context": {"confidence": 0.75}})

    record = json.loads(stream.getvalue())
    assert record["level"] == "info"
    assert record["logger"] == "pickleball_vision"
    assert record["event"] == "test_event"
    assert record["context"] == {"confidence": 0.75}
    assert "timestamp" in record


def test_console_logging_is_human_readable() -> None:
    stream = io.StringIO()
    logger = configure_logging(level="INFO", log_format=LogFormat.CONSOLE, stream=stream)

    logger.info("test_event", extra={"context": {"status": "ok"}})

    assert stream.getvalue() == 'INFO     pickleball_vision: test_event {"status": "ok"}\n'


def teardown_module() -> None:
    logging.getLogger().handlers.clear()
