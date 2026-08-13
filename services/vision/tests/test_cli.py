import json

import pytest

from pickleball_vision.cli import EXIT_OK, EXIT_USAGE_ERROR, main
from pickleball_vision.config import ENV_PREFIX


def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for suffix in ("ENVIRONMENT", "LOG_LEVEL", "LOG_FORMAT", "OUTPUT_DIR"):
        monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)


def test_doctor_reports_valid_foundation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)

    exit_code = main(["doctor"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    log_record = json.loads(captured.err)
    assert exit_code == EXIT_OK
    assert report["service"] == "pickleball-vision"
    assert report["status"] == "ok"
    assert report["configuration"]["environment"] == "development"
    assert log_record["event"] == "foundation_check_complete"


def test_invalid_configuration_has_stable_error_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    monkeypatch.setenv("PICKLEBALL_VISION_LOG_FORMAT", "invalid")

    exit_code = main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "error [configuration_error]" in captured.err


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == EXIT_OK
    assert "doctor" in captured.out
