from pathlib import Path

import pytest

from pickleball_vision.config import Environment, LogFormat, Settings
from pickleball_vision.errors import ConfigurationError, ErrorCode


def test_settings_have_safe_development_defaults() -> None:
    settings = Settings.from_env({})

    assert settings.environment is Environment.DEVELOPMENT
    assert settings.log_level == "INFO"
    assert settings.log_format is LogFormat.JSON
    assert settings.output_dir == Path("output")


def test_settings_load_prefixed_environment_values() -> None:
    settings = Settings.from_env(
        {
            "PICKLEBALL_VISION_ENVIRONMENT": "production",
            "PICKLEBALL_VISION_LOG_LEVEL": "warning",
            "PICKLEBALL_VISION_LOG_FORMAT": "console",
            "PICKLEBALL_VISION_OUTPUT_DIR": "~/pickleball-output",
        }
    )

    assert settings.environment is Environment.PRODUCTION
    assert settings.log_level == "WARNING"
    assert settings.log_format is LogFormat.CONSOLE
    assert settings.output_dir == Path("~/pickleball-output").expanduser()


@pytest.mark.parametrize(
    ("environment", "setting"),
    [
        ({"PICKLEBALL_VISION_ENVIRONMENT": "staging"}, "PICKLEBALL_VISION_ENVIRONMENT"),
        ({"PICKLEBALL_VISION_LOG_LEVEL": "verbose"}, "PICKLEBALL_VISION_LOG_LEVEL"),
        ({"PICKLEBALL_VISION_LOG_FORMAT": "xml"}, "PICKLEBALL_VISION_LOG_FORMAT"),
        ({"PICKLEBALL_VISION_OUTPUT_DIR": "  "}, "PICKLEBALL_VISION_OUTPUT_DIR"),
    ],
)
def test_invalid_settings_raise_typed_errors(
    environment: dict[str, str],
    setting: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        Settings.from_env(environment)

    assert raised.value.code is ErrorCode.CONFIGURATION
    assert raised.value.details == {"setting": setting}
