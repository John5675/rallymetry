from pathlib import Path

import pytest

from pickleball_vision.config import (
    Environment,
    LogFormat,
    PersonDetectionSettings,
    Settings,
)
from pickleball_vision.errors import ConfigurationError, ErrorCode


def test_settings_have_safe_development_defaults() -> None:
    settings = Settings.from_env({})

    assert settings.environment is Environment.DEVELOPMENT
    assert settings.log_level == "INFO"
    assert settings.log_format is LogFormat.JSON
    assert settings.output_dir == Path("output")
    assert settings.person_detection == PersonDetectionSettings()


def test_settings_load_prefixed_environment_values() -> None:
    settings = Settings.from_env(
        {
            "PICKLEBALL_VISION_ENVIRONMENT": "production",
            "PICKLEBALL_VISION_LOG_LEVEL": "warning",
            "PICKLEBALL_VISION_LOG_FORMAT": "console",
            "PICKLEBALL_VISION_OUTPUT_DIR": "~/pickleball-output",
            "PICKLEBALL_VISION_PERSON_MODEL": "custom-person.pt",
            "PICKLEBALL_VISION_PERSON_DEVICE": "CPU",
            "PICKLEBALL_VISION_PERSON_MIN_CONFIDENCE": "0.15",
            "PICKLEBALL_VISION_PERSON_IMAGE_SIZE": "960",
            "PICKLEBALL_VISION_PERSON_IOU_THRESHOLD": "0.6",
            "PICKLEBALL_VISION_PERSON_MAX_DETECTIONS": "250",
        }
    )

    assert settings.environment is Environment.PRODUCTION
    assert settings.log_level == "WARNING"
    assert settings.log_format is LogFormat.CONSOLE
    assert settings.output_dir == Path("~/pickleball-output").expanduser()
    assert settings.person_detection == PersonDetectionSettings(
        model="custom-person.pt",
        device="cpu",
        min_confidence=0.15,
        image_size=960,
        iou_threshold=0.6,
        max_detections=250,
    )


@pytest.mark.parametrize(
    ("environment", "setting"),
    [
        ({"PICKLEBALL_VISION_ENVIRONMENT": "staging"}, "PICKLEBALL_VISION_ENVIRONMENT"),
        ({"PICKLEBALL_VISION_LOG_LEVEL": "verbose"}, "PICKLEBALL_VISION_LOG_LEVEL"),
        ({"PICKLEBALL_VISION_LOG_FORMAT": "xml"}, "PICKLEBALL_VISION_LOG_FORMAT"),
        ({"PICKLEBALL_VISION_OUTPUT_DIR": "  "}, "PICKLEBALL_VISION_OUTPUT_DIR"),
        ({"PICKLEBALL_VISION_PERSON_MODEL": "  "}, "PICKLEBALL_VISION_PERSON_MODEL"),
        ({"PICKLEBALL_VISION_PERSON_DEVICE": "tpu"}, "PICKLEBALL_VISION_PERSON_DEVICE"),
        (
            {"PICKLEBALL_VISION_PERSON_MIN_CONFIDENCE": "1.1"},
            "PICKLEBALL_VISION_PERSON_MIN_CONFIDENCE",
        ),
        (
            {"PICKLEBALL_VISION_PERSON_IMAGE_SIZE": "tiny"},
            "PICKLEBALL_VISION_PERSON_IMAGE_SIZE",
        ),
        (
            {"PICKLEBALL_VISION_PERSON_IOU_THRESHOLD": "-0.1"},
            "PICKLEBALL_VISION_PERSON_IOU_THRESHOLD",
        ),
        (
            {"PICKLEBALL_VISION_PERSON_MAX_DETECTIONS": "0"},
            "PICKLEBALL_VISION_PERSON_MAX_DETECTIONS",
        ),
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
