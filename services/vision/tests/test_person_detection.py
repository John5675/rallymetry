from collections.abc import Callable
from pathlib import Path

import pytest

from pickleball_vision.person_detection import (
    BoundingBox,
    DetectorMetadata,
    PersonDetection,
    PersonDetectionRun,
)
from pickleball_vision.video import VideoMetadata


def test_person_detection_serialization_retains_raw_observation_provenance() -> None:
    detection = PersonDetection(
        bounding_box=BoundingBox(
            left_px=10.25,
            top_px=20.5,
            right_px=110.75,
            bottom_px=220.0,
        ),
        confidence=0.8125,
        frame_number=17,
        timestamp_s=17 / 29.97,
    )
    run = PersonDetectionRun(
        created_at_utc="2026-08-13T12:00:00+00:00",
        source=VideoMetadata(
            filename="match.mp4",
            path=Path("/video/match.mp4"),
            width=1920,
            height=1080,
            fps=29.97,
            frame_count=100,
            duration=100 / 29.97,
            codec="h264",
        ),
        calibration_path="/output/calibration.json",
        calibration_schema_version=2,
        detector=DetectorMetadata(
            adapter="test_adapter",
            model="test-model",
            device="cpu",
            framework="test",
            framework_version="1.0",
        ),
        configuration={"minimum_confidence": 0.2},
        detections=(detection,),
    )

    serialized = run.as_dict()

    assert serialized["record_type"] == "raw_person_observations"
    assert serialized["coordinate_system"] == {
        "origin": "top_left",
        "x_axis": "right",
        "y_axis": "down",
        "unit": "pixels",
        "frame_numbering": "zero_based",
    }
    assert serialized["detections"] == [
        {
            "frame_number": 17,
            "timestamp_s": 17 / 29.97,
            "bounding_box": {
                "left_px": 10.25,
                "top_px": 20.5,
                "right_px": 110.75,
                "bottom_px": 220.0,
            },
            "confidence": 0.8125,
        }
    ]
    assert "player_id" not in str(serialized)
    assert "court_position" not in str(serialized)


@pytest.mark.parametrize(
    "box",
    [
        lambda: BoundingBox(-1, 0, 10, 10),
        lambda: BoundingBox(0, 0, 0, 10),
        lambda: BoundingBox(0, 0, 10, float("nan")),
    ],
)
def test_invalid_bounding_boxes_are_rejected(box: Callable[[], BoundingBox]) -> None:
    with pytest.raises(ValueError):
        box()


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_invalid_detection_confidence_is_rejected(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        PersonDetection(
            bounding_box=BoundingBox(0, 0, 10, 10),
            confidence=confidence,
            frame_number=0,
            timestamp_s=0,
        )
