import json
import shutil
from pathlib import Path

import cv2
import pytest

from pickleball_vision.config import PersonDetectionSettings
from pickleball_vision.errors import OutputWriteError
from pickleball_vision.person_detection import (
    BoundingBox,
    DetectorMetadata,
    PersonDetection,
)
from pickleball_vision.person_detection_pipeline import detect_people_in_video
from pickleball_vision.video import Image


class BroadFakeDetector:
    @property
    def metadata(self) -> DetectorMetadata:
        return DetectorMetadata(
            adapter="broad_fake",
            model="none",
            device="cpu",
            framework="test",
            framework_version="1",
        )

    def detect(
        self,
        frame: Image,
        *,
        frame_number: int,
        timestamp_s: float,
    ) -> tuple[PersonDetection, ...]:
        del frame
        return tuple(
            PersonDetection(
                bounding_box=BoundingBox(
                    left_px=2 + index * 12,
                    top_px=3,
                    right_px=10 + index * 12,
                    bottom_px=40,
                ),
                confidence=0.9 - index * 0.05,
                frame_number=frame_number,
                timestamp_s=timestamp_s,
            )
            for index in range(5)
        )


def test_detection_pipeline_writes_broad_source_coordinate_artifacts(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "person-detection"
    artifacts = detect_people_in_video(
        synthetic_video,
        calibration_path=synthetic_calibration,
        output_dir=output_dir,
        settings=PersonDetectionSettings(model="fake.pt", device="cpu"),
        detector=BroadFakeDetector(),
    )

    assert artifacts.processed_frame_count == 12
    assert artifacts.detection_count == 60
    assert artifacts.detections_path.is_file()
    assert artifacts.annotated_video_path.is_file()
    assert artifacts.summary_path.is_file()

    detections = json.loads(artifacts.detections_path.read_text(encoding="utf-8"))
    assert detections["record_type"] == "raw_person_observations"
    assert len(detections["detections"]) == 60
    assert detections["detections"][0]["frame_number"] == 0
    assert detections["detections"][-1]["frame_number"] == 11
    assert detections["calibration"]["usage"] == ("validated_provenance_only_no_person_projection")
    assert "player_id" not in artifacts.detections_path.read_text(encoding="utf-8")

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["statistics"]["processed_frames"] == 12
    assert summary["statistics"]["total_detections"] == 60
    assert summary["statistics"]["detections_per_frame"]["maximum"] == 5
    assert summary["statistics"]["frames_without_detections"] == 0

    capture = cv2.VideoCapture(str(artifacts.annotated_video_path))
    try:
        assert capture.isOpened()
        assert round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 96
        assert round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 64
        assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    finally:
        capture.release()


def test_detection_pipeline_never_overwrites_the_source_video(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "annotated.mp4"
    shutil.copyfile(synthetic_video, source_path)
    source_size = source_path.stat().st_size

    with pytest.raises(OutputWriteError, match="overwrite the source video"):
        detect_people_in_video(
            source_path,
            calibration_path=synthetic_calibration,
            output_dir=tmp_path,
            settings=PersonDetectionSettings(model="fake.pt", device="cpu"),
            detector=BroadFakeDetector(),
        )

    assert source_path.stat().st_size == source_size
