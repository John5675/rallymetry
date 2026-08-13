from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from pickleball_vision.config import PersonDetectionSettings
from pickleball_vision.detectors.ultralytics_person import (
    UltralyticsPersonDetector,
    resolve_inference_device,
)
from pickleball_vision.video import Image


class FakeTensor:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> object:
        return self._values


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = FakeTensor(
            [
                [10.0, 20.0, 110.0, 220.0],
                [30.0, 40.0, 130.0, 240.0],
                [50.0, 60.0, 150.0, 260.0],
            ]
        )
        self.conf = FakeTensor([0.91, 0.95, 0.49])
        self.cls = FakeTensor([0, 1, 0])


class FakeResult:
    def __init__(self) -> None:
        self.boxes = FakeBoxes()


class FakeModel:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    def predict(
        self,
        *,
        source: Image,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
        classes: list[int],
        max_det: int,
        verbose: bool,
    ) -> Sequence[FakeResult]:
        self.arguments = {
            "source_shape": source.shape,
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "device": device,
            "classes": classes,
            "max_det": max_det,
            "verbose": verbose,
        }
        return [FakeResult()]


def test_adapter_requests_only_people_and_defensively_filters_results() -> None:
    model = FakeModel()
    settings = PersonDetectionSettings(
        model="fake.pt",
        device="cpu",
        min_confidence=0.5,
        image_size=960,
        iou_threshold=0.6,
        max_detections=123,
    )
    detector = UltralyticsPersonDetector(
        settings,
        model=model,
        effective_device="cpu",
        framework_version="test",
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame, frame_number=9, timestamp_s=0.3)

    assert len(detections) == 1
    assert detections[0].bounding_box.left_px == pytest.approx(10.0)
    assert detections[0].bounding_box.bottom_px == pytest.approx(220.0)
    assert detections[0].confidence == pytest.approx(0.91)
    assert detections[0].frame_number == 9
    assert detections[0].timestamp_s == pytest.approx(0.3)
    assert model.arguments == {
        "source_shape": (480, 640, 3),
        "conf": 0.5,
        "iou": 0.6,
        "imgsz": 960,
        "device": "cpu",
        "classes": [0],
        "max_det": 123,
        "verbose": False,
    }


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [(True, True, "cuda:0"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_selection_has_cpu_fallback(
    cuda_available: bool,
    mps_available: bool,
    expected: str,
) -> None:
    assert (
        resolve_inference_device(
            "auto",
            cuda_available=cuda_available,
            mps_available=mps_available,
        )
        == expected
    )


def test_explicit_device_is_preserved() -> None:
    assert resolve_inference_device("cpu", cuda_available=True, mps_available=True) == "cpu"
