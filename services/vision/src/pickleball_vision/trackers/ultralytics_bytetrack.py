"""Ultralytics ByteTrack adapter over persisted person detections."""

from __future__ import annotations

import importlib
import math
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from pickleball_vision.config import PlayerTrackingSettings
from pickleball_vision.errors import PlayerTrackingInputError
from pickleball_vision.person_detection import BoundingBox
from pickleball_vision.player_tracking import (
    IndexedDetection,
    RawTrackerObservation,
    TrackerMetadata,
)


def _framework_version() -> str | None:
    try:
        return version("ultralytics")
    except PackageNotFoundError:
        return None


class UltralyticsByteTracker:
    """Keep Ultralytics result conventions outside the stable tracking contract."""

    def __init__(self, settings: PlayerTrackingSettings, *, fps: float) -> None:
        track_buffer_frames = max(1, math.ceil(settings.track_buffer_seconds * fps))
        args = SimpleNamespace(
            track_high_thresh=settings.track_high_threshold,
            track_low_thresh=settings.track_low_threshold,
            new_track_thresh=settings.new_track_threshold,
            match_thresh=settings.match_threshold,
            track_buffer=track_buffer_frames,
            fuse_score=True,
        )
        try:
            tracker_module = cast(Any, importlib.import_module("ultralytics.trackers.byte_tracker"))
            results_module = cast(Any, importlib.import_module("ultralytics.engine.results"))
            self._boxes_type = results_module.Boxes
            self._tracker = tracker_module.BYTETracker(args)
        except Exception as error:
            raise PlayerTrackingInputError(
                f"unable to initialize Ultralytics ByteTrack: {error}"
            ) from error
        self._metadata = TrackerMetadata(
            adapter="ultralytics_bytetrack",
            implementation="ByteTrack",
            framework="ultralytics",
            framework_version=_framework_version(),
            configuration={
                **settings.as_dict(),
                "track_buffer_frames": track_buffer_frames,
                "fuse_detection_score": True,
            },
        )

    @property
    def metadata(self) -> TrackerMetadata:
        return self._metadata

    def update(
        self,
        *,
        frame_number: int,
        timestamp_s: float,
        detections: tuple[IndexedDetection, ...],
        frame_width_px: int,
        frame_height_px: int,
    ) -> tuple[RawTrackerObservation, ...]:
        """Translate stable xyxy detections to ByteTrack and back."""

        rows = np.asarray(
            [
                [
                    item.detection.bounding_box.left_px,
                    item.detection.bounding_box.top_px,
                    item.detection.bounding_box.right_px,
                    item.detection.bounding_box.bottom_px,
                    item.detection.confidence,
                    0.0,
                ]
                for item in detections
            ],
            dtype=np.float32,
        )
        if not detections:
            rows = np.empty((0, 6), dtype=np.float32)
        try:
            boxes = self._boxes_type(rows, (frame_height_px, frame_width_px))
            tracked = np.asarray(self._tracker.update(boxes), dtype=np.float64)
        except Exception as error:
            raise PlayerTrackingInputError(
                f"ByteTrack failed at frame {frame_number}: {error}"
            ) from error
        if tracked.size == 0:
            return ()
        if tracked.ndim != 2 or tracked.shape[1] < 8:
            raise PlayerTrackingInputError(
                f"ByteTrack returned unexpected shape {tracked.shape} at frame {frame_number}"
            )

        observations: list[RawTrackerObservation] = []
        for row in tracked:
            local_detection_index = round(float(row[7]))
            if not 0 <= local_detection_index < len(detections):
                raise PlayerTrackingInputError(
                    f"ByteTrack returned invalid detection index {local_detection_index} "
                    f"at frame {frame_number}"
                )
            indexed = detections[local_detection_index]
            raw_box = indexed.detection.bounding_box
            left = min(max(0.0, float(row[0])), frame_width_px - 1.0)
            top = min(max(0.0, float(row[1])), frame_height_px - 1.0)
            right = min(max(0.0, float(row[2])), float(frame_width_px))
            bottom = min(max(0.0, float(row[3])), float(frame_height_px))
            tracker_box = (
                BoundingBox(left, top, right, bottom) if right > left and bottom > top else raw_box
            )
            tracker_id = round(float(row[4]))
            observations.append(
                RawTrackerObservation(
                    observation_id=(
                        f"tracker-observation-{frame_number:09d}-{tracker_id:06d}-"
                        f"{indexed.raw_detection_index:09d}"
                    ),
                    tracker_id=tracker_id,
                    raw_detection_index=indexed.raw_detection_index,
                    frame_number=frame_number,
                    timestamp_s=timestamp_s,
                    tracker_bounding_box=tracker_box,
                    detection_confidence=indexed.detection.confidence,
                    tracker_confidence=float(row[5]),
                )
            )
        return tuple(observations)
