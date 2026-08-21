"""Video orchestration for raw custom-model pickleball detections."""

from __future__ import annotations

import json
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from pickleball_vision.ball_config import BallInferenceStrategy
from pickleball_vision.ball_detection import (
    BALL_DETECTION_SCHEMA_VERSION,
    BallDetection,
    BallDetector,
    BallFrameInference,
    infer_ball_frame,
)
from pickleball_vision.calibration import CourtCalibration, load_calibration
from pickleball_vision.detectors import UltralyticsBallDetector
from pickleball_vision.errors import DetectionInputError, OutputWriteError
from pickleball_vision.video import Image, VideoMetadata, inspect_video, iter_video_frames
from pickleball_vision.video_output import CompressedVideoWriter

DETECTIONS_NAME = "detections.json"
ANNOTATED_VIDEO_NAME = "annotated.mp4"
SUMMARY_NAME = "summary.json"


@dataclass(frozen=True, slots=True)
class BallDetectionArtifacts:
    """Paths and counts from one raw, non-temporal ball inference run."""

    detections_path: Path
    annotated_video_path: Path
    summary_path: Path
    processed_frame_count: int
    detection_count: int
    region_prediction_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "detections_path": str(self.detections_path),
            "annotated_video_path": str(self.annotated_video_path),
            "summary_path": str(self.summary_path),
            "processed_frame_count": self.processed_frame_count,
            "detection_count": self.detection_count,
            "region_prediction_count": self.region_prediction_count,
        }


def _prepare_output_dir(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise OutputWriteError(str(output), reason="path is not a directory")
    detections_path = output / DETECTIONS_NAME
    if detections_path.exists():
        raise OutputWriteError(str(detections_path), reason="ball-detection output already exists")
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputWriteError(str(output), reason=str(error)) from error
    return output


def _open_writer(path: Path, metadata: VideoMetadata) -> CompressedVideoWriter:
    return CompressedVideoWriter(
        path,
        fps=metadata.fps,
        dimensions=(metadata.width, metadata.height),
    )


def render_ball_detections(image: Image, detections: tuple[BallDetection, ...]) -> Image:
    """Draw review-only source-coordinate ball boxes."""

    rendered = image.copy()
    line_width = max(1, round(min(image.shape[:2]) / 540))
    font_scale = max(0.35, min(image.shape[:2]) / 1600)
    for detection in detections:
        box = detection.bounding_box
        top_left = (round(box.left_px), round(box.top_px))
        bottom_right = (round(box.right_px), round(box.bottom_px))
        cv2.rectangle(rendered, top_left, bottom_right, (0, 255, 120), line_width)
        cv2.putText(
            rendered,
            f"pickleball {detection.confidence:.2f}",
            (top_left[0], max(12, top_left[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 120),
            line_width,
            cv2.LINE_AA,
        )
    return np.asarray(rendered, dtype=np.uint8)


def _frame_payload(frame: BallFrameInference) -> dict[str, object]:
    return {
        "frame_number": frame.frame_number,
        "timestamp_s": frame.timestamp_s,
        "regions": [region.as_dict() for region in frame.regions],
        "region_predictions": [item.as_dict() for item in frame.region_predictions],
        "detections": [item.as_dict() for item in frame.detections],
    }


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise OutputWriteError(str(path), reason=str(error)) from error


def detect_balls_in_video(
    video_path: Path,
    *,
    weights_path: Path,
    model_version: str,
    strategy: BallInferenceStrategy,
    output_dir: Path,
    calibration_path: Path | None = None,
    device: str = "auto",
    detector: BallDetector | None = None,
) -> BallDetectionArtifacts:
    """Run spatial-only custom ball inference over every decoded video frame."""

    metadata = inspect_video(video_path)
    calibration: CourtCalibration | None = None
    resolved_calibration_path: Path | None = None
    if calibration_path is not None:
        resolved_calibration_path = calibration_path.expanduser().resolve()
        calibration = load_calibration(resolved_calibration_path)
    if strategy.mode.uses_court_roi and calibration is None:
        raise DetectionInputError(
            f"strategy {strategy.name!r} requires --calibration for the primary-court ROI"
        )
    output = _prepare_output_dir(output_dir)
    annotated_path = output / ANNOTATED_VIDEO_NAME
    if annotated_path == metadata.path:
        raise OutputWriteError(str(annotated_path), reason="output would overwrite source video")
    active_detector = detector or UltralyticsBallDetector(
        weights_path,
        model_version=model_version,
        device=device,
    )
    temporary_video = output / ".annotated.tmp.mp4"
    writer = _open_writer(temporary_video, metadata)
    frames: list[BallFrameInference] = []
    logger = logging.getLogger("pickleball_vision.ball_detection")
    started = time.perf_counter()
    try:
        for decoded in iter_video_frames(metadata.path):
            result = infer_ball_frame(
                decoded.image,
                frame_number=decoded.frame_index,
                timestamp_s=decoded.timestamp,
                strategy=strategy,
                detector=active_detector,
                calibration=calibration,
            )
            frames.append(result)
            writer.write(render_ball_detections(decoded.image, result.detections))
            if (decoded.frame_index + 1) % 300 == 0:
                logger.info(
                    "ball_detection_progress",
                    extra={
                        "context": {
                            "processed_frames": decoded.frame_index + 1,
                            "source_frames": metadata.frame_count,
                            "detections": sum(len(frame.detections) for frame in frames),
                            "strategy": strategy.name,
                        }
                    },
                )
    except Exception:
        writer.abort()
        temporary_video.unlink(missing_ok=True)
        raise
    else:
        writer.release()
    if len(frames) != metadata.frame_count:
        raise DetectionInputError(
            f"processed {len(frames)} frames but source reports {metadata.frame_count}"
        )
    if not temporary_video.is_file() or temporary_video.stat().st_size == 0:
        raise OutputWriteError(str(annotated_path), reason="OpenCV wrote an empty video")
    temporary_video.replace(annotated_path)
    elapsed = time.perf_counter() - started
    detection_count = sum(len(frame.detections) for frame in frames)
    proposal_count = sum(len(frame.region_predictions) for frame in frames)
    detections_path = output / DETECTIONS_NAME
    summary_path = output / SUMMARY_NAME
    calibration_payload = {
        "path": str(resolved_calibration_path) if resolved_calibration_path else None,
        "usage": (
            "primary_court_image_roi_only_no_ball_homography_projection"
            if strategy.mode.uses_court_roi
            else "not_used"
        ),
    }
    _write_json(
        detections_path,
        {
            "schema_version": BALL_DETECTION_SCHEMA_VERSION,
            "record_type": "raw_pickleball_detections",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": metadata.as_dict(),
            "detector": active_detector.metadata.as_dict(),
            "strategy": strategy.as_dict(),
            "calibration": calibration_payload,
            "coordinate_system": {
                "origin": "top_left",
                "x_axis": "right",
                "y_axis": "down",
                "unit": "source_frame_pixels",
            },
            "temporal_processing": {
                "tracking": False,
                "interpolation": False,
                "events": False,
            },
            "frames": [_frame_payload(frame) for frame in frames],
        },
    )
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "record_type": "pickleball_detection_summary",
            "source": metadata.as_dict(),
            "detector": active_detector.metadata.as_dict(),
            "strategy": strategy.as_dict(),
            "calibration": calibration_payload,
            "statistics": {
                "frames_processed": len(frames),
                "frames_with_detections": sum(bool(frame.detections) for frame in frames),
                "region_predictions": proposal_count,
                "detections": detection_count,
            },
            "runtime": {
                "elapsed_seconds": elapsed,
                "frames_per_second": len(frames) / elapsed if elapsed > 0 else None,
            },
            "artifacts": {
                "detections": str(detections_path),
                "annotated_video": str(annotated_path),
                "summary": str(summary_path),
            },
        },
    )
    return BallDetectionArtifacts(
        detections_path=detections_path,
        annotated_video_path=annotated_path,
        summary_path=summary_path,
        processed_frame_count=len(frames),
        detection_count=detection_count,
        region_prediction_count=proposal_count,
    )
