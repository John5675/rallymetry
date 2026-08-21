"""Video orchestration and artifacts for broad person detection."""

from __future__ import annotations

import json
import logging
import math
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from pickleball_vision.calibration import load_calibration
from pickleball_vision.config import PersonDetectionSettings
from pickleball_vision.detectors import UltralyticsPersonDetector
from pickleball_vision.errors import DetectionInputError, OutputWriteError
from pickleball_vision.person_detection import (
    PersonDetection,
    PersonDetectionRun,
    PersonDetector,
)
from pickleball_vision.video import Image, VideoMetadata, inspect_video, iter_video_frames
from pickleball_vision.video_output import CompressedVideoWriter

ANNOTATED_VIDEO_NAME = "annotated.mp4"
DETECTIONS_NAME = "detections.json"
SUMMARY_NAME = "summary.json"


@dataclass(frozen=True, slots=True)
class PersonDetectionArtifacts:
    """Paths and counts returned by a completed detection run."""

    detections_path: Path
    annotated_video_path: Path
    summary_path: Path
    processed_frame_count: int
    detection_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "detections_path": str(self.detections_path),
            "annotated_video_path": str(self.annotated_video_path),
            "summary_path": str(self.summary_path),
            "processed_frame_count": self.processed_frame_count,
            "detection_count": self.detection_count,
        }


def _prepare_output_dir(path: Path) -> Path:
    output_dir = path.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise OutputWriteError(str(output_dir), reason="path is not a directory")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputWriteError(str(output_dir), reason=str(error)) from error
    return output_dir


def _open_annotated_writer(path: Path, metadata: VideoMetadata) -> CompressedVideoWriter:
    return CompressedVideoWriter(
        path,
        fps=metadata.fps,
        dimensions=(metadata.width, metadata.height),
    )


def render_person_detections(
    frame: Image,
    detections: tuple[PersonDetection, ...],
) -> Image:
    """Draw debug-only person boxes without changing observation coordinates."""

    annotated = frame.copy()
    line_width = max(1, round(min(frame.shape[:2]) / 360))
    font_scale = max(0.4, min(frame.shape[:2]) / 1200)
    for detection in detections:
        box = detection.bounding_box
        left_top = (round(box.left_px), round(box.top_px))
        right_bottom = (round(box.right_px), round(box.bottom_px))
        cv2.rectangle(annotated, left_top, right_bottom, (0, 210, 255), line_width)
        label = f"person {detection.confidence:.2f}"
        label_origin = (left_top[0], max(14, left_top[1] - 6))
        cv2.putText(
            annotated,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 210, 255),
            line_width,
            cv2.LINE_AA,
        )
    return np.asarray(annotated, dtype=np.uint8)


def _write_json(path: Path, value: object) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except (OSError, TypeError, ValueError) as error:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise OutputWriteError(str(path), reason=str(error)) from error


def _confidence_statistics(detections: tuple[PersonDetection, ...]) -> dict[str, float | None]:
    if not detections:
        return {"minimum": None, "mean": None, "maximum": None}
    values = tuple(detection.confidence for detection in detections)
    return {
        "minimum": min(values),
        "mean": math.fsum(values) / len(values),
        "maximum": max(values),
    }


def _summary(
    *,
    metadata: VideoMetadata,
    calibration_path: Path,
    calibration_video_matches: bool,
    detector: PersonDetector,
    settings: PersonDetectionSettings,
    per_frame_counts: tuple[int, ...],
    detections: tuple[PersonDetection, ...],
    elapsed_seconds: float,
    output_dir: Path,
) -> dict[str, object]:
    processed_frames = len(per_frame_counts)
    frames_with_detections = sum(count > 0 for count in per_frame_counts)
    total_detections = len(detections)
    return {
        "schema_version": 1,
        "record_type": "person_detection_summary",
        "source": metadata.as_dict(),
        "calibration": {
            "path": str(calibration_path),
            "source_video_path_matches": calibration_video_matches,
            "usage": "validated_provenance_only_no_person_projection",
        },
        "detector": detector.metadata.as_dict(),
        "configuration": settings.as_dict(),
        "statistics": {
            "processed_frames": processed_frames,
            "frames_with_detections": frames_with_detections,
            "frames_without_detections": processed_frames - frames_with_detections,
            "total_detections": total_detections,
            "detections_per_frame": {
                "minimum": min(per_frame_counts, default=0),
                "mean": total_detections / processed_frames if processed_frames else 0.0,
                "maximum": max(per_frame_counts, default=0),
            },
            "confidence": _confidence_statistics(detections),
        },
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "processed_frames_per_second": (
                processed_frames / elapsed_seconds if elapsed_seconds > 0 else None
            ),
        },
        "artifacts": {
            "detections": str(output_dir / DETECTIONS_NAME),
            "annotated_video": str(output_dir / ANNOTATED_VIDEO_NAME),
            "summary": str(output_dir / SUMMARY_NAME),
        },
    }


def detect_people_in_video(
    video_path: Path,
    *,
    calibration_path: Path,
    output_dir: Path,
    settings: PersonDetectionSettings,
    detector: PersonDetector | None = None,
) -> PersonDetectionArtifacts:
    """Detect all visible people and write raw JSON, debug video, and statistics."""

    metadata = inspect_video(video_path)
    resolved_calibration_path = calibration_path.expanduser().resolve()
    calibration = load_calibration(resolved_calibration_path)
    calibration_dimensions = (
        calibration.source.frame_width_px,
        calibration.source.frame_height_px,
    )
    video_dimensions = (metadata.width, metadata.height)
    if calibration_dimensions != video_dimensions:
        raise DetectionInputError(
            "calibration frame dimensions "
            f"{calibration_dimensions} do not match video dimensions {video_dimensions}"
        )

    resolved_output_dir = _prepare_output_dir(output_dir)
    annotated_path = resolved_output_dir / ANNOTATED_VIDEO_NAME
    if annotated_path == metadata.path:
        raise OutputWriteError(
            str(annotated_path),
            reason="annotated output would overwrite the source video",
        )
    active_detector = detector or UltralyticsPersonDetector(settings)
    temporary_annotated_path = resolved_output_dir / ".annotated.tmp.mp4"
    writer = _open_annotated_writer(temporary_annotated_path, metadata)
    logger = logging.getLogger("pickleball_vision.person_detection")
    detections: list[PersonDetection] = []
    per_frame_counts: list[int] = []
    started = time.perf_counter()
    try:
        for decoded in iter_video_frames(metadata.path):
            frame_detections = active_detector.detect(
                decoded.image,
                frame_number=decoded.frame_index,
                timestamp_s=decoded.timestamp,
            )
            detections.extend(frame_detections)
            per_frame_counts.append(len(frame_detections))
            annotated = render_person_detections(decoded.image, frame_detections)
            writer.write(annotated)
            if (decoded.frame_index + 1) % 300 == 0:
                logger.info(
                    "person_detection_progress",
                    extra={
                        "context": {
                            "processed_frames": decoded.frame_index + 1,
                            "source_frames": metadata.frame_count,
                            "detections": len(detections),
                        }
                    },
                )
    except Exception:
        writer.abort()
        with suppress(OSError):
            temporary_annotated_path.unlink(missing_ok=True)
        raise
    else:
        writer.release()
    elapsed_seconds = time.perf_counter() - started

    if len(per_frame_counts) != metadata.frame_count:
        raise DetectionInputError(
            f"processed {len(per_frame_counts)} frames but source reports {metadata.frame_count}"
        )
    if not temporary_annotated_path.is_file() or temporary_annotated_path.stat().st_size == 0:
        raise OutputWriteError(str(annotated_path), reason="OpenCV wrote an empty video")
    try:
        temporary_annotated_path.replace(annotated_path)
    except OSError as error:
        raise OutputWriteError(str(annotated_path), reason=str(error)) from error

    completed_at = datetime.now(UTC).isoformat()
    detection_tuple = tuple(detections)
    calibration_video_matches = (
        calibration.source.video_path.expanduser().resolve() == metadata.path
    )
    run = PersonDetectionRun(
        created_at_utc=completed_at,
        source=metadata,
        calibration_path=str(resolved_calibration_path),
        calibration_schema_version=calibration.schema_version,
        detector=active_detector.metadata,
        configuration=settings.as_dict(),
        detections=detection_tuple,
    )
    detections_path = resolved_output_dir / DETECTIONS_NAME
    summary_path = resolved_output_dir / SUMMARY_NAME
    _write_json(detections_path, run.as_dict())
    _write_json(
        summary_path,
        _summary(
            metadata=metadata,
            calibration_path=resolved_calibration_path,
            calibration_video_matches=calibration_video_matches,
            detector=active_detector,
            settings=settings,
            per_frame_counts=tuple(per_frame_counts),
            detections=detection_tuple,
            elapsed_seconds=elapsed_seconds,
            output_dir=resolved_output_dir,
        ),
    )
    return PersonDetectionArtifacts(
        detections_path=detections_path,
        annotated_video_path=annotated_path,
        summary_path=summary_path,
        processed_frame_count=len(per_frame_counts),
        detection_count=len(detections),
    )
