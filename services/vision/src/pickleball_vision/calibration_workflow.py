"""End-to-end manual calibration orchestration for a local video frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pickleball_vision.calibration import (
    CalibrationSource,
    CourtCalibration,
    fit_calibration,
    save_calibration,
)
from pickleball_vision.calibration_render import (
    render_calibration_overlay,
    render_court_topdown,
    write_debug_image,
)
from pickleball_vision.calibration_ui import select_court_landmarks
from pickleball_vision.court import CourtDimensions
from pickleball_vision.errors import CalibrationIoError
from pickleball_vision.video import decode_video_frame


@dataclass(frozen=True, slots=True)
class CalibrationArtifacts:
    """Persisted calibration and visual-debug artifact paths."""

    calibration: CourtCalibration
    calibration_path: Path
    overlay_path: Path
    topdown_path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "calibration_path": str(self.calibration_path),
            "overlay_path": str(self.overlay_path),
            "topdown_path": str(self.topdown_path),
            "correspondence_count": len(self.calibration.correspondences),
            "inlier_count": self.calibration.inlier_count,
            "fit_method": self.calibration.fit_method.value,
            "reprojection_error": self.calibration.reprojection_error.as_dict(),
            "quality": self.calibration.quality.as_dict(),
        }


def calibrate_video(
    video_path: Path,
    *,
    timestamp_seconds: float,
    output_path: Path,
) -> CalibrationArtifacts:
    """Select landmarks, fit a homography, and persist inspectable artifacts."""

    resolved_output = output_path.expanduser().resolve()
    if resolved_output.suffix.lower() != ".json":
        raise CalibrationIoError(
            str(resolved_output), reason="calibration output must use a .json extension"
        )

    decoded = decode_video_frame(video_path, timestamp_seconds=timestamp_seconds)
    dimensions = CourtDimensions()
    correspondences = select_court_landmarks(decoded.image, dimensions)
    source = CalibrationSource(
        video_path=decoded.metadata.path,
        requested_timestamp_s=timestamp_seconds,
        frame_index=decoded.frame_index,
        frame_timestamp_s=decoded.timestamp,
        frame_width_px=decoded.metadata.width,
        frame_height_px=decoded.metadata.height,
        fps=decoded.metadata.fps,
    )
    calibration = fit_calibration(
        source=source,
        court=dimensions,
        correspondences=correspondences,
    )

    overlay_path = resolved_output.parent / "calibration-overlay.jpg"
    topdown_path = resolved_output.parent / "court-topdown.jpg"
    write_debug_image(
        render_calibration_overlay(decoded.image, calibration),
        overlay_path,
    )
    write_debug_image(
        render_court_topdown(decoded.image, calibration),
        topdown_path,
    )
    save_calibration(calibration, resolved_output)
    return CalibrationArtifacts(
        calibration=calibration,
        calibration_path=resolved_output,
        overlay_path=overlay_path,
        topdown_path=topdown_path,
    )
