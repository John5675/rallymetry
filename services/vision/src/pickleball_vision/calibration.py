"""Court-plane homography fitting, transformation, and JSON persistence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from pickleball_vision.court import (
    CourtDimensions,
    CourtPoint,
    ImagePoint,
    LandmarkName,
    court_landmarks,
)
from pickleball_vision.errors import CalibrationIoError, InvalidCalibrationError

FloatMatrix = NDArray[np.float64]
MIN_CORRESPONDENCES = 4
RANSAC_THRESHOLD_METERS = 0.10
ALL_POINT_MAX_IMAGE_ERROR_FRACTION = 0.01
ALL_POINT_MAX_COURT_ERROR_METERS = 0.25
QUALITY_WARNING_IMAGE_RMSE_FRACTION = 0.003
QUALITY_WARNING_COURT_RMSE_METERS = 0.08


class FitMethod(StrEnum):
    """Homography strategy selected from the correspondence residuals."""

    DIRECT_FOUR_POINT = "direct_four_point"
    LEAST_SQUARES_ALL_POINTS = "least_squares_all_points"
    RANSAC = "ransac"


class QualityStatus(StrEnum):
    """Human-review status derived from whole-calibration residuals."""

    PASS = "pass"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class CalibrationSource:
    """Provenance for the decoded frame used during calibration."""

    video_path: Path
    requested_timestamp_s: float
    frame_index: int
    frame_timestamp_s: float
    frame_width_px: int
    frame_height_px: int
    fps: float

    def as_dict(self) -> dict[str, object]:
        return {
            "video_path": str(self.video_path),
            "requested_timestamp_s": self.requested_timestamp_s,
            "frame_index": self.frame_index,
            "frame_timestamp_s": self.frame_timestamp_s,
            "frame_width_px": self.frame_width_px,
            "frame_height_px": self.frame_height_px,
            "fps": self.fps,
        }


@dataclass(frozen=True, slots=True)
class CalibrationCorrespondence:
    """A named image/court-plane point pair and its fitted residuals."""

    landmark: LandmarkName
    label: str
    image_point: ImagePoint
    court_point: CourtPoint
    inlier: bool = True
    court_error_m: float = 0.0
    image_error_px: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "landmark": self.landmark.value,
            "label": self.label,
            "image_point": self.image_point.as_dict(),
            "court_point": self.court_point.as_dict(),
            "inlier": self.inlier,
            "court_error_m": self.court_error_m,
            "image_error_px": self.image_error_px,
        }


@dataclass(frozen=True, slots=True)
class ReprojectionMetrics:
    """Forward and reverse residual summaries for selected correspondences."""

    all_rmse_court_m: float
    all_max_court_m: float
    all_rmse_image_px: float
    all_max_image_px: float
    inlier_rmse_court_m: float
    inlier_max_court_m: float
    inlier_rmse_image_px: float
    inlier_max_image_px: float

    def as_dict(self) -> dict[str, float]:
        return {
            "all_rmse_court_m": self.all_rmse_court_m,
            "all_max_court_m": self.all_max_court_m,
            "all_rmse_image_px": self.all_rmse_image_px,
            "all_max_image_px": self.all_max_image_px,
            "inlier_rmse_court_m": self.inlier_rmse_court_m,
            "inlier_max_court_m": self.inlier_max_court_m,
            "inlier_rmse_image_px": self.inlier_rmse_image_px,
            "inlier_max_image_px": self.inlier_max_image_px,
        }


@dataclass(frozen=True, slots=True)
class CalibrationQuality:
    """Whole-calibration quality assessment, including actionable warnings."""

    status: QualityStatus
    warnings: tuple[str, ...]
    frame_diagonal_px: float
    all_point_max_image_error_px: float
    all_point_max_court_error_m: float

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "warnings": list(self.warnings),
            "frame_diagonal_px": self.frame_diagonal_px,
            "all_point_fit_tolerances": {
                "max_image_error_px": self.all_point_max_image_error_px,
                "max_court_error_m": self.all_point_max_court_error_m,
            },
        }


@dataclass(frozen=True, slots=True)
class CourtCalibration:
    """A persisted bidirectional mapping between an image and the court plane."""

    source: CalibrationSource
    court: CourtDimensions
    correspondences: tuple[CalibrationCorrespondence, ...]
    image_to_court_homography: FloatMatrix
    court_to_image_homography: FloatMatrix
    fit_method: FitMethod
    reprojection_error: ReprojectionMetrics
    quality: CalibrationQuality
    created_at_utc: str
    schema_version: int = 2

    def image_to_court(self, point: ImagePoint) -> CourtPoint:
        """Transform a point known to lie on the court plane into meters."""

        x_m, y_m = _transform_xy(
            self.image_to_court_homography,
            point.x_px,
            point.y_px,
        )
        return CourtPoint(x_m=x_m, y_m=y_m)

    def court_to_image(self, point: CourtPoint) -> ImagePoint:
        """Project a canonical court-plane point into the calibrated image."""

        x_px, y_px = _transform_xy(
            self.court_to_image_homography,
            point.x_m,
            point.y_m,
        )
        return ImagePoint(x_px=x_px, y_px=y_px)

    @property
    def inlier_count(self) -> int:
        return sum(correspondence.inlier for correspondence in self.correspondences)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "source": self.source.as_dict(),
            "coordinate_system": {
                "image": {
                    "origin": "top_left",
                    "x_axis": "right",
                    "y_axis": "down",
                    "unit": "pixels",
                },
                "court": {
                    "origin": "near_baseline_left",
                    "x_axis": "left_to_right_facing_far_baseline",
                    "y_axis": "near_to_far_baseline",
                    "unit": "meters",
                },
            },
            "court": self.court.as_dict(),
            "fit": {
                "method": self.fit_method.value,
                "ransac_threshold_m": (
                    RANSAC_THRESHOLD_METERS if self.fit_method is FitMethod.RANSAC else None
                ),
                "correspondence_count": len(self.correspondences),
                "inlier_count": self.inlier_count,
            },
            "correspondences": [item.as_dict() for item in self.correspondences],
            "homographies": {
                "image_to_court": self.image_to_court_homography.tolist(),
                "court_to_image": self.court_to_image_homography.tolist(),
            },
            "reprojection_error": self.reprojection_error.as_dict(),
            "quality": self.quality.as_dict(),
        }


def _finite_point(point: ImagePoint | CourtPoint) -> bool:
    if isinstance(point, ImagePoint):
        return math.isfinite(point.x_px) and math.isfinite(point.y_px)
    return math.isfinite(point.x_m) and math.isfinite(point.y_m)


def _validate_correspondences(
    correspondences: tuple[CalibrationCorrespondence, ...],
    source: CalibrationSource,
    court: CourtDimensions,
) -> None:
    if (
        source.frame_width_px < 1
        or source.frame_height_px < 1
        or source.frame_index < 0
        or not math.isfinite(source.fps)
        or source.fps <= 0
        or not math.isfinite(source.requested_timestamp_s)
        or source.requested_timestamp_s < 0
        or not math.isfinite(source.frame_timestamp_s)
        or source.frame_timestamp_s < 0
    ):
        raise InvalidCalibrationError("source frame provenance is invalid")

    if len(correspondences) < MIN_CORRESPONDENCES:
        raise InvalidCalibrationError(
            f"at least {MIN_CORRESPONDENCES} correspondences are required; "
            f"received {len(correspondences)}"
        )

    landmark_names = [item.landmark for item in correspondences]
    if len(set(landmark_names)) != len(landmark_names):
        raise InvalidCalibrationError("each named landmark may be selected only once")

    image_tuples = [(item.image_point.x_px, item.image_point.y_px) for item in correspondences]
    court_tuples = [(item.court_point.x_m, item.court_point.y_m) for item in correspondences]
    if len(set(image_tuples)) != len(image_tuples):
        raise InvalidCalibrationError("image points must be unique")
    if len(set(court_tuples)) != len(court_tuples):
        raise InvalidCalibrationError("court points must be unique")
    if not all(
        _finite_point(item.image_point) and _finite_point(item.court_point)
        for item in correspondences
    ):
        raise InvalidCalibrationError("all point coordinates must be finite")

    canonical_points = {landmark.name: landmark.court_point for landmark in court_landmarks(court)}
    if any(
        item.landmark not in canonical_points
        or not math.isclose(
            item.court_point.x_m,
            canonical_points[item.landmark].x_m,
            abs_tol=1e-9,
        )
        or not math.isclose(
            item.court_point.y_m,
            canonical_points[item.landmark].y_m,
            abs_tol=1e-9,
        )
        for item in correspondences
    ):
        raise InvalidCalibrationError("each named landmark must use its canonical court coordinate")

    if any(
        item.image_point.x_px < 0
        or item.image_point.x_px >= source.frame_width_px
        or item.image_point.y_px < 0
        or item.image_point.y_px >= source.frame_height_px
        for item in correspondences
    ):
        raise InvalidCalibrationError("all image points must lie inside the decoded frame")

    image_points = np.asarray(image_tuples, dtype=np.float64)
    court_points = np.asarray(court_tuples, dtype=np.float64)
    if np.linalg.matrix_rank(image_points - image_points.mean(axis=0)) < 2:
        raise InvalidCalibrationError("image points are collinear or otherwise degenerate")
    if np.linalg.matrix_rank(court_points - court_points.mean(axis=0)) < 2:
        raise InvalidCalibrationError("court points are collinear or otherwise degenerate")


def _normalize_homography(matrix: FloatMatrix) -> FloatMatrix:
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise InvalidCalibrationError("homography contains invalid values")
    scale = float(matrix[2, 2])
    if abs(scale) < 1e-12:
        scale = float(np.linalg.norm(matrix))
    if not math.isfinite(scale) or abs(scale) < 1e-12:
        raise InvalidCalibrationError("homography has an invalid scale")
    normalized = np.asarray(matrix / scale, dtype=np.float64)
    if np.linalg.matrix_rank(normalized) < 3:
        raise InvalidCalibrationError("homography is singular")
    return normalized


def _transform_xy(matrix: FloatMatrix, x: float, y: float) -> tuple[float, float]:
    if not math.isfinite(x) or not math.isfinite(y):
        raise InvalidCalibrationError("transformation point must be finite")
    homogeneous = matrix @ np.asarray([x, y, 1.0], dtype=np.float64)
    denominator = float(homogeneous[2])
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        raise InvalidCalibrationError("point projects to infinity under this homography")
    transformed_x = float(homogeneous[0] / denominator)
    transformed_y = float(homogeneous[1] / denominator)
    if not math.isfinite(transformed_x) or not math.isfinite(transformed_y):
        raise InvalidCalibrationError("point transformation produced a non-finite result")
    return transformed_x, transformed_y


def _direct_fit(
    image_points: FloatMatrix,
    court_points: FloatMatrix,
) -> FloatMatrix:
    try:
        matrix, _ = cv2.findHomography(image_points, court_points, 0)
    except cv2.error as error:
        raise InvalidCalibrationError("OpenCV could not fit a homography") from error
    if matrix is None:
        raise InvalidCalibrationError("the selected points do not define a homography")
    return _normalize_homography(np.asarray(matrix, dtype=np.float64))


def _inverse_homography(matrix: FloatMatrix) -> FloatMatrix:
    try:
        return _normalize_homography(np.asarray(np.linalg.inv(matrix), dtype=np.float64))
    except np.linalg.LinAlgError as error:
        raise InvalidCalibrationError("homography is not invertible") from error


def _residual_arrays(
    image_to_court: FloatMatrix,
    image_points: FloatMatrix,
    court_points: FloatMatrix,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    court_to_image = _inverse_homography(image_to_court)
    court_errors: list[float] = []
    image_errors: list[float] = []
    for image_point, court_point in zip(image_points, court_points, strict=True):
        projected_court = _transform_xy(
            image_to_court,
            float(image_point[0]),
            float(image_point[1]),
        )
        projected_image = _transform_xy(
            court_to_image,
            float(court_point[0]),
            float(court_point[1]),
        )
        court_errors.append(
            math.hypot(
                projected_court[0] - float(court_point[0]),
                projected_court[1] - float(court_point[1]),
            )
        )
        image_errors.append(
            math.hypot(
                projected_image[0] - float(image_point[0]),
                projected_image[1] - float(image_point[1]),
            )
        )
    return (
        np.asarray(court_errors, dtype=np.float64),
        np.asarray(image_errors, dtype=np.float64),
    )


def _robust_fit(
    image_points: FloatMatrix,
    court_points: FloatMatrix,
) -> tuple[FloatMatrix, NDArray[np.bool_]]:
    try:
        matrix, raw_mask = cv2.findHomography(
            image_points,
            court_points,
            cv2.RANSAC,
            RANSAC_THRESHOLD_METERS,
        )
    except cv2.error as error:
        raise InvalidCalibrationError("OpenCV could not robustly fit a homography") from error
    if matrix is None:
        raise InvalidCalibrationError("the selected points do not define a homography")

    inliers = np.asarray(raw_mask, dtype=np.uint8).reshape(-1).astype(np.bool_)
    if int(inliers.sum()) < MIN_CORRESPONDENCES:
        raise InvalidCalibrationError("robust fitting found fewer than four inlier points")

    return _direct_fit(image_points[inliers], court_points[inliers]), inliers


def _fit_matrix(
    image_points: FloatMatrix,
    court_points: FloatMatrix,
    source: CalibrationSource,
) -> tuple[FloatMatrix, NDArray[np.bool_], FitMethod]:
    direct = _direct_fit(image_points, court_points)
    all_inliers = np.ones(len(image_points), dtype=np.bool_)
    if len(image_points) == MIN_CORRESPONDENCES:
        return direct, all_inliers, FitMethod.DIRECT_FOUR_POINT

    direct_court_errors, direct_image_errors = _residual_arrays(
        direct,
        image_points,
        court_points,
    )
    frame_diagonal = math.hypot(source.frame_width_px, source.frame_height_px)
    image_tolerance = frame_diagonal * ALL_POINT_MAX_IMAGE_ERROR_FRACTION
    if (
        float(direct_image_errors.max()) <= image_tolerance
        and float(direct_court_errors.max()) <= ALL_POINT_MAX_COURT_ERROR_METERS
    ):
        return direct, all_inliers, FitMethod.LEAST_SQUARES_ALL_POINTS

    robust, inliers = _robust_fit(image_points, court_points)
    return robust, inliers, FitMethod.RANSAC


def _rmse(values: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _assess_quality(
    *,
    source: CalibrationSource,
    fit_method: FitMethod,
    correspondences: tuple[CalibrationCorrespondence, ...],
    metrics: ReprojectionMetrics,
) -> CalibrationQuality:
    frame_diagonal = math.hypot(source.frame_width_px, source.frame_height_px)
    all_point_image_tolerance = frame_diagonal * ALL_POINT_MAX_IMAGE_ERROR_FRACTION
    warnings: list[str] = []
    outlier_count = sum(not item.inlier for item in correspondences)
    if len(correspondences) == MIN_CORRESPONDENCES:
        warnings.append(
            "Exactly four points make reprojection error non-diagnostic; inspect both debug images"
        )
    if outlier_count:
        exclusion_summary = (
            f"Robust fitting excluded {outlier_count} of {len(correspondences)} selected landmarks"
        )
        warnings.append(f"{exclusion_summary}; review orange points and whole-court error")
    if metrics.all_rmse_image_px > frame_diagonal * QUALITY_WARNING_IMAGE_RMSE_FRACTION:
        warnings.append(
            f"Whole-court image RMSE is {metrics.all_rmse_image_px:.2f}px; "
            "wide-angle lens distortion or imprecise clicks may limit one-homography alignment"
        )
    if metrics.all_rmse_court_m > QUALITY_WARNING_COURT_RMSE_METERS:
        warnings.append(
            f"Whole-court coordinate RMSE is {metrics.all_rmse_court_m:.3f}m; "
            "inspect projected lines before using court coordinates"
        )
    if fit_method is FitMethod.RANSAC and metrics.all_max_image_px > all_point_image_tolerance:
        warnings.append(
            f"At least one selected landmark is displaced by {metrics.all_max_image_px:.2f}px"
        )
    return CalibrationQuality(
        status=QualityStatus.WARNING if warnings else QualityStatus.PASS,
        warnings=tuple(warnings),
        frame_diagonal_px=frame_diagonal,
        all_point_max_image_error_px=all_point_image_tolerance,
        all_point_max_court_error_m=ALL_POINT_MAX_COURT_ERROR_METERS,
    )


def fit_calibration(
    *,
    source: CalibrationSource,
    court: CourtDimensions,
    correspondences: tuple[CalibrationCorrespondence, ...],
) -> CourtCalibration:
    """Fit and validate a bidirectional court-plane homography."""

    _validate_correspondences(correspondences, source, court)
    image_points = np.asarray(
        [(item.image_point.x_px, item.image_point.y_px) for item in correspondences],
        dtype=np.float64,
    )
    court_points = np.asarray(
        [(item.court_point.x_m, item.court_point.y_m) for item in correspondences],
        dtype=np.float64,
    )
    image_to_court, inliers, fit_method = _fit_matrix(image_points, court_points, source)
    court_to_image = _inverse_homography(image_to_court)
    court_error_array, image_error_array = _residual_arrays(
        image_to_court,
        image_points,
        court_points,
    )
    fitted_correspondences: list[CalibrationCorrespondence] = []
    for index, item in enumerate(correspondences):
        fitted_correspondences.append(
            replace(
                item,
                inlier=bool(inliers[index]),
                court_error_m=float(court_error_array[index]),
                image_error_px=float(image_error_array[index]),
            )
        )

    inlier_court_errors = court_error_array[inliers]
    inlier_image_errors = image_error_array[inliers]
    metrics = ReprojectionMetrics(
        all_rmse_court_m=_rmse(court_error_array),
        all_max_court_m=float(court_error_array.max()),
        all_rmse_image_px=_rmse(image_error_array),
        all_max_image_px=float(image_error_array.max()),
        inlier_rmse_court_m=_rmse(inlier_court_errors),
        inlier_max_court_m=float(inlier_court_errors.max()),
        inlier_rmse_image_px=_rmse(inlier_image_errors),
        inlier_max_image_px=float(inlier_image_errors.max()),
    )
    fitted_items = tuple(fitted_correspondences)
    quality = _assess_quality(
        source=source,
        fit_method=fit_method,
        correspondences=fitted_items,
        metrics=metrics,
    )
    return CourtCalibration(
        source=source,
        court=court,
        correspondences=fitted_items,
        image_to_court_homography=image_to_court,
        court_to_image_homography=court_to_image,
        fit_method=fit_method,
        reprojection_error=metrics,
        quality=quality,
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def save_calibration(calibration: CourtCalibration, path: Path) -> Path:
    """Persist a calibration as human-inspectable JSON."""

    output_path = path.expanduser().resolve()
    if output_path.suffix.lower() != ".json":
        raise CalibrationIoError(str(output_path), reason="output must use a .json extension")
    if output_path.exists() and not output_path.is_file():
        raise CalibrationIoError(str(output_path), reason="path is not a regular file")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(calibration.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        raise CalibrationIoError(str(output_path), reason=str(error)) from error
    return output_path


def _matrix_from_json(value: object, *, field: str) -> FloatMatrix:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidCalibrationError(f"{field} is not a numeric matrix") from error
    return _normalize_homography(matrix)


def _json_dict(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InvalidCalibrationError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _json_list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise InvalidCalibrationError(f"{field} must be an array")
    return cast(list[object], value)


def _json_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidCalibrationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InvalidCalibrationError(f"{field} must be finite")
    return result


def _json_int(value: object, *, field: str) -> int:
    number = _json_float(value, field=field)
    if not number.is_integer():
        raise InvalidCalibrationError(f"{field} must be an integer")
    return int(number)


def _json_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise InvalidCalibrationError(f"{field} must be a string")
    return value


def _json_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise InvalidCalibrationError(f"{field} must be a boolean")
    return value


def _parse_correspondence(value: object, *, index: int) -> CalibrationCorrespondence:
    field = f"correspondences[{index}]"
    item = _json_dict(value, field=field)
    image_raw = _json_dict(item["image_point"], field=f"{field}.image_point")
    court_raw = _json_dict(item["court_point"], field=f"{field}.court_point")
    return CalibrationCorrespondence(
        landmark=LandmarkName(_json_string(item["landmark"], field=f"{field}.landmark")),
        label=_json_string(item["label"], field=f"{field}.label"),
        image_point=ImagePoint(
            x_px=_json_float(image_raw["x_px"], field=f"{field}.image_point.x_px"),
            y_px=_json_float(image_raw["y_px"], field=f"{field}.image_point.y_px"),
        ),
        court_point=CourtPoint(
            x_m=_json_float(court_raw["x_m"], field=f"{field}.court_point.x_m"),
            y_m=_json_float(court_raw["y_m"], field=f"{field}.court_point.y_m"),
        ),
        inlier=_json_bool(item["inlier"], field=f"{field}.inlier"),
        court_error_m=_json_float(item["court_error_m"], field=f"{field}.court_error_m"),
        image_error_px=_json_float(item["image_error_px"], field=f"{field}.image_error_px"),
    )


def _parse_fit_method(raw: dict[str, object], *, schema_version: int) -> FitMethod:
    fit_raw = _json_dict(raw["fit"], field="fit")
    method = _json_string(fit_raw["method"], field="fit.method")
    if schema_version == 1 and method == "direct":
        return FitMethod.DIRECT_FOUR_POINT
    try:
        return FitMethod(method)
    except ValueError as error:
        raise InvalidCalibrationError(f"unsupported fit method: {method}") from error


def load_calibration(path: Path) -> CourtCalibration:
    """Load and validate calibration JSON produced by this package."""

    input_path = path.expanduser().resolve()
    try:
        decoded_json: object = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationIoError(str(input_path), reason=str(error)) from error

    try:
        raw = _json_dict(decoded_json, field="root")
        schema_version = _json_int(raw["schema_version"], field="schema_version")
        if schema_version not in {1, 2}:
            raise InvalidCalibrationError("unsupported calibration schema version")
        source_raw = _json_dict(raw["source"], field="source")
        court_raw = _json_dict(raw["court"], field="court")
        homographies = _json_dict(raw["homographies"], field="homographies")
        correspondence_items = _json_list(raw["correspondences"], field="correspondences")
        error_raw = _json_dict(raw["reprojection_error"], field="reprojection_error")
        source = CalibrationSource(
            video_path=Path(_json_string(source_raw["video_path"], field="source.video_path")),
            requested_timestamp_s=_json_float(
                source_raw["requested_timestamp_s"], field="source.requested_timestamp_s"
            ),
            frame_index=_json_int(source_raw["frame_index"], field="source.frame_index"),
            frame_timestamp_s=_json_float(
                source_raw["frame_timestamp_s"], field="source.frame_timestamp_s"
            ),
            frame_width_px=_json_int(source_raw["frame_width_px"], field="source.frame_width_px"),
            frame_height_px=_json_int(
                source_raw["frame_height_px"], field="source.frame_height_px"
            ),
            fps=_json_float(source_raw["fps"], field="source.fps"),
        )
        court = CourtDimensions(
            width_m=_json_float(court_raw["width_m"], field="court.width_m"),
            length_m=_json_float(court_raw["length_m"], field="court.length_m"),
            non_volley_zone_depth_m=_json_float(
                court_raw["non_volley_zone_depth_m"],
                field="court.non_volley_zone_depth_m",
            ),
        )
        correspondences = tuple(
            _parse_correspondence(item, index=index)
            for index, item in enumerate(correspondence_items)
        )
        metrics = ReprojectionMetrics(
            all_rmse_court_m=_json_float(
                error_raw["all_rmse_court_m"], field="reprojection_error.all_rmse_court_m"
            ),
            all_max_court_m=_json_float(
                error_raw["all_max_court_m"], field="reprojection_error.all_max_court_m"
            ),
            all_rmse_image_px=_json_float(
                error_raw["all_rmse_image_px"], field="reprojection_error.all_rmse_image_px"
            ),
            all_max_image_px=_json_float(
                error_raw["all_max_image_px"], field="reprojection_error.all_max_image_px"
            ),
            inlier_rmse_court_m=_json_float(
                error_raw["inlier_rmse_court_m"],
                field="reprojection_error.inlier_rmse_court_m",
            ),
            inlier_max_court_m=_json_float(
                error_raw["inlier_max_court_m"],
                field="reprojection_error.inlier_max_court_m",
            ),
            inlier_rmse_image_px=_json_float(
                error_raw["inlier_rmse_image_px"],
                field="reprojection_error.inlier_rmse_image_px",
            ),
            inlier_max_image_px=_json_float(
                error_raw["inlier_max_image_px"],
                field="reprojection_error.inlier_max_image_px",
            ),
        )
        fit_method = _parse_fit_method(raw, schema_version=schema_version)
        quality = _assess_quality(
            source=source,
            fit_method=fit_method,
            correspondences=correspondences,
            metrics=metrics,
        )
        calibration = CourtCalibration(
            source=source,
            court=court,
            correspondences=correspondences,
            image_to_court_homography=_matrix_from_json(
                homographies["image_to_court"], field="image_to_court"
            ),
            court_to_image_homography=_matrix_from_json(
                homographies["court_to_image"], field="court_to_image"
            ),
            fit_method=fit_method,
            reprojection_error=metrics,
            quality=quality,
            created_at_utc=_json_string(raw["created_at_utc"], field="created_at_utc"),
            schema_version=schema_version,
        )
        _validate_correspondences(
            calibration.correspondences,
            calibration.source,
            calibration.court,
        )
        return calibration
    except InvalidCalibrationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidCalibrationError(f"calibration JSON has an invalid schema: {error}") from error
