import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from pickleball_vision.calibration import (
    CalibrationCorrespondence,
    CalibrationSource,
    CourtCalibration,
    FitMethod,
    QualityStatus,
    fit_calibration,
    load_calibration,
    save_calibration,
)
from pickleball_vision.calibration_render import (
    render_calibration_overlay,
    render_court_topdown,
    write_debug_image,
)
from pickleball_vision.court import CourtDimensions, CourtPoint, ImagePoint, court_landmarks
from pickleball_vision.errors import CalibrationIoError, InvalidCalibrationError


@pytest.fixture
def calibration_source(tmp_path: Path) -> CalibrationSource:
    return CalibrationSource(
        video_path=(tmp_path / "synthetic.mp4").resolve(),
        requested_timestamp_s=10.25,
        frame_index=307,
        frame_timestamp_s=307 / 30,
        frame_width_px=1920,
        frame_height_px=1080,
        fps=30.0,
    )


@pytest.fixture
def known_court_to_image() -> np.ndarray:
    court = CourtDimensions()
    canonical_corners = np.asarray(
        [[0, 0], [court.width_m, 0], [court.width_m, court.length_m], [0, court.length_m]],
        dtype=np.float32,
    )
    low_angle_image_corners = np.asarray(
        [[100, 1000], [1820, 1000], [1270, 260], [650, 260]],
        dtype=np.float32,
    )
    return np.asarray(
        cv2.getPerspectiveTransform(canonical_corners, low_angle_image_corners),
        dtype=np.float64,
    )


def _project(matrix: np.ndarray, point: CourtPoint) -> ImagePoint:
    value = matrix @ np.asarray([point.x_m, point.y_m, 1.0], dtype=np.float64)
    return ImagePoint(x_px=float(value[0] / value[2]), y_px=float(value[1] / value[2]))


def _correspondences(
    matrix: np.ndarray,
    mutate: Callable[[int, ImagePoint], ImagePoint] | None = None,
) -> tuple[CalibrationCorrespondence, ...]:
    result: list[CalibrationCorrespondence] = []
    for index, landmark in enumerate(court_landmarks(CourtDimensions())):
        image_point = _project(matrix, landmark.court_point)
        if mutate is not None:
            image_point = mutate(index, image_point)
        result.append(
            CalibrationCorrespondence(
                landmark=landmark.name,
                label=landmark.label,
                image_point=image_point,
                court_point=landmark.court_point,
            )
        )
    return tuple(result)


def _fit(
    source: CalibrationSource,
    matrix: np.ndarray,
    mutate: Callable[[int, ImagePoint], ImagePoint] | None = None,
) -> CourtCalibration:
    return fit_calibration(
        source=source,
        court=CourtDimensions(),
        correspondences=_correspondences(matrix, mutate),
    )


def test_multi_point_calibration_transforms_in_both_directions(
    calibration_source: CalibrationSource,
    known_court_to_image: np.ndarray,
) -> None:
    calibration = _fit(calibration_source, known_court_to_image)
    expected_court = CourtPoint(2.25, 7.4)
    expected_image = _project(known_court_to_image, expected_court)

    transformed_court = calibration.image_to_court(expected_image)
    transformed_image = calibration.court_to_image(expected_court)

    assert transformed_court.x_m == pytest.approx(expected_court.x_m, abs=1e-5)
    assert transformed_court.y_m == pytest.approx(expected_court.y_m, abs=1e-5)
    assert transformed_image.x_px == pytest.approx(expected_image.x_px, abs=1e-3)
    assert transformed_image.y_px == pytest.approx(expected_image.y_px, abs=1e-3)
    assert calibration.inlier_count == 10
    assert calibration.fit_method is FitMethod.LEAST_SQUARES_ALL_POINTS
    assert calibration.quality.status is QualityStatus.PASS
    assert calibration.reprojection_error.inlier_rmse_image_px < 1e-3


def test_additional_points_allow_robust_outlier_rejection(
    calibration_source: CalibrationSource,
    known_court_to_image: np.ndarray,
) -> None:
    def introduce_outlier(index: int, point: ImagePoint) -> ImagePoint:
        if index == 4:
            return ImagePoint(point.x_px + 350, point.y_px - 180)
        return point

    calibration = _fit(calibration_source, known_court_to_image, introduce_outlier)

    assert calibration.inlier_count == 9
    assert calibration.fit_method is FitMethod.RANSAC
    assert calibration.quality.status is QualityStatus.WARNING
    assert calibration.correspondences[4].inlier is False
    assert calibration.correspondences[4].image_error_px > 100
    expected = CourtPoint(4.2, 10.0)
    result = calibration.image_to_court(_project(known_court_to_image, expected))
    assert result.x_m == pytest.approx(expected.x_m, abs=1e-4)
    assert result.y_m == pytest.approx(expected.y_m, abs=1e-4)


def test_four_non_collinear_points_are_supported(
    calibration_source: CalibrationSource,
    known_court_to_image: np.ndarray,
) -> None:
    selected = _correspondences(known_court_to_image)
    calibration = fit_calibration(
        source=calibration_source,
        court=CourtDimensions(),
        correspondences=(selected[0], selected[1], selected[8], selected[9]),
    )

    assert calibration.inlier_count == 4
    assert calibration.fit_method is FitMethod.DIRECT_FOUR_POINT
    assert calibration.quality.status is QualityStatus.WARNING
    assert calibration.reprojection_error.all_rmse_image_px < 1e-3


def test_coherent_low_angle_clicks_use_all_points_instead_of_sacrificing_boundary(
    calibration_source: CalibrationSource,
) -> None:
    """Regress the observed case where strict RANSAC displaced a valid corner 133 px."""

    selected_coordinates = {
        "near_baseline_left": (147.6, 903.6),
        "near_baseline_right": (1002.0, 1058.4),
        "near_kitchen_left": (811.2, 883.2),
        "near_kitchen_right": (1597.2, 937.2),
        "near_centerline_kitchen_intersection": (1082.4, 902.4),
        "far_baseline_left": (1293.6, 864.0),
        "far_baseline_right": (1858.8, 884.4),
    }
    correspondences = tuple(
        CalibrationCorrespondence(
            landmark=landmark.name,
            label=landmark.label,
            image_point=ImagePoint(*selected_coordinates[landmark.name.value]),
            court_point=landmark.court_point,
        )
        for landmark in court_landmarks(CourtDimensions())
        if landmark.name.value in selected_coordinates
    )

    calibration = fit_calibration(
        source=calibration_source,
        court=CourtDimensions(),
        correspondences=correspondences,
    )

    assert calibration.fit_method is FitMethod.LEAST_SQUARES_ALL_POINTS
    assert calibration.inlier_count == 7
    assert calibration.reprojection_error.all_rmse_image_px == pytest.approx(11.63, abs=0.02)
    assert calibration.reprojection_error.all_max_image_px < 22
    assert calibration.quality.status is QualityStatus.WARNING
    assert any("wide-angle lens distortion" in warning for warning in calibration.quality.warnings)


def test_invalid_correspondence_configurations_are_rejected(
    calibration_source: CalibrationSource,
    known_court_to_image: np.ndarray,
) -> None:
    selected = _correspondences(known_court_to_image)
    with pytest.raises(InvalidCalibrationError, match="at least 4"):
        fit_calibration(
            source=calibration_source,
            court=CourtDimensions(),
            correspondences=selected[:3],
        )

    duplicate_image = (
        selected[0],
        selected[1],
        selected[2],
        CalibrationCorrespondence(
            landmark=selected[3].landmark,
            label=selected[3].label,
            image_point=selected[0].image_point,
            court_point=selected[3].court_point,
        ),
    )
    with pytest.raises(InvalidCalibrationError, match="image points must be unique"):
        fit_calibration(
            source=calibration_source,
            court=CourtDimensions(),
            correspondences=duplicate_image,
        )

    collinear_image = tuple(
        CalibrationCorrespondence(
            landmark=item.landmark,
            label=item.label,
            image_point=ImagePoint(100 + index * 10, 200 + index * 20),
            court_point=item.court_point,
        )
        for index, item in enumerate(selected[:4])
    )
    with pytest.raises(InvalidCalibrationError, match="image points are collinear"):
        fit_calibration(
            source=calibration_source,
            court=CourtDimensions(),
            correspondences=collinear_image,
        )


def test_out_of_frame_and_non_finite_points_are_rejected(
    calibration_source: CalibrationSource,
    known_court_to_image: np.ndarray,
) -> None:
    selected = list(_correspondences(known_court_to_image)[:4])
    selected[0] = CalibrationCorrespondence(
        landmark=selected[0].landmark,
        label=selected[0].label,
        image_point=ImagePoint(-1, 200),
        court_point=selected[0].court_point,
    )
    with pytest.raises(InvalidCalibrationError, match="inside the decoded frame"):
        fit_calibration(
            source=calibration_source,
            court=CourtDimensions(),
            correspondences=tuple(selected),
        )

    selected[0] = CalibrationCorrespondence(
        landmark=selected[0].landmark,
        label=selected[0].label,
        image_point=ImagePoint(float("nan"), 200),
        court_point=selected[0].court_point,
    )
    with pytest.raises(InvalidCalibrationError, match="must be finite"):
        fit_calibration(
            source=calibration_source,
            court=CourtDimensions(),
            correspondences=tuple(selected),
        )


def test_calibration_json_round_trip_preserves_transformations(
    calibration_source: CalibrationSource,
    known_court_to_image: np.ndarray,
    tmp_path: Path,
) -> None:
    calibration = _fit(calibration_source, known_court_to_image)
    output_path = save_calibration(calibration, tmp_path / "nested" / "calibration.json")

    loaded = load_calibration(output_path)
    raw = json.loads(output_path.read_text())

    assert raw["schema_version"] == 2
    assert raw["court"]["unit"] == "meters"
    assert raw["fit"]["method"] == "least_squares_all_points"
    assert raw["fit"]["correspondence_count"] == 10
    assert raw["quality"]["status"] == "pass"
    assert raw["source"]["frame_index"] == 307
    assert len(raw["homographies"]["image_to_court"]) == 3
    point = CourtPoint(1.1, 5.2)
    expected = calibration.court_to_image(point)
    actual = loaded.court_to_image(point)
    assert actual.x_px == pytest.approx(expected.x_px)
    assert actual.y_px == pytest.approx(expected.y_px)


def test_calibration_json_requires_json_extension(
    calibration_source: CalibrationSource,
    known_court_to_image: np.ndarray,
    tmp_path: Path,
) -> None:
    calibration = _fit(calibration_source, known_court_to_image)

    with pytest.raises(CalibrationIoError, match=r"\.json extension"):
        save_calibration(calibration, tmp_path / "calibration.txt")


def test_debug_artifacts_render_and_write(
    calibration_source: CalibrationSource,
    known_court_to_image: np.ndarray,
    tmp_path: Path,
) -> None:
    calibration = _fit(calibration_source, known_court_to_image)
    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)

    overlay = render_calibration_overlay(frame, calibration)
    topdown = render_court_topdown(frame, calibration)
    overlay_path = write_debug_image(overlay, tmp_path / "calibration-overlay.jpg")
    topdown_path = write_debug_image(topdown, tmp_path / "court-topdown.jpg")

    assert overlay.shape == frame.shape
    assert np.any(overlay != frame)
    assert topdown.shape[0] > topdown.shape[1]
    assert overlay_path.is_file()
    assert topdown_path.is_file()
