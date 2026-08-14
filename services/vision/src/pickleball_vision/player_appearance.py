"""Lightweight clothing appearance evidence for logical player identity resolution."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pickleball_vision.person_detection import PersonDetection
from pickleball_vision.player_isolation import LogicalPlayerAssignments, LogicalPlayerRole
from pickleball_vision.player_tracking import RawTrackerObservation
from pickleball_vision.video import Image, iter_video_frames

HISTOGRAM_BINS = (8, 4, 4)
TORSO_BANDS = ((0.10, 0.46), (0.46, 0.76))


@dataclass(frozen=True, slots=True)
class AppearanceDescriptor:
    """Normalized two-band HSV histogram for one detected person's clothing crop."""

    values: tuple[float, ...]
    crop_width_px: int
    crop_height_px: int
    quality: float

    def __post_init__(self) -> None:
        if not self.values or not all(math.isfinite(value) for value in self.values):
            raise ValueError("appearance descriptor values must be finite and nonempty")
        if self.crop_width_px < 1 or self.crop_height_px < 1:
            raise ValueError("appearance crop dimensions must be positive")
        if not 0 <= self.quality <= 1 or not math.isfinite(self.quality):
            raise ValueError("appearance quality must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class AppearancePrototype:
    """Robust role appearance aggregated around its manual identity anchor."""

    logical_player: LogicalPlayerRole
    values: tuple[float, ...]
    sample_count: int
    anchor_tracker_id: int
    window_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "logical_player": self.logical_player.value,
            "method": "mean_two_band_hsv_histogram",
            "sample_count": self.sample_count,
            "anchor_tracker_id": self.anchor_tracker_id,
            "prototype_window_seconds": self.window_seconds,
            "descriptor_dimensions": len(self.values),
        }


def extract_appearance_descriptor(
    frame: Image,
    detection: PersonDetection,
) -> AppearanceDescriptor | None:
    """Extract central upper/lower clothing histograms from a source-pixel box."""

    box = detection.bounding_box
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, math.floor(box.left_px)))
    right = max(0, min(width, math.ceil(box.right_px)))
    top = max(0, min(height - 1, math.floor(box.top_px)))
    bottom = max(0, min(height, math.ceil(box.bottom_px)))
    if right - left < 4 or bottom - top < 8:
        return None
    person_crop = frame[top:bottom, left:right]
    crop_height, crop_width = person_crop.shape[:2]
    central_left = max(0, round(crop_width * 0.12))
    central_right = min(crop_width, round(crop_width * 0.88))
    if central_right - central_left < 2:
        return None

    histograms: list[np.ndarray] = []
    for start_fraction, end_fraction in TORSO_BANDS:
        band_top = max(0, round(crop_height * start_fraction))
        band_bottom = min(crop_height, round(crop_height * end_fraction))
        band = person_crop[band_top:band_bottom, central_left:central_right]
        if band.size == 0:
            return None
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            list(HISTOGRAM_BINS),
            [0, 180, 0, 256, 0, 256],
        ).reshape(-1)
        histogram = np.sqrt(histogram / max(float(histogram.sum()), 1.0))
        histograms.append(histogram)
    values = np.concatenate(histograms).astype(np.float64)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0:
        return None
    values /= norm
    torso_pixels = (central_right - central_left) * round(crop_height * 0.66)
    quality = min(1.0, max(0.1, torso_pixels / 5000))
    return AppearanceDescriptor(
        values=tuple(float(value) for value in values),
        crop_width_px=crop_width,
        crop_height_px=crop_height,
        quality=quality,
    )


def extract_tracker_appearance(
    video_path: Path,
    *,
    detections: tuple[PersonDetection, ...],
    raw_observations: tuple[RawTrackerObservation, ...],
) -> dict[str, AppearanceDescriptor]:
    """Decode once and derive appearance evidence for every emitted tracker observation."""

    observations_by_frame: dict[int, list[RawTrackerObservation]] = defaultdict(list)
    for observation in raw_observations:
        observations_by_frame[observation.frame_number].append(observation)
    descriptors: dict[str, AppearanceDescriptor] = {}
    for decoded in iter_video_frames(video_path):
        for observation in observations_by_frame.get(decoded.frame_index, ()):
            descriptor = extract_appearance_descriptor(
                decoded.image,
                detections[observation.raw_detection_index],
            )
            if descriptor is not None:
                descriptors[observation.observation_id] = descriptor
    return descriptors


def build_appearance_prototypes(
    *,
    raw_observations: tuple[RawTrackerObservation, ...],
    descriptors: dict[str, AppearanceDescriptor],
    assignments: LogicalPlayerAssignments,
    window_seconds: float,
) -> dict[LogicalPlayerRole, AppearancePrototype]:
    """Build one robust prototype from the anchor tracker near each manual anchor."""

    by_detection = {item.raw_detection_index: item for item in raw_observations}
    prototypes: dict[LogicalPlayerRole, AppearancePrototype] = {}
    for assignment in assignments.assignments:
        anchor = by_detection.get(assignment.anchor_detection_index)
        if anchor is None:
            raise ValueError(f"manual anchor for {assignment.logical_player.value} has no track")
        values = [
            np.asarray(descriptors[item.observation_id].values, dtype=np.float64)
            for item in raw_observations
            if item.tracker_id == anchor.tracker_id
            and abs(item.timestamp_s - assignment.anchor_timestamp_s) <= window_seconds
            and item.observation_id in descriptors
        ]
        if not values:
            raise ValueError(
                f"manual anchor for {assignment.logical_player.value} has no usable appearance crop"
            )
        prototype = np.mean(np.stack(values), axis=0)
        norm = float(np.linalg.norm(prototype))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(
                f"appearance prototype for {assignment.logical_player.value} is degenerate"
            )
        prototype /= norm
        prototypes[assignment.logical_player] = AppearancePrototype(
            assignment.logical_player,
            tuple(float(value) for value in prototype),
            len(values),
            anchor.tracker_id,
            window_seconds,
        )
    return prototypes


def appearance_similarity(
    descriptor: AppearanceDescriptor | None,
    prototype: AppearancePrototype,
) -> float | None:
    """Return bounded cosine similarity between normalized histogram vectors."""

    if descriptor is None or len(descriptor.values) != len(prototype.values):
        return None
    similarity = sum(
        first * second for first, second in zip(descriptor.values, prototype.values, strict=True)
    )
    return min(1.0, max(0.0, similarity))
