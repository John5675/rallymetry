"""Provenance-safe AI visual-review overlays for exact reviewed source videos."""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import cast

from pickleball_vision.errors import ShotModelInputError

_BUNDLED_INDEX = "resources/eight_video_ai_review_v1.json"


@dataclass(frozen=True, slots=True)
class ShotReviewOverlayArtifacts:
    """Result of enriching machine shots without changing their predictions."""

    output_path: Path
    matched_source: bool
    source_video_id: str | None
    shot_count: int
    matched_shot_count: int
    unused_review_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "outputPath": str(self.output_path),
            "matchedSource": self.matched_source,
            "sourceVideoId": self.source_video_id,
            "shotCount": self.shot_count,
            "matchedShotCount": self.matched_shot_count,
            "unusedReviewCount": self.unused_review_count,
            "machinePredictionsMutated": False,
            "humanCorrectionsCreated": False,
        }


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ShotModelInputError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ShotModelInputError(f"{field} must be an array")
    return cast(list[object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ShotModelInputError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShotModelInputError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ShotModelInputError(f"{field} must be finite")
    return result


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShotModelInputError(f"unable to read {label} {path}: {error}") from error
    return _object(raw, label)


def _read_review_index(path: Path | None) -> dict[str, object]:
    if path is not None:
        return _read_json(path, label="AI review index")
    try:
        raw = json.loads(files("pickleball_vision").joinpath(_BUNDLED_INDEX).read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShotModelInputError(f"unable to read bundled AI review index: {error}") from error
    return _object(raw, "AI review index")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.expanduser().resolve().open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise ShotModelInputError(f"unable to hash source video {path}: {error}") from error
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: object) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, resolved)
    except OSError as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise ShotModelInputError(
            f"unable to write AI-reviewed shots {resolved}: {error}"
        ) from error


def _validate_index(root: dict[str, object]) -> tuple[str, list[dict[str, object]]]:
    if root.get("schemaVersion") != 1:
        raise ShotModelInputError("AI review index schemaVersion must be 1")
    if root.get("recordType") != "ai_reviewed_shot_label_index":
        raise ShotModelInputError("AI review index recordType must be ai_reviewed_shot_label_index")
    review_version = _string(root.get("reviewVersion"), "reviewVersion")
    provenance = _object(root.get("provenance"), "provenance")
    if provenance.get("humanAccepted") is not False or provenance.get("groundTruth") is not False:
        raise ShotModelInputError(
            "AI review index must explicitly declare humanAccepted=false and groundTruth=false"
        )
    videos = [
        _object(item, f"videos[{index}]")
        for index, item in enumerate(_array(root.get("videos"), "videos"))
    ]
    source_hashes: set[str] = set()
    for index, video in enumerate(videos):
        source_hash = _string(video.get("sourceSha256"), f"videos[{index}].sourceSha256")
        if source_hash in source_hashes:
            raise ShotModelInputError("AI review index contains a duplicate sourceSha256")
        source_hashes.add(source_hash)
        _string(video.get("videoId"), f"videos[{index}].videoId")
        _array(video.get("reviews"), f"videos[{index}].reviews")
    return review_version, videos


def _match_reviews(
    shots: list[dict[str, object]],
    reviews: list[dict[str, object]],
    *,
    maximum_timing_delta_seconds: float,
) -> tuple[int, int]:
    unused = set(range(len(reviews)))
    matched = 0
    ordered_shots = sorted(
        shots,
        key=lambda item: _number(item.get("contactTimestamp"), "shots[].contactTimestamp"),
    )
    for shot in ordered_shots:
        timestamp = _number(shot.get("contactTimestamp"), "shots[].contactTimestamp")
        candidates: list[tuple[float, int]] = []
        for index in unused:
            review = reviews[index]
            reviewed = _object(review.get("aiReview"), f"reviews[{index}].aiReview")
            reviewed_timestamp = _number(
                reviewed.get("timestampSeconds"),
                f"reviews[{index}].aiReview.timestampSeconds",
            )
            candidates.append((abs(timestamp - reviewed_timestamp), index))
        if not candidates:
            continue
        delta, review_index = min(candidates)
        if delta > maximum_timing_delta_seconds:
            continue
        review = reviews[review_index]
        reviewed = _object(review.get("aiReview"), f"reviews[{review_index}].aiReview")
        if reviewed.get("humanAccepted") is not False:
            raise ShotModelInputError("AI review records must declare humanAccepted=false")
        shot["aiVisualReview"] = {
            "sourceContactId": _string(
                review.get("sourceContactId"),
                f"reviews[{review_index}].sourceContactId",
            ),
            "matchMethod": "exact_source_sha256_and_nearest_contact_timestamp",
            "timingDeltaMs": round(delta * 1000.0, 3),
            "review": deepcopy(reviewed),
            "predictionPreserved": True,
            "humanCorrectionCreated": False,
        }
        unused.remove(review_index)
        matched += 1
    return matched, len(unused)


def apply_ai_shot_review_overlay(
    source_video_path: Path,
    *,
    shots_path: Path,
    output_path: Path,
    review_index_path: Path | None = None,
    maximum_timing_delta_ms: float = 250.0,
) -> ShotReviewOverlayArtifacts:
    """Attach AI review evidence only when the exact reviewed media hash matches.

    The original ``shotType``, hitter, confidence, and classification evidence are
    never changed. The overlay is not a human correction and is not consumed by
    deterministic analytics.
    """

    if not math.isfinite(maximum_timing_delta_ms) or maximum_timing_delta_ms <= 0:
        raise ShotModelInputError("maximum timing delta must be positive and finite")
    source = source_video_path.expanduser().resolve()
    source_hash = _sha256(source)
    index = _read_review_index(review_index_path)
    review_version, videos = _validate_index(index)
    matching = [video for video in videos if video["sourceSha256"] == source_hash]
    if len(matching) > 1:
        raise ShotModelInputError("AI review index matched more than one source video")

    shots_root = _read_json(shots_path, label="reconstructed shots")
    raw_shots = _array(shots_root.get("shots"), "shots")
    shots = [deepcopy(_object(item, f"shots[{index}]")) for index, item in enumerate(raw_shots)]
    matched_shot_count = 0
    unused_review_count = 0
    source_video_id: str | None = None
    if matching:
        video = matching[0]
        source_video_id = _string(video.get("videoId"), "videos[].videoId")
        reviews = [
            _object(item, f"videos.{source_video_id}.reviews[{index}]")
            for index, item in enumerate(_array(video.get("reviews"), "videos[].reviews"))
        ]
        matched_shot_count, unused_review_count = _match_reviews(
            shots,
            reviews,
            maximum_timing_delta_seconds=maximum_timing_delta_ms / 1000.0,
        )

    output = deepcopy(shots_root)
    output["shots"] = shots
    output["aiReviewOverlay"] = {
        "schemaVersion": 1,
        "reviewVersion": review_version,
        "matchedSource": bool(matching),
        "sourceVideoId": source_video_id,
        "sourceSha256": source_hash,
        "shotCount": len(shots),
        "matchedShotCount": matched_shot_count,
        "unusedReviewCount": unused_review_count,
        "maximumTimingDeltaMs": maximum_timing_delta_ms,
        "provenance": "AI_VISUAL_REVIEW",
        "humanAccepted": False,
        "groundTruth": False,
        "machinePredictionsMutated": False,
        "humanCorrectionsCreated": False,
        "analyticsUsage": "excluded_unless_later_verified_by_a_human_correction",
    }
    resolved_output = output_path.expanduser().resolve()
    _atomic_write_json(resolved_output, output)
    return ShotReviewOverlayArtifacts(
        output_path=resolved_output,
        matched_source=bool(matching),
        source_video_id=source_video_id,
        shot_count=len(shots),
        matched_shot_count=matched_shot_count,
        unused_review_count=unused_review_count,
    )
