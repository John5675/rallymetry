"""Safe RacketVision static-annotation adapter for representation pretraining."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from pickleball_vision.errors import ShotModelInputError

SPORTS = ("badminton", "tabletennis", "tennis")
FRAME_WIDTH_PX = 1920.0
FRAME_HEIGHT_PX = 1080.0
FEATURE_COUNT = 28


@dataclass(frozen=True, slots=True)
class RacketVisionSequence:
    """One safely paired ball/racket sequence from an external sport clip."""

    sport: str
    match_id: str
    rally_id: str
    partition: str
    ball_csv_path: Path
    racket_json_path: Path

    @property
    def sequence_id(self) -> str:
        return f"{self.sport}/{self.match_id}/{self.rally_id}"


@dataclass(frozen=True, slots=True)
class RacketVisionManifest:
    """Validated external-data identity without importing pickle artifacts."""

    root: Path
    upstream_repo: str
    upstream_revision: str
    license_name: str
    sequences: tuple[RacketVisionSequence, ...]
    content_sha256: str

    def as_dict(self) -> dict[str, object]:
        by_sport = {sport: 0 for sport in SPORTS}
        by_partition = {partition: 0 for partition in ("train", "validation", "test")}
        for sequence in self.sequences:
            by_sport[sequence.sport] += 1
            by_partition[sequence.partition] += 1
        return {
            "recordType": "racketvision_safe_annotation_manifest",
            "schemaVersion": 1,
            "root": str(self.root),
            "upstreamRepo": self.upstream_repo,
            "upstreamRevision": self.upstream_revision,
            "license": self.license_name,
            "sequenceCount": len(self.sequences),
            "sequenceCountsBySport": by_sport,
            "sequenceCountsByPartition": by_partition,
            "contentSha256": self.content_sha256,
            "acceptedExtensions": [".csv", ".json", ".md"],
            "pickleLoaded": False,
            "pickleballSemanticLabelsProvided": False,
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise ShotModelInputError(f"unable to hash external annotation {path}: {error}") from error
    return digest.hexdigest()


def discover_racketvision_sequences(
    root: Path,
    *,
    upstream_revision: str,
) -> RacketVisionManifest:
    """Pair only safe static files and reject unsafe serialized objects."""

    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise ShotModelInputError(f"RacketVision root is not a directory: {resolved}")
    if not upstream_revision or len(upstream_revision) < 7:
        raise ShotModelInputError("RacketVision upstream revision must be an immutable revision")
    unsafe = [
        path
        for path in resolved.rglob("*")
        if path.is_file()
        and ".cache" not in path.parts
        and path.suffix.lower() in {".pkl", ".pickle"}
    ]
    if unsafe:
        raise ShotModelInputError(
            f"unsafe pickle input is forbidden; remove or exclude {unsafe[0]}"
        )
    readme = resolved / "README.md"
    if not readme.is_file():
        raise ShotModelInputError("RacketVision README.md is required for license provenance")
    try:
        readme_text = readme.read_text(encoding="utf-8").lower()
    except OSError as error:
        raise ShotModelInputError(f"unable to read RacketVision dataset card: {error}") from error
    if "license: mit" not in readme_text:
        raise ShotModelInputError(
            "RacketVision dataset card does not declare the expected MIT license"
        )

    sequences: list[RacketVisionSequence] = []
    digest = hashlib.sha256()
    for sport in SPORTS:
        partition_by_pair: dict[tuple[str, str], str] = {}
        for filename, partition in (
            ("train.json", "train"),
            ("val.json", "validation"),
            ("test.json", "test"),
        ):
            split_path = resolved / sport / "info" / filename
            try:
                raw_split = json.loads(split_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ShotModelInputError(
                    f"unable to read RacketVision split {split_path}: {error}"
                ) from error
            if not isinstance(raw_split, list):
                raise ShotModelInputError(f"RacketVision split {split_path} must be an array")
            for item in raw_split:
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or not all(isinstance(value, str) for value in item)
                ):
                    raise ShotModelInputError(f"RacketVision split {split_path} has invalid pair")
                pair = (cast(str, item[0]), cast(str, item[1]))
                if pair in partition_by_pair:
                    raise ShotModelInputError(
                        f"RacketVision sequence {sport}/{pair[0]}/{pair[1]} leaks across splits"
                    )
                partition_by_pair[pair] = partition
        ball_root = resolved / sport / "interp_ball"
        for ball_path in sorted(ball_root.glob("*/*/results.csv")):
            relative = ball_path.relative_to(ball_root)
            if len(relative.parts) != 3:
                continue
            match_id, rally_id, _ = relative.parts
            racket_path = resolved / sport / "merged_racket" / match_id / rally_id / "result.json"
            if not racket_path.is_file():
                continue
            sequence_partition = partition_by_pair.get((match_id, rally_id))
            if sequence_partition is None:
                raise ShotModelInputError(
                    f"RacketVision sequence {sport}/{match_id}/{rally_id} has no fixed split"
                )
            sequence = RacketVisionSequence(
                sport=sport,
                match_id=match_id,
                rally_id=rally_id,
                partition=sequence_partition,
                ball_csv_path=ball_path,
                racket_json_path=racket_path,
            )
            sequences.append(sequence)
            for artifact in (ball_path, racket_path):
                digest.update(str(artifact.relative_to(resolved)).encode("utf-8"))
                digest.update(_file_sha256(artifact).encode("ascii"))
    if not sequences:
        raise ShotModelInputError(
            "RacketVision has no paired interp_ball/merged_racket sequences; "
            "finish the safe download"
        )
    return RacketVisionManifest(
        root=resolved,
        upstream_repo="linfeng302/RacketVision",
        upstream_revision=upstream_revision,
        license_name="MIT",
        sequences=tuple(sequences),
        content_sha256=digest.hexdigest(),
    )


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ShotModelInputError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ShotModelInputError(f"{field} must be finite")
    return result


def _racket_features(raw_frame: object, *, field: str) -> list[float]:
    if not isinstance(raw_frame, list):
        raise ShotModelInputError(f"{field} must be an array")
    rackets: list[tuple[float, list[float]]] = []
    for instance_index, raw_instance in enumerate(raw_frame):
        if not isinstance(raw_instance, dict):
            continue
        instance = cast(dict[str, object], raw_instance)
        raw_keypoints = instance.get("keypoints")
        if not isinstance(raw_keypoints, list) or len(raw_keypoints) != 5:
            continue
        coordinates: list[float] = []
        x_values: list[float] = []
        valid = True
        for point_index, raw_point in enumerate(raw_keypoints):
            if not isinstance(raw_point, list) or len(raw_point) < 2:
                valid = False
                break
            x = _number(raw_point[0], f"{field}[{instance_index}].keypoints[{point_index}].x")
            y = _number(raw_point[1], f"{field}[{instance_index}].keypoints[{point_index}].y")
            coordinates.extend((x / FRAME_WIDTH_PX, y / FRAME_HEIGHT_PX))
            x_values.append(x)
        if not valid:
            continue
        raw_scores = instance.get("keypoint_scores")
        scores = (
            [_number(value, f"{field}[{instance_index}].keypoint_scores[]") for value in raw_scores]
            if isinstance(raw_scores, list) and raw_scores
            else [0.0]
        )
        features = [*coordinates, sum(scores) / len(scores), 1.0]
        rackets.append((sum(x_values) / len(x_values), features))
    rackets.sort(key=lambda item: item[0])
    result: list[float] = []
    for index in range(2):
        result.extend(rackets[index][1] if index < len(rackets) else [0.0] * 12)
    return result


def load_racketvision_features(sequence: RacketVisionSequence) -> NDArray[np.float32]:
    """Load normalized per-frame ball plus up-to-two-racket features safely."""

    ball_rows: list[tuple[int, float, float, float, float]] = []
    try:
        with sequence.ball_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Frame", "X", "Y", "Visibility", "Confidence"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ShotModelInputError(f"{sequence.sequence_id} ball CSV lacks required columns")
            for row_index, row in enumerate(reader):
                try:
                    frame = int(row["Frame"])
                    x = float(row["X"])
                    y = float(row["Y"])
                    visibility = float(row["Visibility"])
                    confidence = float(row["Confidence"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ShotModelInputError(
                        f"invalid ball row {row_index} in {sequence.sequence_id}: {error}"
                    ) from error
                ball_rows.append(
                    (
                        frame,
                        x / FRAME_WIDTH_PX if visibility > 0 else 0.0,
                        y / FRAME_HEIGHT_PX if visibility > 0 else 0.0,
                        1.0 if visibility > 0 else 0.0,
                        max(0.0, min(1.0, confidence)),
                    )
                )
    except OSError as error:
        raise ShotModelInputError(
            f"unable to read ball sequence {sequence.sequence_id}: {error}"
        ) from error
    try:
        raw_rackets = json.loads(sequence.racket_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShotModelInputError(
            f"unable to read racket sequence {sequence.sequence_id}: {error}"
        ) from error
    if not isinstance(raw_rackets, list):
        raise ShotModelInputError(f"{sequence.sequence_id} racket root must be an array")
    frame_count = max(len(ball_rows), len(raw_rackets))
    if frame_count == 0:
        return np.empty((0, FEATURE_COUNT), dtype=np.float32)
    features = np.zeros((frame_count, FEATURE_COUNT), dtype=np.float32)
    for expected_frame, x, y, visibility, confidence in ball_rows:
        if expected_frame < 0 or expected_frame >= frame_count:
            continue
        features[expected_frame, :4] = (x, y, visibility, confidence)
    for frame, raw_frame in enumerate(raw_rackets):
        features[frame, 4:] = np.asarray(
            _racket_features(raw_frame, field=f"{sequence.sequence_id}.rackets[{frame}]"),
            dtype=np.float32,
        )
    return features
