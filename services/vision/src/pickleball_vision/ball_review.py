"""Local browser interface for human review of detector training annotations."""

from __future__ import annotations

import json
import math
import mimetypes
import threading
import webbrowser
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from pickleball_vision.ball_dataset import (
    BALL_ANNOTATION_SCHEMA_VERSION,
    BallCourtSide,
    BallScope,
    BallVisibility,
    create_ball_annotation_template,
)
from pickleball_vision.dataset import DatasetSplit
from pickleball_vision.errors import DatasetInputError, DatasetIoError
from pickleball_vision.person_detection import BoundingBox

REVIEW_SUMMARY_SCHEMA_VERSION = 1
MAXIMUM_REQUEST_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ReviewFrame:
    """One fixed-split source image available for human review."""

    index: int
    record_id: str
    source_id: str
    source_media_path: Path
    image_path: Path
    frame_number: int
    timestamp_s: float
    width: int
    height: int
    split: DatasetSplit
    clip_id: str | None
    group_id: str | None


@dataclass(frozen=True, slots=True)
class BallReviewArtifacts:
    """Paths and final progress from one local review-server session."""

    url: str
    annotations_path: Path
    summary_path: Path
    reviewed_frames: int
    total_frames: int

    def as_dict(self) -> dict[str, object]:
        return {
            "url": self.url,
            "annotations_path": str(self.annotations_path),
            "summary_path": str(self.summary_path),
            "reviewed_frames": self.reviewed_frames,
            "total_frames": self.total_frames,
        }


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast(list[object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _read_json(path: Path, kind: str) -> dict[str, object]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), "root")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise DatasetIoError(str(path), reason=f"unable to load {kind}: {error}") from error


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise DatasetIoError(str(path), reason=f"unable to save review data: {error}") from error


def _resolve_manifest_path(value: str, *, split_path: Path) -> Path:
    path = Path(value).expanduser()
    return (split_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _relative_artifact_path(value: object, field: str) -> Path:
    raw = Path(_string(value, field))
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{field} must be a safe relative path")
    return raw


def _load_review_frames(split_path: Path) -> tuple[ReviewFrame, ...]:
    split_root = _read_json(split_path, "dataset split manifest")
    try:
        if split_root.get("schema_version") != 1 or split_root.get("record_type") != (
            "ball_dataset_split_assignments"
        ):
            raise ValueError("split manifest has an unsupported schema or record_type")
        manifest_cache: dict[Path, dict[str, object]] = {}
        source_frame_cache: dict[Path, dict[str, dict[str, object]]] = {}
        frames: list[ReviewFrame] = []
        seen_record_ids: set[str] = set()
        for index, raw_value in enumerate(_array(split_root.get("frames"), "frames")):
            field = f"frames[{index}]"
            split_item = _object(raw_value, field)
            record_id = _string(split_item.get("record_id"), f"{field}.record_id")
            if record_id in seen_record_ids:
                raise ValueError(f"duplicate record_id {record_id!r}")
            seen_record_ids.add(record_id)
            manifest_path = _resolve_manifest_path(
                _string(split_item.get("manifest_path"), f"{field}.manifest_path"),
                split_path=split_path,
            )
            if manifest_path not in manifest_cache:
                manifest = _read_json(manifest_path, "source dataset manifest")
                if manifest.get("schema_version") != 1 or manifest.get("record_type") != (
                    "ball_annotation_frame_dataset"
                ):
                    raise ValueError(f"unsupported source dataset manifest {manifest_path}")
                manifest_cache[manifest_path] = manifest
                source_frame_cache[manifest_path] = {
                    _string(_object(item, "source frame").get("record_id"), "record_id"): _object(
                        item, "source frame"
                    )
                    for item in _array(manifest.get("frames"), "source.frames")
                }
            manifest = manifest_cache[manifest_path]
            source_frame = source_frame_cache[manifest_path].get(record_id)
            if source_frame is None:
                raise ValueError(f"record {record_id!r} is missing from {manifest_path}")
            source = _object(manifest.get("source"), "source")
            source_id = _string(source.get("source_id"), "source.source_id")
            if source_id != _string(split_item.get("source_id"), f"{field}.source_id"):
                raise ValueError(f"{field}.source_id differs from its source manifest")
            media = _object(source.get("media"), "source.media")
            relative_image = _relative_artifact_path(
                source_frame.get("relative_image_path"), f"{record_id}.relative_image_path"
            )
            image_path = (manifest_path.parent / relative_image).resolve()
            if not image_path.is_file():
                raise ValueError(f"image for record {record_id!r} does not exist: {image_path}")
            try:
                split = DatasetSplit(_string(split_item.get("split"), f"{field}.split"))
            except ValueError as error:
                raise ValueError(f"{field}.split is unsupported") from error
            frame_number = _integer(source_frame.get("frame_number"), f"{record_id}.frame_number")
            if frame_number < 0:
                raise ValueError(f"{record_id}.frame_number must be non-negative")
            width = _integer(source_frame.get("width"), f"{record_id}.width")
            height = _integer(source_frame.get("height"), f"{record_id}.height")
            if width < 1 or height < 1:
                raise ValueError(f"{record_id} has invalid image dimensions")
            frames.append(
                ReviewFrame(
                    index=index,
                    record_id=record_id,
                    source_id=source_id,
                    source_media_path=Path(_string(media.get("path"), "source.media.path"))
                    .expanduser()
                    .resolve(),
                    image_path=image_path,
                    frame_number=frame_number,
                    timestamp_s=_number(source_frame.get("timestamp_s"), "timestamp_s"),
                    width=width,
                    height=height,
                    split=split,
                    clip_id=_optional_string(source_frame.get("clip_id"), "clip_id"),
                    group_id=_optional_string(source_frame.get("group_id"), "group_id"),
                )
            )
        if not frames:
            raise ValueError("split manifest contains no frames")
        return tuple(frames)
    except ValueError as error:
        raise DatasetInputError(f"invalid annotation-review split: {error}") from error


def _validate_box(value: object, *, field: str, width: int, height: int) -> BoundingBox:
    raw = _object(value, field)
    box = BoundingBox(
        left_px=_number(raw.get("left_px"), f"{field}.left_px"),
        top_px=_number(raw.get("top_px"), f"{field}.top_px"),
        right_px=_number(raw.get("right_px"), f"{field}.right_px"),
        bottom_px=_number(raw.get("bottom_px"), f"{field}.bottom_px"),
    )
    if box.right_px > width or box.bottom_px > height:
        raise ValueError(f"{field} lies outside the {width}x{height} source image")
    return box


def _load_suggestions(
    prediction_paths: Sequence[Path],
    frames: tuple[ReviewFrame, ...],
) -> dict[int, tuple[dict[str, object], ...]]:
    suggestions: dict[int, list[dict[str, object]]] = {frame.index: [] for frame in frames}
    source_paths = {frame.source_media_path: frame.source_id for frame in frames}
    frames_by_source_number = {(frame.source_id, frame.frame_number): frame for frame in frames}
    for prediction_path in prediction_paths:
        resolved = prediction_path.expanduser().resolve()
        root = _read_json(resolved, "raw pickleball detections")
        try:
            if root.get("schema_version") != 1 or root.get("record_type") != (
                "raw_pickleball_detections"
            ):
                raise ValueError("predictions have an unsupported schema or record_type")
            temporal = _object(root.get("temporal_processing"), "temporal_processing")
            if any(
                temporal.get(key) is not False for key in ("tracking", "interpolation", "events")
            ):
                raise ValueError("review suggestions must be raw non-temporal detections")
            source = _object(root.get("source"), "source")
            source_path = Path(_string(source.get("path"), "source.path")).expanduser().resolve()
            source_id = source_paths.get(source_path)
            if source_id is None:
                raise ValueError(
                    f"prediction source {source_path} is not represented by the fixed split"
                )
            detector = _object(root.get("detector"), "detector")
            strategy = _object(root.get("strategy"), "strategy")
            for frame_value in _array(root.get("frames"), "prediction.frames"):
                prediction_frame = _object(frame_value, "prediction frame")
                frame_number = _integer(
                    prediction_frame.get("frame_number"), "prediction.frame_number"
                )
                review_frame = frames_by_source_number.get((source_id, frame_number))
                if review_frame is None:
                    continue
                for detection_value in _array(
                    prediction_frame.get("detections"), "prediction.detections"
                ):
                    detection = _object(detection_value, "prediction detection")
                    box = _validate_box(
                        detection.get("bounding_box"),
                        field="prediction.bounding_box",
                        width=review_frame.width,
                        height=review_frame.height,
                    )
                    confidence = _number(detection.get("confidence"), "prediction.confidence")
                    if not 0 <= confidence <= 1:
                        raise ValueError("prediction.confidence must be in [0, 1]")
                    suggestions[review_frame.index].append(
                        {
                            "suggestion_id": _string(
                                detection.get("detection_id"), "prediction.detection_id"
                            ),
                            "bounding_box": box.as_dict(),
                            "confidence": confidence,
                            "model_version": detector.get("model_version"),
                            "weights_sha256": detector.get("weights_sha256"),
                            "strategy": strategy.get("name"),
                            "prediction_file": str(resolved),
                        }
                    )
        except ValueError as error:
            raise DatasetInputError(f"invalid review predictions {resolved}: {error}") from error
    return {
        index: tuple(sorted(items, key=lambda item: cast(float, item["confidence"]), reverse=True))
        for index, items in suggestions.items()
    }


class BallAnnotationReviewStore:
    """Thread-safe, crash-resistant bridge between review UI and ground truth."""

    def __init__(
        self,
        split_manifest_path: Path,
        *,
        annotations_path: Path,
        dataset_version: str | None = None,
        prediction_paths: Sequence[Path] = (),
    ) -> None:
        self.split_manifest_path = split_manifest_path.expanduser().resolve()
        self.annotations_path = annotations_path.expanduser().resolve()
        self.summary_path = self.annotations_path.with_name(
            f"{self.annotations_path.stem}.review-summary.json"
        )
        self.frames = _load_review_frames(self.split_manifest_path)
        if not self.annotations_path.exists():
            if dataset_version is None or not dataset_version.strip():
                raise DatasetInputError(
                    "--dataset-version is required when creating a new annotations file"
                )
            create_ball_annotation_template(
                self.split_manifest_path,
                dataset_version=dataset_version,
                output_path=self.annotations_path,
            )
        self._root = _read_json(self.annotations_path, "ball annotations")
        self._validate_annotations(dataset_version)
        annotation_frames = _array(self._root.get("frames"), "annotation.frames")
        self._annotations_by_id = {
            _string(_object(item, "annotation frame").get("record_id"), "record_id"): _object(
                item, "annotation frame"
            )
            for item in annotation_frames
        }
        self._annotation_positions = {
            _string(_object(item, "annotation frame").get("record_id"), "record_id"): index
            for index, item in enumerate(annotation_frames)
        }
        self.suggestions = _load_suggestions(prediction_paths, self.frames)
        self.prediction_paths = tuple(path.expanduser().resolve() for path in prediction_paths)
        self._lock = threading.RLock()
        self._write_summary()

    def _validate_annotations(self, dataset_version: str | None) -> None:
        try:
            if self._root.get("schema_version") != BALL_ANNOTATION_SCHEMA_VERSION:
                raise ValueError("annotations have an unsupported schema_version")
            if self._root.get("record_type") != "pickleball_detection_annotations":
                raise ValueError("annotation record_type must be pickleball_detection_annotations")
            if self._root.get("class_name") != "pickleball":
                raise ValueError("annotation class_name must be pickleball")
            actual_version = _string(self._root.get("dataset_version"), "dataset_version")
            if dataset_version is not None and actual_version != dataset_version:
                raise ValueError(
                    f"annotations use dataset version {actual_version!r}, not {dataset_version!r}"
                )
            records = [
                _object(item, "annotation frame")
                for item in _array(self._root.get("frames"), "frames")
            ]
            record_ids = [_string(item.get("record_id"), "record_id") for item in records]
            if len(record_ids) != len(set(record_ids)):
                raise ValueError("annotations contain duplicate record IDs")
            expected = {frame.record_id for frame in self.frames}
            actual = set(record_ids)
            if actual != expected:
                missing = len(expected - actual)
                extra = len(actual - expected)
                raise ValueError(
                    f"annotations do not match fixed split ({missing} missing, {extra} extra)"
                )
        except ValueError as error:
            raise DatasetInputError(f"invalid annotation review file: {error}") from error

    def _annotation(self, frame: ReviewFrame) -> dict[str, object]:
        return self._annotations_by_id[frame.record_id]

    def _summary_payload(self) -> dict[str, object]:
        frame_rows: list[dict[str, object]] = []
        positive = 0
        negative = 0
        reviewed = 0
        object_count = 0
        split_counts: dict[str, dict[str, int]] = {
            split.value: {"total": 0, "reviewed": 0, "positive": 0, "negative": 0}
            for split in DatasetSplit
        }
        for frame in self.frames:
            annotation = self._annotation(frame)
            status = annotation.get("review_status", "unreviewed")
            objects = _array(annotation.get("objects", []), f"{frame.record_id}.objects")
            is_reviewed = status == "reviewed"
            outcome = "unreviewed"
            if is_reviewed:
                reviewed += 1
                split_counts[frame.split.value]["reviewed"] += 1
                if objects:
                    positive += 1
                    split_counts[frame.split.value]["positive"] += 1
                    outcome = "positive"
                else:
                    negative += 1
                    split_counts[frame.split.value]["negative"] += 1
                    outcome = "negative"
            split_counts[frame.split.value]["total"] += 1
            object_count += len(objects)
            frame_suggestions = self.suggestions.get(frame.index, ())
            confidences = [cast(float, item["confidence"]) for item in frame_suggestions]
            frame_rows.append(
                {
                    "index": frame.index,
                    "record_id": frame.record_id,
                    "source_id": frame.source_id,
                    "frame_number": frame.frame_number,
                    "timestamp_s": frame.timestamp_s,
                    "split": frame.split.value,
                    "review_status": status,
                    "review_outcome": outcome,
                    "object_count": len(objects),
                    "suggestion_count": len(frame_suggestions),
                    "maximum_suggestion_confidence": max(confidences) if confidences else None,
                }
            )
        total = len(self.frames)
        return {
            "schema_version": REVIEW_SUMMARY_SCHEMA_VERSION,
            "record_type": "pickleball_annotation_review_summary",
            "annotations_path": str(self.annotations_path),
            "split_manifest_path": str(self.split_manifest_path),
            "dataset_version": self._root["dataset_version"],
            "prediction_files": [str(path) for path in self.prediction_paths],
            "contracts": {
                "model_suggestions_are_ground_truth": False,
                "explicit_human_review_required": True,
                "unreviewed_is_negative": False,
                "source_images_modified": False,
            },
            "counts": {
                "total_frames": total,
                "reviewed_frames": reviewed,
                "unreviewed_frames": total - reviewed,
                "positive_frames": positive,
                "negative_frames": negative,
                "annotation_objects": object_count,
                "frames_with_suggestions": sum(
                    bool(self.suggestions.get(frame.index)) for frame in self.frames
                ),
            },
            "by_split": split_counts,
            "frames": frame_rows,
        }

    def _write_summary(self) -> None:
        _write_json_atomic(self.summary_path, self._summary_payload())

    def session_payload(self) -> dict[str, object]:
        with self._lock:
            return self._summary_payload()

    def frame_payload(self, index: int) -> dict[str, object]:
        with self._lock:
            frame = self.frame(index)
            return {
                "index": frame.index,
                "record_id": frame.record_id,
                "source_id": frame.source_id,
                "frame_number": frame.frame_number,
                "timestamp_s": frame.timestamp_s,
                "width": frame.width,
                "height": frame.height,
                "split": frame.split.value,
                "clip_id": frame.clip_id,
                "group_id": frame.group_id,
                "image_url": f"/api/frames/{frame.index}/image",
                "annotation": self._annotation(frame),
                "suggestions": list(self.suggestions.get(frame.index, ())),
            }

    def frame(self, index: int) -> ReviewFrame:
        if index < 0 or index >= len(self.frames):
            raise DatasetInputError(f"review frame index {index} is out of range")
        return self.frames[index]

    def update_frame(self, index: int, payload: object) -> dict[str, object]:
        with self._lock:
            frame = self.frame(index)
            try:
                raw = _object(payload, "review update")
                status = _string(raw.get("review_status"), "review_status")
                if status not in {"reviewed", "unreviewed"}:
                    raise ValueError("review_status must be reviewed or unreviewed")
                reviewer = _string(raw.get("reviewer", "local-reviewer"), "reviewer")
                objects: list[dict[str, object]] = []
                seen_ids: set[str] = set()
                for object_index, value in enumerate(_array(raw.get("objects"), "objects")):
                    field = f"objects[{object_index}]"
                    item = _object(value, field)
                    annotation_id_value = item.get("annotation_id")
                    annotation_id = (
                        _string(annotation_id_value, f"{field}.annotation_id")
                        if annotation_id_value is not None
                        else f"manual-ball-{frame.frame_number:09d}-{object_index + 1:02d}"
                    )
                    if annotation_id in seen_ids:
                        raise ValueError(f"duplicate annotation_id {annotation_id!r}")
                    seen_ids.add(annotation_id)
                    box = _validate_box(
                        item.get("bounding_box"),
                        field=f"{field}.bounding_box",
                        width=frame.width,
                        height=frame.height,
                    )
                    try:
                        court_side = BallCourtSide(
                            _string(item.get("court_side", "unknown"), f"{field}.court_side")
                        )
                        scope = BallScope(_string(item.get("scope", "unknown"), f"{field}.scope"))
                        visibility = BallVisibility(
                            _string(item.get("visibility", "clear"), f"{field}.visibility")
                        )
                    except ValueError as error:
                        raise ValueError(f"{field} has unsupported annotation metadata") from error
                    if status == "reviewed" and visibility is BallVisibility.AMBIGUOUS:
                        raise ValueError(
                            "ambiguous boxes cannot be promoted to reviewed ground truth"
                        )
                    confidence = _number(
                        item.get("annotation_confidence", 1.0),
                        f"{field}.annotation_confidence",
                    )
                    if not 0 <= confidence <= 1:
                        raise ValueError(f"{field}.annotation_confidence must be in [0, 1]")
                    objects.append(
                        {
                            "annotation_id": annotation_id,
                            "class_name": "pickleball",
                            "bounding_box": box.as_dict(),
                            "court_side": court_side.value,
                            "scope": scope.value,
                            "visibility": visibility.value,
                            "annotation_confidence": confidence,
                        }
                    )
                annotation: dict[str, object] = {
                    "record_id": frame.record_id,
                    "review_status": status,
                    "review_outcome": (
                        "positive"
                        if status == "reviewed" and objects
                        else "negative"
                        if status == "reviewed"
                        else "unreviewed"
                    ),
                    "objects": objects,
                    "review_metadata": {
                        "interface": "pickleball-vision-ball-review-v1",
                        "reviewer": reviewer,
                        "updated_at_utc": datetime.now(UTC).isoformat(),
                        "model_suggestions_automatically_promoted": False,
                    },
                }
                self._annotations_by_id[frame.record_id] = annotation
                annotation_frames = cast(list[object], self._root["frames"])
                annotation_frames[self._annotation_positions[frame.record_id]] = annotation
                summary = self._summary_payload()
                self._root["review_summary"] = summary["counts"]
                self._root["last_reviewed_at_utc"] = datetime.now(UTC).isoformat()
                _write_json_atomic(self.annotations_path, self._root)
                _write_json_atomic(self.summary_path, summary)
                return self.frame_payload(index)
            except (KeyError, ValueError) as error:
                raise DatasetInputError(f"invalid annotation review update: {error}") from error


REVIEW_UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Pickleball Vision — Ball Review</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101418;
      --panel: #182027;
      --panel-2: #222c34;
      --text: #eef3f6;
      --muted: #9eb0bc;
      --green: #43e083;
      --yellow: #ffd84a;
      --red: #ff6b6b;
      --blue: #57b5ff;
      --border: #34434e;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.4 system-ui, sans-serif; }
    button, select, input { font: inherit; }
    button, select, input[type="text"] {
      color: var(--text); background: var(--panel-2); border: 1px solid var(--border);
      border-radius: 7px; padding: 7px 10px;
    }
    button { cursor: pointer; }
    button:hover { border-color: var(--blue); }
    button.primary { background: #176c43; border-color: #2bb56f; }
    button.negative { background: #703339; border-color: #b6535e; }
    button:disabled { cursor: default; opacity: .45; }
    #app { height: 100vh; display: grid; grid-template-rows: auto 1fr auto; }
    header { display: flex; gap: 14px; align-items: center; padding: 10px 14px; background: var(--panel); border-bottom: 1px solid var(--border); }
    header h1 { margin: 0; font-size: 17px; white-space: nowrap; }
    #progress { color: var(--muted); min-width: 200px; }
    header .spacer { flex: 1; }
    main { min-height: 0; display: grid; grid-template-columns: 280px 1fr; }
    aside { overflow: auto; padding: 12px; background: var(--panel); border-right: 1px solid var(--border); }
    aside h2 { margin: 0 0 9px; font-size: 14px; }
    .field { display: grid; gap: 4px; margin-bottom: 10px; }
    .field label { color: var(--muted); font-size: 12px; }
    .row { display: flex; gap: 7px; align-items: center; }
    .row > * { min-width: 0; flex: 1; }
    #metadata { margin: 12px 0; padding: 9px; border-radius: 7px; background: var(--panel-2); color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    #object-editor { border-top: 1px solid var(--border); padding-top: 12px; }
    #object-editor.disabled { opacity: .45; pointer-events: none; }
    .legend { display: grid; gap: 5px; margin-top: 14px; color: var(--muted); font-size: 12px; }
    .swatch { display: inline-block; width: 18px; height: 10px; margin-right: 6px; vertical-align: -1px; border: 2px solid; }
    .swatch.suggestion { border-color: var(--yellow); border-style: dashed; }
    .swatch.annotation { border-color: var(--green); }
    .swatch.selected { border-color: var(--blue); }
    #stage { min-width: 0; min-height: 0; overflow: auto; display: grid; place-items: start center; background: #07090b; }
    #canvas-wrap { position: relative; margin: 14px; box-shadow: 0 0 0 1px #000; }
    canvas { display: block; cursor: crosshair; image-rendering: auto; touch-action: none; }
    footer { display: flex; flex-wrap: wrap; gap: 8px; padding: 9px 12px; background: var(--panel); border-top: 1px solid var(--border); align-items: center; }
    footer .spacer { flex: 1; }
    #status { color: var(--muted); }
    #status.error { color: var(--red); }
    #dirty { color: var(--yellow); font-weight: 700; }
    kbd { background: #0c1115; border: 1px solid var(--border); border-bottom-width: 2px; border-radius: 4px; padding: 1px 5px; }
    @media (max-width: 850px) {
      main { grid-template-columns: 220px 1fr; }
      header { flex-wrap: wrap; }
    }
  </style>
</head>
<body>
<div id="app">
  <header>
    <h1>Pickleball Vision · Manual Ball Review</h1>
    <span id="progress">Loading…</span>
    <div class="spacer"></div>
    <label>Queue
      <select id="filter">
        <option value="unreviewed">Unreviewed</option>
        <option value="suggestions">With suggestions</option>
        <option value="low-confidence">Low-confidence suggestions</option>
        <option value="no-suggestions">Without suggestions</option>
        <option value="reviewed">Reviewed</option>
        <option value="all">All frames</option>
      </select>
    </label>
    <button id="previous">← Previous</button>
    <button id="next">Next →</button>
  </header>
  <main>
    <aside>
      <div class="field">
        <label for="reviewer">Reviewer</label>
        <input id="reviewer" type="text" value="local-reviewer">
      </div>
      <div class="row">
        <div class="field">
          <label for="frame-index">Queue index</label>
          <input id="frame-index" type="text" inputmode="numeric">
        </div>
        <button id="go-index">Go</button>
      </div>
      <div class="field">
        <label for="zoom">Zoom <span id="zoom-label"></span></label>
        <input id="zoom" type="range" min="20" max="300" step="5" value="65">
      </div>
      <button id="fit">Fit image</button>
      <div id="metadata"></div>
      <div class="row">
        <button id="accept-all">Accept all suggestions</button>
        <button id="clear-boxes">Clear boxes</button>
      </div>
      <div id="object-editor" class="disabled">
        <h2>Selected annotation</h2>
        <div class="field">
          <label for="court-side">Court side (human context)</label>
          <select id="court-side">
            <option value="unknown">Unknown</option>
            <option value="near">Near</option>
            <option value="far">Far</option>
          </select>
        </div>
        <div class="field">
          <label for="scope">Scope</label>
          <select id="scope">
            <option value="primary_match">Primary match</option>
            <option value="neighboring_court">Neighboring court</option>
            <option value="unknown">Unknown</option>
          </select>
        </div>
        <div class="field">
          <label for="visibility">Visibility</label>
          <select id="visibility">
            <option value="clear">Clear</option>
            <option value="blurred">Blurred</option>
            <option value="partial">Partial</option>
            <option value="ambiguous">Ambiguous draft only</option>
          </select>
        </div>
        <div class="field">
          <label for="confidence">Human annotation confidence</label>
          <input id="confidence" type="range" min="0" max="1" step="0.05" value="1">
          <span id="confidence-label">1.00</span>
        </div>
        <button id="delete-box" class="negative">Delete selected box</button>
      </div>
      <div class="legend">
        <span><i class="swatch suggestion"></i>Model suggestion — not ground truth</span>
        <span><i class="swatch annotation"></i>Human annotation</span>
        <span><i class="swatch selected"></i>Selected annotation</span>
        <span>Drag to draw · click yellow to accept · right-click green to remove</span>
      </div>
    </aside>
    <section id="stage"><div id="canvas-wrap"><canvas id="canvas"></canvas></div></section>
  </main>
  <footer>
    <button id="save-reviewed" class="primary"><kbd>R</kbd> Save reviewed</button>
    <button id="mark-negative" class="negative"><kbd>N</kbd> Reviewed negative</button>
    <button id="save-draft"><kbd>S</kbd> Save draft</button>
    <button id="unreview"><kbd>U</kbd> Mark unreviewed</button>
    <span id="dirty"></span>
    <div class="spacer"></div>
    <span id="status">Model suggestions never save automatically.</span>
    <button id="stop-server">Stop server</button>
  </footer>
</div>
<script>
"use strict";
const state = {
  session: null, index: 0, frame: null, image: null, boxes: [], selected: -1,
  dirty: false, dragStart: null, dragNow: null, scale: 0.65
};
const $ = id => document.getElementById(id);
const canvas = $("canvas"), ctx = canvas.getContext("2d"), stage = $("stage");

function setStatus(message, error=false) {
  $("status").textContent = message;
  $("status").className = error ? "error" : "";
}
function setDirty(value=true) {
  state.dirty = value;
  $("dirty").textContent = value ? "Unsaved changes" : "";
}
function humanBox(raw={}) {
  return {
    annotation_id: raw.annotation_id || null,
    class_name: "pickleball",
    bounding_box: {...raw.bounding_box},
    court_side: raw.court_side || "unknown",
    scope: raw.scope || "unknown",
    visibility: raw.visibility || "clear",
    annotation_confidence: raw.annotation_confidence ?? 1.0
  };
}
function imagePoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * canvas.width / rect.width)),
    y: Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * canvas.height / rect.height))
  };
}
function contains(box, point, padding=0) {
  const b = box.bounding_box;
  return point.x >= b.left_px-padding && point.x <= b.right_px+padding &&
         point.y >= b.top_px-padding && point.y <= b.bottom_px+padding;
}
function drawBox(box, color, dashed=false, width=3) {
  const b = box.bounding_box;
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = width / state.scale;
  ctx.setLineDash(dashed ? [10/state.scale, 7/state.scale] : []);
  ctx.strokeRect(b.left_px, b.top_px, b.right_px-b.left_px, b.bottom_px-b.top_px);
  ctx.restore();
}
function render() {
  if (!state.frame || !state.image) return;
  canvas.width = state.frame.width; canvas.height = state.frame.height;
  canvas.style.width = `${Math.round(state.frame.width * state.scale)}px`;
  canvas.style.height = `${Math.round(state.frame.height * state.scale)}px`;
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  for (const suggestion of state.frame.suggestions) drawBox(suggestion, "#ffd84a", true, 2);
  state.boxes.forEach((box, index) => drawBox(box, index === state.selected ? "#57b5ff" : "#43e083", false, index === state.selected ? 4 : 3));
  if (state.dragStart && state.dragNow) {
    drawBox({bounding_box: normalizedBox(state.dragStart, state.dragNow)}, "#57b5ff", true, 3);
  }
}
function normalizedBox(a, b) {
  return {left_px: Math.min(a.x,b.x), top_px: Math.min(a.y,b.y), right_px: Math.max(a.x,b.x), bottom_px: Math.max(a.y,b.y)};
}
function refreshEditor() {
  const box = state.boxes[state.selected];
  $("object-editor").classList.toggle("disabled", !box);
  if (!box) return;
  $("court-side").value = box.court_side;
  $("scope").value = box.scope;
  $("visibility").value = box.visibility;
  $("confidence").value = String(box.annotation_confidence);
  $("confidence-label").textContent = Number(box.annotation_confidence).toFixed(2);
}
function selectBox(index) { state.selected = index; refreshEditor(); render(); }
function addBox(box) { state.boxes.push(humanBox(box)); selectBox(state.boxes.length-1); setDirty(); }
function acceptSuggestion(suggestion) {
  const b = suggestion.bounding_box;
  const exists = state.boxes.some(box => {
    const a = box.bounding_box;
    return Math.abs(a.left_px-b.left_px) < 0.01 && Math.abs(a.top_px-b.top_px) < 0.01 &&
           Math.abs(a.right_px-b.right_px) < 0.01 && Math.abs(a.bottom_px-b.bottom_px) < 0.01;
  });
  if (!exists) addBox({bounding_box: b});
}
function removeSelected() {
  if (state.selected < 0) return;
  state.boxes.splice(state.selected, 1); state.selected = -1; refreshEditor(); setDirty(); render();
}
function queueMatch(row) {
  const filter = $("filter").value;
  if (filter === "all") return true;
  if (filter === "unreviewed") return row.review_status !== "reviewed";
  if (filter === "reviewed") return row.review_status === "reviewed";
  if (filter === "suggestions") return row.suggestion_count > 0;
  if (filter === "no-suggestions") return row.suggestion_count === 0;
  if (filter === "low-confidence") return row.maximum_suggestion_confidence !== null && row.maximum_suggestion_confidence < 0.5;
  return true;
}
function moveQueue(direction) {
  if (state.dirty && !confirm("Discard unsaved changes on this frame?")) return;
  const rows = state.session.frames, total = rows.length;
  for (let step=1; step<=total; step++) {
    const candidate = (state.index + direction*step + total) % total;
    if (queueMatch(rows[candidate])) { loadFrame(candidate); return; }
  }
  setStatus("No frame matches the selected queue.", true);
}
async function request(url, options={}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
  return body;
}
async function loadSession() {
  state.session = await request("/api/session");
  const c = state.session.counts;
  $("progress").textContent = `${c.reviewed_frames}/${c.total_frames} reviewed · ${c.positive_frames} positive · ${c.negative_frames} negative · ${c.annotation_objects} balls`;
}
async function loadFrame(index) {
  try {
    setStatus("Loading frame…");
    const frame = await request(`/api/frames/${index}`);
    const image = new Image();
    image.onload = () => { state.image = image; fitImage(); render(); setStatus("Ready."); };
    image.onerror = () => setStatus("Unable to load source image.", true);
    image.src = `${frame.image_url}?v=${Date.now()}`;
    state.index = index; state.frame = frame;
    state.boxes = (frame.annotation.objects || []).map(humanBox);
    state.selected = state.boxes.length ? 0 : -1; setDirty(false);
    $("frame-index").value = String(index+1);
    $("metadata").innerHTML = `Split: <b>${frame.split}</b><br>Source frame: ${frame.frame_number}<br>Time: ${frame.timestamp_s.toFixed(3)}s<br>Clip: ${frame.clip_id || "—"}<br>Group: ${frame.group_id || "—"}<br>Suggestions: ${frame.suggestions.length}<br>Status: <b>${frame.annotation.review_status}</b>`;
    refreshEditor();
  } catch (error) { setStatus(error.message, true); }
}
function fitImage() {
  if (!state.frame) return;
  const availableWidth = Math.max(200, stage.clientWidth - 32), availableHeight = Math.max(200, stage.clientHeight - 32);
  state.scale = Math.min(1, availableWidth/state.frame.width, availableHeight/state.frame.height);
  $("zoom").value = String(Math.round(state.scale*100));
  $("zoom-label").textContent = `${Math.round(state.scale*100)}%`;
  render();
}
async function save(status, clear=false) {
  try {
    if (clear) { state.boxes = []; state.selected = -1; refreshEditor(); }
    if (status === "reviewed" && state.boxes.some(box => box.visibility === "ambiguous")) {
      throw new Error("Ambiguous boxes must remain a draft or be resolved before review.");
    }
    const reviewer = $("reviewer").value.trim() || "local-reviewer";
    localStorage.setItem("pickleball-reviewer", reviewer);
    await request(`/api/frames/${state.index}`, {
      method: "PUT", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({review_status: status, reviewer, objects: state.boxes})
    });
    setDirty(false); await loadSession();
    state.frame.annotation.review_status = status;
    state.session.frames[state.index].review_status = status;
    state.session.frames[state.index].object_count = state.boxes.length;
    setStatus(status === "reviewed" ? "Frame saved as reviewed." : "Draft saved.");
    render();
  } catch (error) { setStatus(error.message, true); }
}

canvas.addEventListener("pointerdown", event => {
  const point = imagePoint(event), padding = 8/state.scale;
  const humanIndex = state.boxes.findIndex(box => contains(box, point, padding));
  if (humanIndex >= 0) { selectBox(humanIndex); return; }
  const suggestion = state.frame.suggestions.find(box => contains(box, point, padding));
  if (suggestion) { acceptSuggestion(suggestion); return; }
  state.dragStart = point; state.dragNow = point; canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", event => {
  if (!state.dragStart) return; state.dragNow = imagePoint(event); render();
});
canvas.addEventListener("pointerup", event => {
  if (!state.dragStart) return;
  const end = imagePoint(event), box = normalizedBox(state.dragStart, end);
  state.dragStart = null; state.dragNow = null;
  if (box.right_px-box.left_px >= 2 && box.bottom_px-box.top_px >= 2) addBox({bounding_box: box});
  else render();
});
canvas.addEventListener("contextmenu", event => {
  event.preventDefault(); const point = imagePoint(event), padding = 8/state.scale;
  const index = state.boxes.findIndex(box => contains(box, point, padding));
  if (index >= 0) { state.selected = index; removeSelected(); }
});

$("zoom").oninput = event => { state.scale = Number(event.target.value)/100; $("zoom-label").textContent = `${event.target.value}%`; render(); };
$("fit").onclick = fitImage;
$("previous").onclick = () => moveQueue(-1); $("next").onclick = () => moveQueue(1);
$("filter").onchange = () => { if (!queueMatch(state.session.frames[state.index])) moveQueue(1); };
$("go-index").onclick = () => { const value = Number($("frame-index").value)-1; if (Number.isInteger(value) && value >= 0 && value < state.session.frames.length) loadFrame(value); else setStatus("Queue index is out of range.", true); };
$("accept-all").onclick = () => { state.frame.suggestions.forEach(acceptSuggestion); };
$("clear-boxes").onclick = () => { state.boxes=[]; selectBox(-1); setDirty(); };
$("delete-box").onclick = removeSelected;
for (const [id, key] of [["court-side","court_side"],["scope","scope"],["visibility","visibility"]]) {
  $(id).onchange = event => { if (state.selected >= 0) { state.boxes[state.selected][key]=event.target.value; setDirty(); render(); } };
}
$("confidence").oninput = event => { if (state.selected >= 0) { const value=Number(event.target.value); state.boxes[state.selected].annotation_confidence=value; $("confidence-label").textContent=value.toFixed(2); setDirty(); } };
$("save-reviewed").onclick = () => save("reviewed");
$("mark-negative").onclick = () => save("reviewed", true);
$("save-draft").onclick = () => save("unreviewed");
$("unreview").onclick = () => save("unreviewed", true);
$("stop-server").onclick = async () => { if (state.dirty && !confirm("Stop and discard unsaved changes?")) return; await request("/api/shutdown", {method:"POST"}); document.body.innerHTML="<main style='padding:40px'><h1>Review server stopped.</h1><p>You may close this tab.</p></main>"; };
document.addEventListener("keydown", event => {
  if (["INPUT","SELECT"].includes(event.target.tagName)) return;
  if (event.key === "ArrowRight") moveQueue(1);
  else if (event.key === "ArrowLeft") moveQueue(-1);
  else if (event.key.toLowerCase() === "r") save("reviewed");
  else if (event.key.toLowerCase() === "n") save("reviewed", true);
  else if (event.key.toLowerCase() === "s") save("unreviewed");
  else if (event.key.toLowerCase() === "u") save("unreviewed", true);
  else if ((event.key === "Delete" || event.key === "Backspace") && state.selected >= 0) removeSelected();
});
window.addEventListener("beforeunload", event => { if (state.dirty) { event.preventDefault(); event.returnValue=""; } });
(async () => {
  $("reviewer").value = localStorage.getItem("pickleball-reviewer") || "local-reviewer";
  try { await loadSession(); const first = state.session.frames.find(row => row.review_status !== "reviewed"); await loadFrame(first ? first.index : 0); }
  catch (error) { setStatus(error.message, true); }
})();
</script>
</body>
</html>
"""


class _ReviewRequestHandler(BaseHTTPRequestHandler):
    """Small loopback-only API; not a product backend."""

    server_version = "PickleballVisionReview/1"

    def __init__(self, *args: Any, store: BallAnnotationReviewStore, **kwargs: Any) -> None:
        self.store = store
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _send_bytes(self, payload: bytes, *, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: object, *, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(payload, allow_nan=False).encode("utf-8"),
            content_type="application/json; charset=utf-8",
            status=status,
        )

    def _frame_index(self, path: str, *, image: bool = False) -> int | None:
        parts = path.strip("/").split("/")
        expected = 4 if image else 3
        if len(parts) != expected or parts[:2] != ["api", "frames"]:
            return None
        if image and parts[3] != "image":
            return None
        try:
            return int(parts[2])
        except ValueError:
            return None

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._send_bytes(
                    REVIEW_UI_HTML.encode("utf-8"), content_type="text/html; charset=utf-8"
                )
                return
            if path == "/api/session":
                self._send_json(self.store.session_payload())
                return
            image_index = self._frame_index(path, image=True)
            if image_index is not None:
                frame = self.store.frame(image_index)
                content_type = mimetypes.guess_type(frame.image_path.name)[0] or "image/jpeg"
                self._send_bytes(frame.image_path.read_bytes(), content_type=content_type)
                return
            frame_index = self._frame_index(path)
            if frame_index is not None:
                self._send_json(self.store.frame_payload(frame_index))
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except (DatasetInputError, DatasetIoError, OSError) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

    def do_PUT(self) -> None:
        path = urlparse(self.path).path
        frame_index = self._frame_index(path)
        if frame_index is None:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 1 or content_length > MAXIMUM_REQUEST_BYTES:
                raise DatasetInputError("review update body has an invalid size")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            self._send_json(self.store.update_frame(frame_index, payload))
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            DatasetInputError,
            DatasetIoError,
        ) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/shutdown":
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json({"status": "stopping"})
        threading.Thread(target=self.server.shutdown, daemon=True).start()


def serve_ball_annotation_review(
    split_manifest_path: Path,
    *,
    annotations_path: Path,
    dataset_version: str | None = None,
    prediction_paths: Sequence[Path] = (),
    port: int = 8765,
    open_browser: bool = True,
    on_started: Callable[[str], None] | None = None,
) -> BallReviewArtifacts:
    """Serve a resumable local review UI until stopped in-browser or with Ctrl-C."""

    if not 0 <= port <= 65535:
        raise DatasetInputError("review server port must be between 0 and 65535")
    store = BallAnnotationReviewStore(
        split_manifest_path,
        annotations_path=annotations_path,
        dataset_version=dataset_version,
        prediction_paths=prediction_paths,
    )

    def handler(*args: Any, **kwargs: Any) -> _ReviewRequestHandler:
        return _ReviewRequestHandler(*args, store=store, **kwargs)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as error:
        raise DatasetIoError(
            "127.0.0.1", reason=f"unable to start annotation review server: {error}"
        ) from error
    actual_port = cast(tuple[str, int], server.server_address)[1]
    url = f"http://127.0.0.1:{actual_port}/"
    if on_started is not None:
        on_started(url)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    counts = _object(store.session_payload()["counts"], "counts")
    return BallReviewArtifacts(
        url=url,
        annotations_path=store.annotations_path,
        summary_path=store.summary_path,
        reviewed_frames=_integer(counts["reviewed_frames"], "reviewed_frames"),
        total_frames=_integer(counts["total_frames"], "total_frames"),
    )
