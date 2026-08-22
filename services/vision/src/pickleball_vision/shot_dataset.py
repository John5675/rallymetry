"""Leakage-safe shot-label audit and correction-layer materialization."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pickleball_vision.errors import ShotModelInputError
from pickleball_vision.shot_taxonomy import ShotLabelProvenance, semantics_from_legacy

SPLITS = ("train", "validation", "test")
AXES = ("phase", "contactMode", "strokeSide", "intent")


@dataclass(frozen=True, slots=True)
class ShotDatasetArtifacts:
    """Paths and core gate outcome for one corrected dataset audit."""

    dataset_path: Path
    audit_path: Path
    semantic_training_allowed: bool
    contact_count: int
    correction_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "datasetPath": str(self.dataset_path),
            "auditPath": str(self.audit_path),
            "semanticTrainingAllowed": self.semantic_training_allowed,
            "contactCount": self.contact_count,
            "correctionCount": self.correction_count,
        }


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShotModelInputError(f"unable to read {label} {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ShotModelInputError(f"{label} root must be an object")
    return cast(dict[str, object], raw)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.expanduser().resolve().open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise ShotModelInputError(f"unable to hash {path}: {error}") from error
    return digest.hexdigest()


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ShotModelInputError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ShotModelInputError(f"{field} must be an array")
    return cast(list[object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ShotModelInputError(f"{field} must be a non-empty string")
    return value.strip()


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ShotModelInputError(f"{field} must be a boolean")
    return value


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise ShotModelInputError(f"unable to write {path}: {error}") from error


def _split_members(split_policy: dict[str, object]) -> dict[str, tuple[str, ...]]:
    unit = _string(split_policy.get("unit"), "splitPolicy.unit")
    if unit != "whole_video":
        raise ShotModelInputError("splitPolicy.unit must be whole_video")
    result: dict[str, tuple[str, ...]] = {}
    all_ids: list[str] = []
    for split in SPLITS:
        raw = _list(split_policy.get(split), f"splitPolicy.{split}")
        members = tuple(_string(item, f"splitPolicy.{split}[]") for item in raw)
        if len(set(members)) != len(members):
            raise ShotModelInputError(f"splitPolicy.{split} contains duplicate videos")
        result[split] = members
        all_ids.extend(members)
    if len(set(all_ids)) != len(all_ids):
        raise ShotModelInputError("one or more videos appear in multiple dataset splits")
    return result


def _find_record(
    video: dict[str, object],
    *,
    target_kind: str,
    target_id: str,
    video_id: str,
) -> dict[str, object]:
    collections = {
        "CONTACT": ("contacts", "contactId"),
        "PLAYER_OBSERVATION": ("playerObservations", "observationId"),
    }
    if target_kind not in collections:
        raise ShotModelInputError(f"unsupported correction targetKind {target_kind}")
    collection_name, id_field = collections[target_kind]
    records = _list(video.get(collection_name), f"videos.{video_id}.{collection_name}")
    matching = [
        _object(record, f"videos.{video_id}.{collection_name}[]")
        for record in records
        if isinstance(record, dict) and record.get(id_field) == target_id
    ]
    if len(matching) != 1:
        raise ShotModelInputError(
            f"correction target {video_id}/{target_kind}/{target_id} matched "
            f"{len(matching)} records"
        )
    return matching[0]


def _apply_corrections(
    dataset: dict[str, object],
    corrections: dict[str, object] | None,
    *,
    source_sha256: str,
) -> list[dict[str, object]]:
    if corrections is None:
        return []
    if corrections.get("recordType") != "shot_dataset_corrections":
        raise ShotModelInputError("corrections.recordType must be shot_dataset_corrections")
    if corrections.get("schemaVersion") != 1:
        raise ShotModelInputError("corrections.schemaVersion must be 1")
    expected_sha = _string(corrections.get("sourceDatasetSha256"), "sourceDatasetSha256")
    if expected_sha != source_sha256:
        raise ShotModelInputError(
            "corrections sourceDatasetSha256 does not match the source dataset"
        )
    videos = _object(dataset.get("videos"), "videos")
    applied: list[dict[str, object]] = []
    seen_targets: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(_list(corrections.get("corrections"), "corrections")):
        field = f"corrections[{index}]"
        correction = _object(raw, field)
        correction_id = _string(correction.get("correctionId"), f"{field}.correctionId")
        video_id = _string(correction.get("videoId"), f"{field}.videoId")
        target_kind = _string(correction.get("targetKind"), f"{field}.targetKind")
        target_id = _string(correction.get("targetId"), f"{field}.targetId")
        target_key = (video_id, target_kind, target_id)
        if target_key in seen_targets:
            raise ShotModelInputError(f"duplicate correction target {target_key}")
        seen_targets.add(target_key)
        video = _object(videos.get(video_id), f"videos.{video_id}")
        record = _find_record(
            video,
            target_kind=target_kind,
            target_id=target_id,
            video_id=video_id,
        )
        expected = _object(correction.get("expected"), f"{field}.expected")
        replacement = _object(correction.get("replacement"), f"{field}.replacement")
        if not expected or not replacement:
            raise ShotModelInputError(f"{field} expected and replacement must not be empty")
        for key, value in expected.items():
            if record.get(key) != value:
                raise ShotModelInputError(
                    f"{field} expected {key}={value!r}, found {record.get(key)!r}"
                )
        before = {key: record.get(key) for key in replacement}
        for key, value in replacement.items():
            record[key] = deepcopy(value)
        record.setdefault("correctionReferences", [])
        references = _list(record["correctionReferences"], f"{field}.correctionReferences")
        references.append(correction_id)
        applied.append(
            {
                "correctionId": correction_id,
                "videoId": video_id,
                "targetKind": target_kind,
                "targetId": target_id,
                "before": before,
                "after": deepcopy(replacement),
                "reviewer": deepcopy(correction.get("reviewer")),
                "evidence": deepcopy(correction.get("evidence")),
            }
        )
    return applied


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def build_shot_training_dataset(
    source_dataset_path: Path,
    *,
    corrections_path: Path | None,
    output_dir: Path,
    minimum_train_examples_per_class: int = 10,
    minimum_held_out_examples_per_class: int = 5,
) -> ShotDatasetArtifacts:
    """Apply corrections to a copy, derive axes, and report honest training gates."""

    if minimum_train_examples_per_class < 1 or minimum_held_out_examples_per_class < 1:
        raise ShotModelInputError("minimum class-support thresholds must be positive")
    resolved_source = source_dataset_path.expanduser().resolve()
    source = _read_json(resolved_source, label="source dataset")
    if source.get("recordType") != "ai_adjudicated_multievent_dataset":
        raise ShotModelInputError(
            "source dataset recordType must be ai_adjudicated_multievent_dataset"
        )
    if source.get("schemaVersion") != 1:
        raise ShotModelInputError("source dataset schemaVersion must be 1")
    source_sha = _sha256(resolved_source)
    dataset = deepcopy(source)
    correction_payload = (
        _read_json(corrections_path.expanduser().resolve(), label="corrections")
        if corrections_path is not None
        else None
    )
    applied = _apply_corrections(dataset, correction_payload, source_sha256=source_sha)
    split_policy = _object(dataset.get("splitPolicy"), "splitPolicy")
    split_members = _split_members(split_policy)
    videos = _object(dataset.get("videos"), "videos")
    declared_ids = {video_id for members in split_members.values() for video_id in members}
    if declared_ids != set(videos):
        raise ShotModelInputError("splitPolicy videos must exactly match dataset videos")

    counts: dict[str, dict[str, Counter[str]]] = {
        split: {axis: Counter() for axis in AXES} for split in SPLITS
    }
    legacy_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    human_counts: dict[str, int] = {split: 0 for split in SPLITS}
    contact_counts: dict[str, int] = {split: 0 for split in SPLITS}
    for split, members in split_members.items():
        for video_id in members:
            video = _object(videos.get(video_id), f"videos.{video_id}")
            reviewer = _object(video.get("reviewer"), f"videos.{video_id}.reviewer")
            human_review = reviewer.get("humanReview") is True
            contacts = _list(video.get("contacts"), f"videos.{video_id}.contacts")
            for contact_index, raw_contact in enumerate(contacts):
                field = f"videos.{video_id}.contacts[{contact_index}]"
                contact = _object(raw_contact, field)
                legacy_type = _string(contact.get("shotType", "UNKNOWN"), f"{field}.shotType")
                rally_anchored = contact.get("rallyId") is not None
                semantics = semantics_from_legacy(
                    legacy_type,
                    rally_anchored=rally_anchored,
                    provenance=(
                        ShotLabelProvenance.HUMAN_ACCEPTED
                        if human_review
                        else ShotLabelProvenance.AI_PSEUDO_LABEL
                    ),
                    human_accepted=human_review,
                )
                contact["shotSemantics"] = semantics.as_dict()
                contact_counts[split] += 1
                legacy_counts[split][legacy_type.upper()] += 1
                semantic_dict = semantics.as_dict()
                for axis in AXES:
                    counts[split][axis][_string(semantic_dict[axis], f"{field}.{axis}")] += 1
                if human_review:
                    human_counts[split] += 1

    blockers: list[dict[str, object]] = []
    if human_counts["validation"] != contact_counts["validation"]:
        blockers.append(
            {
                "code": "VALIDATION_NOT_HUMAN_ACCEPTED",
                "accepted": human_counts["validation"],
                "total": contact_counts["validation"],
            }
        )
    if human_counts["test"] != contact_counts["test"]:
        blockers.append(
            {
                "code": "TEST_NOT_HUMAN_ACCEPTED",
                "accepted": human_counts["test"],
                "total": contact_counts["test"],
            }
        )
    if human_counts["train"] == 0:
        blockers.append({"code": "NO_HUMAN_ACCEPTED_TRAINING_LABELS"})

    support: dict[str, dict[str, dict[str, int]]] = {}
    for axis in AXES:
        claimed_classes = sorted(label for label in counts["train"][axis] if label != "UNKNOWN")
        support[axis] = {}
        for label in claimed_classes:
            class_support = {split: counts[split][axis][label] for split in SPLITS}
            support[axis][label] = class_support
            if class_support["train"] < minimum_train_examples_per_class:
                blockers.append(
                    {
                        "code": "INSUFFICIENT_TRAIN_CLASS_SUPPORT",
                        "axis": axis,
                        "label": label,
                        "actual": class_support["train"],
                        "required": minimum_train_examples_per_class,
                    }
                )
            for held_out in ("validation", "test"):
                if class_support[held_out] < minimum_held_out_examples_per_class:
                    blockers.append(
                        {
                            "code": "INSUFFICIENT_HELD_OUT_CLASS_SUPPORT",
                            "axis": axis,
                            "label": label,
                            "partition": held_out,
                            "actual": class_support[held_out],
                            "required": minimum_held_out_examples_per_class,
                        }
                    )

    semantic_training_allowed = not blockers
    dataset["recordType"] = "corrected_multiaxis_shot_training_dataset"
    dataset["schemaVersion"] = 2
    dataset["taxonomyVersion"] = 1
    dataset["provenance"] = {
        **_object(dataset.get("provenance"), "provenance"),
        "parentDatasetPath": str(resolved_source),
        "parentDatasetSha256": source_sha,
        "correctionLayerPath": (
            str(corrections_path.expanduser().resolve()) if corrections_path is not None else None
        ),
        "correctionLayerSha256": _sha256(corrections_path) if corrections_path else None,
        "correctionsApplied": applied,
        "aiPseudoLabelsRepresentedAsHuman": False,
    }
    dataset["trainingGate"] = {
        "semanticTrainingAllowed": semantic_training_allowed,
        "blockers": blockers,
        "minimumTrainExamplesPerClass": minimum_train_examples_per_class,
        "minimumHeldOutExamplesPerClass": minimum_held_out_examples_per_class,
    }
    audit = {
        "recordType": "shot_training_dataset_audit",
        "schemaVersion": 1,
        "sourceDatasetSha256": source_sha,
        "correctedDatasetSchemaVersion": 2,
        "splitUnit": "whole_video",
        "splitMembers": {key: list(value) for key, value in split_members.items()},
        "contactCount": sum(contact_counts.values()),
        "contactCountsBySplit": contact_counts,
        "legacyShotTypeCountsBySplit": {
            split: _counter_dict(legacy_counts[split]) for split in SPLITS
        },
        "axisCountsBySplit": {
            split: {axis: _counter_dict(counts[split][axis]) for axis in AXES} for split in SPLITS
        },
        "humanAcceptedSemanticCountsBySplit": human_counts,
        "classSupport": support,
        "correctionCount": len(applied),
        "correctionsApplied": applied,
        "semanticTrainingAllowed": semantic_training_allowed,
        "blockers": blockers,
        "externalRepresentationPretrainingChangesSemanticGroundTruth": False,
    }
    resolved_output = output_dir.expanduser().resolve()
    dataset_path = resolved_output / "shot-training-dataset.json"
    audit_path = resolved_output / "shot-training-audit.json"
    if dataset_path.exists() or audit_path.exists():
        raise ShotModelInputError(f"shot dataset output already exists in {resolved_output}")
    _atomic_write_json(dataset_path, dataset)
    _atomic_write_json(audit_path, audit)
    return ShotDatasetArtifacts(
        dataset_path=dataset_path,
        audit_path=audit_path,
        semantic_training_allowed=semantic_training_allowed,
        contact_count=sum(contact_counts.values()),
        correction_count=len(applied),
    )
