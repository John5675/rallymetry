import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from pickleball_vision.errors import ShotModelInputError
from pickleball_vision.shot_dataset import build_shot_training_dataset


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _dataset() -> dict[str, object]:
    def video(video_id: str, shot_type: str, player: str) -> dict[str, object]:
        return {
            "recordType": "ai_adjudicated_multievent_video",
            "reviewer": {"type": "AI_ADJUDICATED", "humanReview": False},
            "source": {"path": f"/{video_id}.mp4"},
            "contacts": [
                {
                    "contactId": f"{video_id}-contact-1",
                    "rallyId": f"{video_id}-rally-1",
                    "shotIndex": 1,
                    "shotType": shot_type,
                    "playerId": player,
                    "logicalRole": "PARTNER" if player == "DENNY" else "OPPONENT_2",
                }
            ],
            "playerObservations": [
                {
                    "observationId": f"{video_id}-observation-1",
                    "playerId": player,
                    "logicalRole": "PARTNER" if player == "DENNY" else "OPPONENT_2",
                }
            ],
        }

    return {
        "recordType": "ai_adjudicated_multievent_dataset",
        "schemaVersion": 1,
        "provenance": {"humanAcceptance": False},
        "splitPolicy": {
            "unit": "whole_video",
            "fixedBeforeReview": True,
            "train": ["vid1"],
            "validation": ["vid7"],
            "test": ["vid8"],
        },
        "videos": {
            "vid1": video("vid1", "DINK", "JOHN"),
            "vid7": video("vid7", "SERVE", "JOHN"),
            "vid8": video("vid8", "SERVE", "DENNY"),
        },
    }


def test_correction_layer_preserves_source_and_blocks_invalid_training(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source = _dataset()
    _write_json(source_path, source)
    source_bytes = source_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    corrections_path = tmp_path / "corrections.json"
    _write_json(
        corrections_path,
        {
            "recordType": "shot_dataset_corrections",
            "schemaVersion": 1,
            "sourceDatasetSha256": source_sha,
            "corrections": [
                {
                    "correctionId": "user-vid8-diana-1",
                    "videoId": "vid8",
                    "targetKind": "CONTACT",
                    "targetId": "vid8-contact-1",
                    "expected": {"playerId": "DENNY", "logicalRole": "PARTNER"},
                    "replacement": {"playerId": "DIANA", "logicalRole": "OPPONENT_2"},
                    "reviewer": {"type": "USER_CONFIRMED"},
                    "evidence": {"note": "Sleeveless, hatted player is Diana"},
                }
            ],
        },
    )

    artifacts = build_shot_training_dataset(
        source_path,
        corrections_path=corrections_path,
        output_dir=tmp_path / "output",
        minimum_train_examples_per_class=1,
        minimum_held_out_examples_per_class=1,
    )

    assert source_path.read_bytes() == source_bytes
    corrected = json.loads(artifacts.dataset_path.read_text(encoding="utf-8"))
    contact = corrected["videos"]["vid8"]["contacts"][0]
    assert contact["playerId"] == "DIANA"
    assert contact["logicalRole"] == "OPPONENT_2"
    assert contact["correctionReferences"] == ["user-vid8-diana-1"]
    assert contact["shotSemantics"]["phase"] == "SERVE"
    assert contact["shotSemantics"]["humanAccepted"] is False
    assert artifacts.semantic_training_allowed is False
    audit = json.loads(artifacts.audit_path.read_text(encoding="utf-8"))
    blocker_codes = {item["code"] for item in audit["blockers"]}
    assert "VALIDATION_NOT_HUMAN_ACCEPTED" in blocker_codes
    assert "TEST_NOT_HUMAN_ACCEPTED" in blocker_codes
    assert "NO_HUMAN_ACCEPTED_TRAINING_LABELS" in blocker_codes


def test_correction_expected_values_prevent_silent_identity_overwrite(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    _write_json(source_path, _dataset())
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    corrections_path = tmp_path / "corrections.json"
    _write_json(
        corrections_path,
        {
            "recordType": "shot_dataset_corrections",
            "schemaVersion": 1,
            "sourceDatasetSha256": source_sha,
            "corrections": [
                {
                    "correctionId": "wrong-precondition",
                    "videoId": "vid8",
                    "targetKind": "CONTACT",
                    "targetId": "vid8-contact-1",
                    "expected": {"playerId": "OKSANA"},
                    "replacement": {"playerId": "DIANA"},
                }
            ],
        },
    )

    with pytest.raises(ShotModelInputError, match="expected playerId"):
        build_shot_training_dataset(
            source_path,
            corrections_path=corrections_path,
            output_dir=tmp_path / "output",
        )


def test_dataset_rejects_video_leakage_between_splits(tmp_path: Path) -> None:
    source = _dataset()
    split_policy = cast(dict[str, object], source["splitPolicy"])
    split_policy["test"] = ["vid7", "vid8"]
    source_path = tmp_path / "source.json"
    _write_json(source_path, source)

    with pytest.raises(ShotModelInputError, match="multiple dataset splits"):
        build_shot_training_dataset(
            source_path,
            corrections_path=None,
            output_dir=tmp_path / "output",
        )
