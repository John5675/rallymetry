from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import pytest

from pickleball_vision.errors import ShotModelInputError
from pickleball_vision.shot_review_overlay import apply_ai_shot_review_overlay


def _write_index(path: Path, *, source_sha256: str, human_accepted: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "recordType": "ai_reviewed_shot_label_index",
                "reviewVersion": "test-review-v1",
                "provenance": {
                    "reviewerType": "AI",
                    "humanAccepted": human_accepted,
                    "groundTruth": False,
                },
                "videos": [
                    {
                        "videoId": "vid-test",
                        "sourceSha256": source_sha256,
                        "fps": 30.0,
                        "frameCount": 300,
                        "reviews": [
                            {
                                "sourceContactId": "review-contact-1",
                                "sourceFrame": 31,
                                "aiReview": {
                                    "timestampSeconds": 1.033,
                                    "legacyBestGuess": "DINK",
                                    "legacyBestGuessConfidence": 0.72,
                                    "semantics": {
                                        "phase": {
                                            "authoritative": "UNKNOWN",
                                            "bestGuess": "RALLY",
                                            "confidence": 0.72,
                                            "abstained": True,
                                        }
                                    },
                                    "provenance": "AI_VISUAL_REVIEW",
                                    "humanAccepted": False,
                                    "trainingEligibility": "PSEUDO_LABEL_ONLY",
                                },
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_shots(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "recordType": "reconstructed_shots",
                "shots": [
                    {
                        "shotId": "shot-1",
                        "contactTimestamp": 1.0,
                        "shotType": "UNKNOWN",
                        "hitterId": "ME",
                        "confidence": 0.2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_exact_source_overlay_preserves_machine_prediction(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"reviewed-source")
    index = tmp_path / "review.json"
    _write_index(index, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    shots = tmp_path / "shots.json"
    _write_shots(shots)
    output = tmp_path / "reviewed-shots.json"

    artifacts = apply_ai_shot_review_overlay(
        source,
        shots_path=shots,
        output_path=output,
        review_index_path=index,
        maximum_timing_delta_ms=100.0,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    reviewed = payload["shots"][0]
    assert artifacts.matched_source is True
    assert artifacts.matched_shot_count == 1
    assert reviewed["shotType"] == "UNKNOWN"
    assert reviewed["hitterId"] == "ME"
    assert reviewed["aiVisualReview"]["review"]["legacyBestGuess"] == "DINK"
    assert reviewed["aiVisualReview"]["predictionPreserved"] is True
    assert payload["aiReviewOverlay"]["humanAccepted"] is False
    assert payload["aiReviewOverlay"]["machinePredictionsMutated"] is False


def test_unreviewed_source_is_copied_without_record_overlays(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"different-source")
    index = tmp_path / "review.json"
    _write_index(index, source_sha256=hashlib.sha256(b"reviewed-source").hexdigest())
    shots = tmp_path / "shots.json"
    _write_shots(shots)
    output = tmp_path / "reviewed-shots.json"

    artifacts = apply_ai_shot_review_overlay(
        source,
        shots_path=shots,
        output_path=output,
        review_index_path=index,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert artifacts.matched_source is False
    assert artifacts.matched_shot_count == 0
    assert "aiVisualReview" not in payload["shots"][0]
    assert payload["aiReviewOverlay"]["matchedSource"] is False


def test_review_index_cannot_claim_human_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    index = tmp_path / "review.json"
    _write_index(
        index,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        human_accepted=True,
    )
    shots = tmp_path / "shots.json"
    _write_shots(shots)

    with pytest.raises(ShotModelInputError, match="humanAccepted=false"):
        apply_ai_shot_review_overlay(
            source,
            shots_path=shots,
            output_path=tmp_path / "output.json",
            review_index_path=index,
        )


def test_bundled_eight_video_review_index_is_packaged() -> None:
    payload = json.loads(
        files("pickleball_vision")
        .joinpath("resources/eight_video_ai_review_v1.json")
        .read_text("utf-8")
    )

    assert payload["recordType"] == "ai_reviewed_shot_label_index"
    assert payload["provenance"]["humanAccepted"] is False
    assert payload["provenance"]["groundTruth"] is False
    assert len(payload["videos"]) == 8
    assert sum(len(video["reviews"]) for video in payload["videos"]) == 55
