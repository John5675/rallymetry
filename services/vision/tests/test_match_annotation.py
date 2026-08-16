import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from pickleball_vision.errors import MatchAnnotationInputError
from pickleball_vision.match_annotation import (
    MATCH_ANNOTATION_RECORD_TYPE,
    MATCH_ANNOTATION_VERSION,
    MatchAnnotationStore,
)
from pickleball_vision.match_annotation_ui import _range_bounds
from pickleball_vision.media import MediaTimeline


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "RALLY_START",
        "frame": 2,
        "playerId": None,
        "team": None,
        "shotType": None,
        "courtPosition": None,
        "audioLabel": None,
        "notes": None,
        "annotationConfidence": 1.0,
        "annotator": "human-reviewer",
    }
    payload.update(updates)
    return payload


def _audio_context(path: Path, *, video: Path, waveform: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "recordType": "audio_analysis_observations",
                "audioAnalysisAvailable": True,
                "sourceMedia": {"path": str(video.resolve())},
                "configuration": {"audioVideoOffsetMs": 25.0},
                "artifacts": {"waveform": str(waveform.resolve())},
                "audioEventCandidates": [
                    {
                        "id": "audio-transient-0000001",
                        "mediaTimestampSeconds": 0.35,
                        "confidence": 0.8,
                        "source": "AUDIO",
                        "candidateType": "TRANSIENT",
                        "semanticClassification": None,
                    },
                    {
                        "id": "audio-transient-0000002",
                        "mediaTimestampSeconds": 0.9,
                        "confidence": 0.6,
                        "source": "AUDIO",
                        "candidateType": "TRANSIENT",
                        "semanticClassification": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_annotation_document_starts_empty_with_video_provenance(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    source_hash = _sha256(synthetic_video)
    output = tmp_path / "annotations.json"

    store = MatchAnnotationStore(synthetic_video, output_path=output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["annotationVersion"] == MATCH_ANNOTATION_VERSION
    assert saved["recordType"] == MATCH_ANNOTATION_RECORD_TYPE
    assert saved["video"]["contentSha256"] == source_hash
    assert saved["video"]["fps"] == pytest.approx(7.5)
    assert saved["video"]["frame_count"] == 12
    assert saved["events"] == []
    assert saved["audioContext"]["requested"] is False
    assert saved["contracts"]["automaticEventInference"] is False
    assert saved["contracts"]["audioTransientsAreSemanticEvents"] is False
    assert _sha256(synthetic_video) == source_hash
    assert store.session_payload()["events"] == []


def test_add_edit_delete_serializes_all_optional_metadata(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "annotations.json"
    store = MatchAnnotationStore(
        synthetic_video,
        output_path=output,
        timeline=MediaTimeline(audio_video_offset_ms=40.0),
    )

    added = store.add_event(
        _event_payload(
            type="PADDLE_CONTACT",
            frame=3,
            playerId="ME",
            team="MY_TEAM",
            shotType="DRIVE",
            courtPosition={
                "xMeters": 2.4,
                "yMeters": 5.7,
                "coordinateSystem": "canonical_pickleball_court",
                "source": "HUMAN_ANNOTATION",
            },
            audioLabel="PRIMARY_EVENT_AUDIBLE",
            notes="Clear visual contact; audio supports but does not define it.",
            annotationConfidence=0.94,
        )
    )

    assert added["id"] == "match-event-0000001"
    assert added["frame"] == 3
    assert added["videoTimestampSeconds"] == pytest.approx(0.4)
    assert added["mediaTimestampSeconds"] == pytest.approx(0.4)
    position = cast(dict[str, object], added["courtPosition"])
    assert position["source"] == "HUMAN_ANNOTATION"
    assert added["source"] == "HUMAN"
    assert added["audioLabel"] == "PRIMARY_EVENT_AUDIBLE"
    assert store.document_payload()["events"] == [added]

    updated = store.update_event(
        added["id"],
        {
            "type": "BOUNCE",
            "frame": 5,
            "playerId": None,
            "audioLabel": "AMBIGUOUS_AUDIO",
            "notes": "Corrected after frame stepping.",
        },
    )
    assert updated["id"] == added["id"]
    assert updated["createdAtUtc"] == added["createdAtUtc"]
    assert updated["type"] == "BOUNCE"
    assert updated["frame"] == 5
    assert updated["videoTimestampSeconds"] == pytest.approx(5 / 7.5)
    assert updated["playerId"] is None
    assert updated["shotType"] == "DRIVE"

    deleted = store.delete_event(added["id"])
    assert deleted["id"] == added["id"]
    assert store.document_payload()["events"] == []
    assert json.loads(output.read_text(encoding="utf-8"))["events"] == []


def test_events_sort_by_frame_keep_stable_ids_and_resume(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "annotations.json"
    store = MatchAnnotationStore(synthetic_video, output_path=output)
    later = store.add_event(_event_payload(type="RALLY_END", frame=9))
    earlier = store.add_event(_event_payload(type="SERVE_CONTACT", frame=1))

    events = cast(list[dict[str, object]], store.document_payload()["events"])
    assert [event["frame"] for event in events] == [1, 9]
    assert earlier["id"] == "match-event-0000002"
    assert later["id"] == "match-event-0000001"

    resumed = MatchAnnotationStore(synthetic_video, output_path=output)
    resumed_events = cast(list[dict[str, object]], resumed.document_payload()["events"])
    assert resumed_events == events
    third = resumed.add_event(_event_payload(type="RALLY_WINNER", frame=10, team="MY_TEAM"))
    assert third["id"] == "match-event-0000003"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"type": "AUTOMATIC_BOUNCE"}, "type is unsupported"),
        ({"frame": -1}, "frame must be in"),
        ({"frame": 12}, "frame must be in"),
        ({"annotationConfidence": 1.2}, r"must be in \[0, 1\]"),
        ({"audioLabel": "LOUD"}, "audioLabel is unsupported"),
        ({"courtPosition": {"xMeters": 1.0}}, "yMeters must be numeric"),
        ({"inferred": True}, "unsupported fields"),
    ],
)
def test_invalid_event_edits_are_rejected_without_changing_file(
    synthetic_video: Path,
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    output = tmp_path / "annotations.json"
    store = MatchAnnotationStore(synthetic_video, output_path=output)
    before = output.read_bytes()

    with pytest.raises(MatchAnnotationInputError, match=message):
        store.add_event(_event_payload(**updates))

    assert output.read_bytes() == before


def test_reopen_rejects_tampered_frame_timestamp(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "annotations.json"
    store = MatchAnnotationStore(synthetic_video, output_path=output)
    store.add_event(_event_payload(frame=4))
    root = json.loads(output.read_text(encoding="utf-8"))
    root["events"][0]["mediaTimestampSeconds"] = 999.0
    output.write_text(json.dumps(root), encoding="utf-8")

    with pytest.raises(MatchAnnotationInputError, match="inconsistent with frame"):
        MatchAnnotationStore(synthetic_video, output_path=output)


def test_optional_audio_context_stays_separate_and_resumes(
    synthetic_media_with_audio: Path,
    tmp_path: Path,
) -> None:
    waveform = tmp_path / "waveform.png"
    waveform.write_bytes(b"synthetic waveform artifact")
    audio_events = tmp_path / "audio-events.json"
    _audio_context(audio_events, video=synthetic_media_with_audio, waveform=waveform)
    output = tmp_path / "annotations.json"

    store = MatchAnnotationStore(
        synthetic_media_with_audio,
        output_path=output,
        audio_events_path=audio_events,
    )

    session = store.session_payload()
    context = cast(dict[str, object], session["audioContext"])
    markers = cast(list[dict[str, object]], context["transientMarkers"])
    assert context["audioAnalysisAvailable"] is True
    assert context["transientCandidateCount"] == 2
    assert context["waveformUrl"] == "/media/waveform"
    assert [marker["semanticEvent"] for marker in markers] == [False, False]
    document = store.document_payload()
    assert document["events"] == []
    document_audio = cast(dict[str, object], document["audioContext"])
    assert document_audio["candidatesImportedAsGroundTruth"] is False

    resumed = MatchAnnotationStore(synthetic_media_with_audio, output_path=output)
    resumed_context = cast(dict[str, object], resumed.session_payload()["audioContext"])
    assert resumed_context["transientCandidateCount"] == 2


def test_byte_ranges_support_video_seeking() -> None:
    assert _range_bounds("bytes=0-99", file_size=1_000) == (0, 99)
    assert _range_bounds("bytes=900-", file_size=1_000) == (900, 999)
    assert _range_bounds("bytes=-100", file_size=1_000) == (900, 999)
    assert _range_bounds("bytes=950-1200", file_size=1_000) == (950, 999)
    with pytest.raises(ValueError, match="outside"):
        _range_bounds("bytes=1000-1001", file_size=1_000)
