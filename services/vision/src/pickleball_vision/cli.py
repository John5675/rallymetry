"""Command-line interface for Pickleball Vision."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pickleball_vision import __version__
from pickleball_vision.audio_analysis_workflow import analyze_audio_in_video
from pickleball_vision.ball_config import load_ball_experiment_configuration
from pickleball_vision.ball_dataset import create_ball_annotation_template
from pickleball_vision.ball_detection_workflow import detect_balls_in_video
from pickleball_vision.ball_evaluation import (
    compare_ball_inference_strategies,
    evaluate_ball_detector,
)
from pickleball_vision.ball_review import serve_ball_annotation_review
from pickleball_vision.ball_tracking_workflow import track_ball_in_video
from pickleball_vision.ball_training import train_ball_detector
from pickleball_vision.calibration_workflow import calibrate_video
from pickleball_vision.config import Settings
from pickleball_vision.dataset import (
    DatasetLabelGroup,
    DatasetSplit,
    FrameSelectionSettings,
    SplitRatios,
    SplitUnit,
)
from pickleball_vision.dataset_workflow import (
    extract_ball_dataset_frames,
    split_ball_dataset,
)
from pickleball_vision.errors import ErrorCode, PickleballVisionError
from pickleball_vision.logging import configure_logging
from pickleball_vision.media import (
    AudioExtractionOptions,
    MediaTimeline,
    extract_audio,
    inspect_media,
)
from pickleball_vision.person_detection_pipeline import detect_people_in_video
from pickleball_vision.player_analysis_workflow import analyze_players_in_video
from pickleball_vision.player_isolation_workflow import isolate_primary_players
from pickleball_vision.player_tracking_workflow import track_players_in_video
from pickleball_vision.video import extract_frame, sample_frames

EXIT_OK = 0
EXIT_INTERNAL_ERROR = 1
EXIT_USAGE_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without reading configuration or performing I/O."""

    parser = argparse.ArgumentParser(
        prog="pickleball-vision",
        description="Local, inspectable doubles-pickleball vision pipeline",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "doctor",
        help="validate Foundation configuration and report service metadata",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="inspect metadata for a local video file",
    )
    inspect_parser.add_argument("video", type=Path, help="path to a local video file")

    extract_parser = subparsers.add_parser(
        "extract-frame",
        help="extract a source-resolution frame at a timestamp",
    )
    extract_parser.add_argument("video", type=Path, help="path to a local video file")
    extract_parser.add_argument(
        "--timestamp",
        type=float,
        required=True,
        help="timestamp in seconds in the range [0, duration)",
    )
    extract_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output image path (.jpg, .png, .webp, .bmp, or .tiff)",
    )

    extract_audio_parser = subparsers.add_parser(
        "extract-audio",
        help="extract synchronized lossless PCM WAV analysis audio",
    )
    extract_audio_parser.add_argument("video", type=Path, help="path to a local video file")
    extract_audio_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output PCM WAV path; a .metadata.json timing sidecar is also written",
    )
    extract_audio_parser.add_argument(
        "--sample-rate",
        type=int,
        help="optional explicit output sample rate in Hz (source rate is preserved by default)",
    )
    extract_audio_parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 2),
        help="optional explicit mono/stereo conversion (source channels are preserved by default)",
    )

    analyze_audio_parser = subparsers.add_parser(
        "analyze-audio",
        help="extract synchronized raw audio features and generic transient candidates",
    )
    analyze_audio_parser.add_argument("video", type=Path, help="path to a local video file")
    analyze_audio_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for audio JSON, waveform, and transient timeline artifacts",
    )

    sample_parser = subparsers.add_parser(
        "sample-frames",
        help="sample unique frames across a local video's duration",
    )
    sample_parser.add_argument("video", type=Path, help="path to a local video file")
    sample_parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="number of frames to sample",
    )
    sample_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for sampled JPEG images",
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="manually calibrate a video frame to the canonical court plane",
    )
    calibrate_parser.add_argument("video", type=Path, help="path to a local video file")
    calibrate_parser.add_argument(
        "--timestamp",
        type=float,
        required=True,
        help="timestamp of a clear calibration frame in [0, duration)",
    )
    calibrate_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output calibration JSON path",
    )

    detect_people_parser = subparsers.add_parser(
        "detect-people",
        help="detect every visible person without selecting match participants",
    )
    detect_people_parser.add_argument("video", type=Path, help="path to a local video file")
    detect_people_parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="court calibration JSON used as validated run provenance",
    )
    detect_people_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for detections.json, annotated.mp4, and summary.json",
    )

    isolate_parser = subparsers.add_parser(
        "isolate-players",
        help="derive primary-court candidates and manually assign four logical players",
    )
    isolate_parser.add_argument("video", type=Path, help="path to the detected local video")
    isolate_parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="raw detections.json from detect-people",
    )
    isolate_parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="court calibration JSON for bottom-center ground-point projection",
    )
    isolate_parser.add_argument(
        "--timestamp",
        type=float,
        required=True,
        help="initial manual-selection timestamp in seconds",
    )
    isolate_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for candidates, assignments, debug video, and summary",
    )
    isolate_parser.add_argument(
        "--assignments",
        type=Path,
        help="existing player-assignments.json to review or correct",
    )

    track_parser = subparsers.add_parser(
        "track-players",
        help="persist the four manually assigned logical players across the video",
    )
    track_parser.add_argument("video", type=Path, help="path to the detected local video")
    track_parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="court calibration JSON for court-aware identity resolution",
    )
    track_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for tracks.json, annotated.mp4, and tracking-summary.json",
    )
    track_parser.add_argument(
        "--detections",
        type=Path,
        help="raw detections.json; defaults to the path recorded by assignments",
    )
    track_parser.add_argument(
        "--assignments",
        type=Path,
        help=(
            "manual player-assignments.json; defaults to the sibling "
            "player-isolation output directory"
        ),
    )
    track_parser.add_argument(
        "--player-names",
        type=Path,
        help=(
            "optional JSON mapping ME/PARTNER/OPPONENT_1/OPPONENT_2 to display names; "
            "defaults to player-names.json beside assignments when present"
        ),
    )

    analyze_parser = subparsers.add_parser(
        "analyze-players",
        help="derive Release 0.1 player positions, movement metrics, and visualizations",
    )
    analyze_parser.add_argument("video", type=Path, help="path to the tracked local video")
    analyze_parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="court calibration JSON recorded by persistent tracking",
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for Release 0.1 player-position analysis artifacts",
    )
    analyze_parser.add_argument(
        "--tracks",
        type=Path,
        help="tracks.json; defaults to the sibling player-tracking output directory",
    )
    analyze_parser.add_argument(
        "--position-corrections",
        type=Path,
        help=(
            "optional recording-local court-position correction JSON; defaults to "
            "player-position-corrections.json beside tracks.json when present"
        ),
    )

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="extract and split local ball-annotation datasets without training a model",
    )
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")
    dataset_extract_parser = dataset_subparsers.add_parser(
        "extract-frames",
        help="extract source-resolution frames with dataset provenance",
    )
    dataset_extract_parser.add_argument("video", type=Path, help="path to a local video file")
    dataset_extract_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new dataset run directory for images, clips, and dataset-manifest.json",
    )
    sampling_group = dataset_extract_parser.add_mutually_exclusive_group(required=True)
    sampling_group.add_argument(
        "--every",
        dest="every_frames",
        type=int,
        help="extract the first eligible frame and then every N source frames",
    )
    sampling_group.add_argument(
        "--random-count",
        type=int,
        help="extract N unique eligible frames using the configured seed",
    )
    dataset_extract_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random sampling seed (default: 0)",
    )
    dataset_extract_parser.add_argument(
        "--start-time",
        type=float,
        help="inclusive source-video start time in seconds (default: 0)",
    )
    dataset_extract_parser.add_argument(
        "--end-time",
        type=float,
        help="exclusive source-video end time in seconds (default: source duration)",
    )
    dataset_extract_parser.add_argument(
        "--clips",
        type=Path,
        help="named clip/group JSON; cannot be combined with start/end time",
    )
    dataset_extract_parser.add_argument(
        "--write-clips",
        action="store_true",
        help="also write lossless synchronized MKV review clips for selected ranges",
    )
    dataset_extract_parser.add_argument(
        "--label-group",
        choices=tuple(group.value for group in DatasetLabelGroup),
        default=DatasetLabelGroup.UNLABELED.value,
        help="default human-curation bucket for extracted frames",
    )
    dataset_extract_parser.add_argument(
        "--group-id",
        help="optional rally/group ID used as an indivisible split unit",
    )

    dataset_split_parser = dataset_subparsers.add_parser(
        "split",
        help="assign whole videos, clips, or groups to leakage-safe dataset splits",
    )
    dataset_split_parser.add_argument(
        "manifests",
        type=Path,
        nargs="+",
        help="one or more dataset-manifest.json files",
    )
    dataset_split_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output split-assignment JSON path",
    )
    dataset_split_parser.add_argument(
        "--by",
        choices=tuple(unit.value for unit in SplitUnit),
        default=SplitUnit.VIDEO.value,
        help="indivisible split unit (default: video)",
    )
    dataset_split_parser.add_argument("--train", type=float, default=0.70)
    dataset_split_parser.add_argument("--validation", type=float, default=0.15)
    dataset_split_parser.add_argument("--test", type=float, default=0.15)
    dataset_split_parser.add_argument("--seed", type=int, default=0)

    ball_parser = subparsers.add_parser(
        "ball",
        help="prepare, train, infer, and evaluate the custom pickleball detector",
    )
    ball_subparsers = ball_parser.add_subparsers(dest="ball_command")
    annotation_template_parser = ball_subparsers.add_parser(
        "create-annotation-template",
        help="create unreviewed annotation records for a fixed dataset split",
    )
    annotation_template_parser.add_argument("split_manifest", type=Path)
    annotation_template_parser.add_argument("--dataset-version", required=True)
    annotation_template_parser.add_argument("--output", type=Path, required=True)

    review_ball_parser = ball_subparsers.add_parser(
        "review",
        help="launch a local resumable browser UI for human ball annotation review",
    )
    review_ball_parser.add_argument("split_manifest", type=Path)
    review_ball_parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="annotation JSON to create or resume",
    )
    review_ball_parser.add_argument(
        "--dataset-version",
        help="required only when --annotations does not exist; otherwise validates its version",
    )
    review_ball_parser.add_argument(
        "--predictions",
        type=Path,
        action="append",
        default=[],
        help="optional raw detections.json suggestion file; repeat for multiple source videos",
    )
    review_ball_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="loopback review server port; use 0 to select an available port",
    )
    review_ball_parser.add_argument(
        "--no-open",
        action="store_true",
        help="do not open the default browser automatically",
    )

    train_ball_parser = ball_subparsers.add_parser(
        "train",
        help="train one custom model from a versioned experiment configuration",
    )
    train_ball_parser.add_argument("--config", type=Path, required=True)
    train_ball_parser.add_argument("--output-dir", type=Path, required=True)

    detect_ball_parser = ball_subparsers.add_parser(
        "detect",
        help="write raw frame-local pickleball detections without tracking",
    )
    detect_ball_parser.add_argument("video", type=Path)
    detect_ball_parser.add_argument("--config", type=Path, required=True)
    detect_ball_parser.add_argument("--weights", type=Path, required=True)
    detect_ball_parser.add_argument("--strategy", required=True)
    detect_ball_parser.add_argument("--output-dir", type=Path, required=True)
    detect_ball_parser.add_argument("--calibration", type=Path)
    detect_ball_parser.add_argument("--device", default="auto")

    evaluate_ball_parser = ball_subparsers.add_parser(
        "evaluate",
        help="evaluate one strategy on the fixed validation or test partition",
    )
    evaluate_ball_parser.add_argument("--config", type=Path, required=True)
    evaluate_ball_parser.add_argument("--weights", type=Path, required=True)
    evaluate_ball_parser.add_argument("--strategy", required=True)
    evaluate_ball_parser.add_argument(
        "--partition", choices=("validation", "test"), default="validation"
    )
    evaluate_ball_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_ball_parser.add_argument("--device", default="auto")

    compare_ball_parser = ball_subparsers.add_parser(
        "compare",
        help="compare multiple strategies on exactly the same fixed frames",
    )
    compare_ball_parser.add_argument("--config", type=Path, required=True)
    compare_ball_parser.add_argument("--weights", type=Path, required=True)
    compare_ball_parser.add_argument(
        "--partition", choices=("validation", "test"), default="validation"
    )
    compare_ball_parser.add_argument(
        "--strategies",
        nargs="+",
        help="strategy names from config (default: every configured strategy)",
    )
    compare_ball_parser.add_argument("--output-dir", type=Path, required=True)
    compare_ball_parser.add_argument("--device", default="auto")

    track_ball_parser = subparsers.add_parser(
        "track-ball",
        help="reconstruct the primary-match ball trajectory from raw candidates",
    )
    track_ball_parser.add_argument("video", type=Path, help="source video used for detection")
    track_ball_parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="raw detections.json from ball detect",
    )
    track_ball_parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="court calibration used only for an image-space relevance envelope",
    )
    track_ball_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for ball_tracks.json, ball-debug.mp4, and tracking summary",
    )
    return parser


def _run_doctor(settings: Settings) -> int:
    logger = logging.getLogger("pickleball_vision.cli")
    logger.info(
        "foundation_check_complete",
        extra={"context": {"environment": settings.environment.value, "status": "ok"}},
    )
    report = {
        "service": "pickleball-vision",
        "status": "ok",
        "version": __version__,
        "configuration": settings.public_values(),
    }
    print(json.dumps(report, sort_keys=True))
    return EXIT_OK


def _print_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True))


def _media_timeline(settings: Settings) -> MediaTimeline:
    return MediaTimeline(audio_video_offset_ms=settings.media.audio_video_offset_ms)


def _run_inspect(video_path: Path, *, settings: Settings) -> int:
    metadata = inspect_media(video_path, timeline=_media_timeline(settings))
    logging.getLogger("pickleball_vision.cli").info(
        "video_inspected",
        extra={
            "context": {
                "path": str(metadata.video.path),
                "has_audio": metadata.audio is not None,
            }
        },
    )
    _print_json(metadata.as_dict())
    return EXIT_OK


def _run_extract_frame(video_path: Path, *, timestamp: float, output_path: Path) -> int:
    artifact = extract_frame(
        video_path,
        timestamp_seconds=timestamp,
        output_path=output_path,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "frame_extracted",
        extra={
            "context": {
                "video": str(video_path.expanduser().resolve()),
                "frame_index": artifact.frame_index,
                "output_path": str(artifact.output_path),
            }
        },
    )
    _print_json(
        {
            "video": str(video_path.expanduser().resolve()),
            "requested_timestamp": timestamp,
            "frame": artifact.as_dict(),
        }
    )
    return EXIT_OK


def _run_extract_audio(
    video_path: Path,
    *,
    output_path: Path,
    sample_rate_hz: int | None,
    channels: int | None,
    settings: Settings,
) -> int:
    artifact = extract_audio(
        video_path,
        output_path=output_path,
        options=AudioExtractionOptions(sample_rate_hz=sample_rate_hz, channels=channels),
        timeline=_media_timeline(settings),
    )
    logging.getLogger("pickleball_vision.cli").info(
        "audio_extracted",
        extra={
            "context": {
                "video": str(artifact.source_path),
                "output_path": str(artifact.output_path),
                "sample_rate_hz": artifact.output_audio.sample_rate_hz,
                "channels": artifact.output_audio.channels,
            }
        },
    )
    _print_json(artifact.as_dict())
    return EXIT_OK


def _run_analyze_audio(
    video_path: Path,
    *,
    output_dir: Path,
    settings: Settings,
) -> int:
    artifacts = analyze_audio_in_video(
        video_path,
        output_dir=output_dir,
        settings=settings.audio_analysis,
        timeline=_media_timeline(settings),
    )
    logging.getLogger("pickleball_vision.cli").info(
        "audio_analyzed",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_sample_frames(video_path: Path, *, count: int, output_dir: Path) -> int:
    artifacts = sample_frames(video_path, count=count, output_dir=output_dir)
    resolved_video = video_path.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()
    logging.getLogger("pickleball_vision.cli").info(
        "frames_sampled",
        extra={
            "context": {
                "video": str(resolved_video),
                "count": len(artifacts),
                "output_dir": str(resolved_output_dir),
            }
        },
    )
    _print_json(
        {
            "video": str(resolved_video),
            "count": len(artifacts),
            "output_dir": str(resolved_output_dir),
            "frames": [artifact.as_dict() for artifact in artifacts],
        }
    )
    return EXIT_OK


def _run_calibrate(video_path: Path, *, timestamp: float, output_path: Path) -> int:
    artifacts = calibrate_video(
        video_path,
        timestamp_seconds=timestamp,
        output_path=output_path,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "court_calibrated",
        extra={
            "context": {
                "video": str(video_path.expanduser().resolve()),
                "frame_index": artifacts.calibration.source.frame_index,
                "correspondence_count": len(artifacts.calibration.correspondences),
                "inlier_count": artifacts.calibration.inlier_count,
                "fit_method": artifacts.calibration.fit_method.value,
                "quality_status": artifacts.calibration.quality.status.value,
                "calibration_path": str(artifacts.calibration_path),
            }
        },
    )
    _print_json(
        {
            "video": str(video_path.expanduser().resolve()),
            "requested_timestamp": timestamp,
            **artifacts.as_dict(),
        }
    )
    return EXIT_OK


def _run_detect_people(
    video_path: Path,
    *,
    calibration_path: Path,
    output_dir: Path,
    settings: Settings,
) -> int:
    artifacts = detect_people_in_video(
        video_path,
        calibration_path=calibration_path,
        output_dir=output_dir,
        settings=settings.person_detection,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "people_detected",
        extra={
            "context": {
                "video": str(video_path.expanduser().resolve()),
                "processed_frames": artifacts.processed_frame_count,
                "detections": artifacts.detection_count,
                "output_dir": str(output_dir.expanduser().resolve()),
            }
        },
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_isolate_players(
    video_path: Path,
    *,
    detections_path: Path,
    calibration_path: Path,
    timestamp: float,
    output_dir: Path,
    assignments_path: Path | None,
    settings: Settings,
) -> int:
    artifacts = isolate_primary_players(
        video_path,
        detections_path=detections_path,
        calibration_path=calibration_path,
        selection_timestamp_s=timestamp,
        output_dir=output_dir,
        settings=settings.player_isolation,
        existing_assignments_path=assignments_path,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "primary_players_isolated",
        extra={
            "context": {
                "video": str(video_path.expanduser().resolve()),
                "candidate_count": artifacts.candidate_count,
                "eligible_candidate_count": artifacts.eligible_candidate_count,
                "output_dir": str(output_dir.expanduser().resolve()),
            }
        },
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_track_players(
    video_path: Path,
    *,
    calibration_path: Path,
    output_dir: Path,
    detections_path: Path | None,
    assignments_path: Path | None,
    player_names_path: Path | None,
    settings: Settings,
) -> int:
    artifacts = track_players_in_video(
        video_path,
        calibration_path=calibration_path,
        output_dir=output_dir,
        tracking_settings=settings.player_tracking,
        isolation_settings=settings.player_isolation,
        detections_path=detections_path,
        assignments_path=assignments_path,
        player_names_path=player_names_path,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "logical_players_tracked",
        extra={
            "context": {
                "video": str(video_path.expanduser().resolve()),
                "frames_processed": artifacts.frames_processed,
                "raw_tracker_observations": artifacts.raw_tracker_observation_count,
                "suspected_identity_switches": artifacts.suspected_identity_switch_count,
                "output_dir": str(output_dir.expanduser().resolve()),
            }
        },
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_analyze_players(
    video_path: Path,
    *,
    calibration_path: Path,
    output_dir: Path,
    tracks_path: Path | None,
    position_corrections_path: Path | None,
    settings: Settings,
) -> int:
    artifacts = analyze_players_in_video(
        video_path,
        calibration_path=calibration_path,
        output_dir=output_dir,
        settings=settings.player_analysis,
        tracks_path=tracks_path,
        position_corrections_path=position_corrections_path,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "logical_player_positions_analyzed",
        extra={
            "context": {
                "video": str(video_path.expanduser().resolve()),
                "frames_processed": artifacts.frames_processed,
                "release_version": "0.1",
                "output_dir": str(output_dir.expanduser().resolve()),
            }
        },
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_dataset_extract_frames(
    video_path: Path,
    *,
    output_dir: Path,
    every_frames: int | None,
    random_count: int | None,
    random_seed: int,
    start_time_s: float | None,
    end_time_s: float | None,
    clips_path: Path | None,
    write_clips: bool,
    label_group: str,
    group_id: str | None,
    settings: Settings,
) -> int:
    artifacts = extract_ball_dataset_frames(
        video_path,
        output_dir=output_dir,
        selection=FrameSelectionSettings(
            every_frames=every_frames,
            random_count=random_count,
            random_seed=random_seed,
        ),
        label_group=DatasetLabelGroup(label_group),
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        clip_definitions_path=clips_path,
        group_id=group_id,
        write_clips=write_clips,
        timeline=_media_timeline(settings),
    )
    logging.getLogger("pickleball_vision.cli").info(
        "ball_dataset_frames_extracted",
        extra={
            "context": {
                "video": str(video_path.expanduser().resolve()),
                "frame_count": artifacts.frame_count,
                "written_clip_count": artifacts.written_clip_count,
                "output_dir": str(artifacts.output_dir),
            }
        },
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_dataset_split(
    manifest_paths: tuple[Path, ...],
    *,
    output_path: Path,
    split_by: str,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> int:
    artifact = split_ball_dataset(
        manifest_paths,
        output_path=output_path,
        split_by=SplitUnit(split_by),
        ratios=SplitRatios(
            train=train_ratio,
            validation=validation_ratio,
            test=test_ratio,
        ),
        random_seed=random_seed,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "ball_dataset_split_created",
        extra={
            "context": {
                "frame_count": artifact.frame_count,
                "unit_count": artifact.unit_count,
                "output_path": str(artifact.output_path),
            }
        },
    )
    _print_json(artifact.as_dict())
    return EXIT_OK


def _run_ball_annotation_template(
    split_manifest_path: Path,
    *,
    dataset_version: str,
    output_path: Path,
) -> int:
    artifact = create_ball_annotation_template(
        split_manifest_path,
        dataset_version=dataset_version,
        output_path=output_path,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "ball_annotation_template_created",
        extra={"context": artifact.as_dict()},
    )
    _print_json(artifact.as_dict())
    return EXIT_OK


def _run_ball_review(
    split_manifest_path: Path,
    *,
    annotations_path: Path,
    dataset_version: str | None,
    prediction_paths: tuple[Path, ...],
    port: int,
    open_browser: bool,
) -> int:
    artifacts = serve_ball_annotation_review(
        split_manifest_path,
        annotations_path=annotations_path,
        dataset_version=dataset_version,
        prediction_paths=prediction_paths,
        port=port,
        open_browser=open_browser,
        on_started=lambda url: print(f"Ball review interface: {url}", file=sys.stderr),
    )
    logging.getLogger("pickleball_vision.cli").info(
        "ball_annotation_review_stopped",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_train_ball(config_path: Path, *, output_dir: Path) -> int:
    artifacts = train_ball_detector(config_path, output_dir=output_dir)
    logging.getLogger("pickleball_vision.cli").info(
        "ball_detector_trained",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_detect_ball(
    video_path: Path,
    *,
    config_path: Path,
    weights_path: Path,
    strategy_name: str,
    output_dir: Path,
    calibration_path: Path | None,
    device: str,
) -> int:
    config = load_ball_experiment_configuration(config_path)
    artifacts = detect_balls_in_video(
        video_path,
        weights_path=weights_path,
        model_version=config.model.version,
        strategy=config.strategy(strategy_name),
        output_dir=output_dir,
        calibration_path=calibration_path,
        device=device,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "balls_detected",
        extra={"context": {**artifacts.as_dict(), "strategy": strategy_name}},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_evaluate_ball(
    config_path: Path,
    *,
    weights_path: Path,
    strategy_name: str,
    partition: str,
    output_dir: Path,
    device: str,
) -> int:
    artifacts = evaluate_ball_detector(
        config_path,
        weights_path=weights_path,
        strategy_name=strategy_name,
        partition=DatasetSplit(partition),
        output_dir=output_dir,
        device=device,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "ball_detector_evaluated",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_compare_ball(
    config_path: Path,
    *,
    weights_path: Path,
    partition: str,
    strategy_names: tuple[str, ...] | None,
    output_dir: Path,
    device: str,
) -> int:
    artifacts = compare_ball_inference_strategies(
        config_path,
        weights_path=weights_path,
        partition=DatasetSplit(partition),
        output_dir=output_dir,
        strategy_names=strategy_names,
        device=device,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "ball_inference_strategies_compared",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_track_ball(
    video_path: Path,
    *,
    detections_path: Path,
    calibration_path: Path,
    output_dir: Path,
    settings: Settings,
) -> int:
    artifacts = track_ball_in_video(
        video_path,
        detections_path=detections_path,
        calibration_path=calibration_path,
        output_dir=output_dir,
        settings=settings.ball_tracking,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "primary_match_ball_tracked",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command and translate failures into stable process exit codes."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK

    try:
        settings = Settings.from_env()
        logger = configure_logging(level=settings.log_level, log_format=settings.log_format)
        if args.command == "doctor":
            return _run_doctor(settings)
        if args.command == "inspect":
            return _run_inspect(cast(Path, args.video), settings=settings)
        if args.command == "extract-frame":
            return _run_extract_frame(
                cast(Path, args.video),
                timestamp=cast(float, args.timestamp),
                output_path=cast(Path, args.output),
            )
        if args.command == "extract-audio":
            return _run_extract_audio(
                cast(Path, args.video),
                output_path=cast(Path, args.output),
                sample_rate_hz=cast(int | None, args.sample_rate),
                channels=cast(int | None, args.channels),
                settings=settings,
            )
        if args.command == "analyze-audio":
            return _run_analyze_audio(
                cast(Path, args.video),
                output_dir=cast(Path, args.output_dir),
                settings=settings,
            )
        if args.command == "sample-frames":
            return _run_sample_frames(
                cast(Path, args.video),
                count=cast(int, args.count),
                output_dir=cast(Path, args.output_dir),
            )
        if args.command == "calibrate":
            return _run_calibrate(
                cast(Path, args.video),
                timestamp=cast(float, args.timestamp),
                output_path=cast(Path, args.output),
            )
        if args.command == "detect-people":
            return _run_detect_people(
                cast(Path, args.video),
                calibration_path=cast(Path, args.calibration),
                output_dir=cast(Path, args.output_dir),
                settings=settings,
            )
        if args.command == "isolate-players":
            return _run_isolate_players(
                cast(Path, args.video),
                detections_path=cast(Path, args.detections),
                calibration_path=cast(Path, args.calibration),
                timestamp=cast(float, args.timestamp),
                output_dir=cast(Path, args.output_dir),
                assignments_path=cast(Path | None, args.assignments),
                settings=settings,
            )
        if args.command == "track-players":
            return _run_track_players(
                cast(Path, args.video),
                calibration_path=cast(Path, args.calibration),
                output_dir=cast(Path, args.output_dir),
                detections_path=cast(Path | None, args.detections),
                assignments_path=cast(Path | None, args.assignments),
                player_names_path=cast(Path | None, args.player_names),
                settings=settings,
            )
        if args.command == "analyze-players":
            return _run_analyze_players(
                cast(Path, args.video),
                calibration_path=cast(Path, args.calibration),
                output_dir=cast(Path, args.output_dir),
                tracks_path=cast(Path | None, args.tracks),
                position_corrections_path=cast(Path | None, args.position_corrections),
                settings=settings,
            )
        if args.command == "dataset":
            if args.dataset_command == "extract-frames":
                return _run_dataset_extract_frames(
                    cast(Path, args.video),
                    output_dir=cast(Path, args.output_dir),
                    every_frames=cast(int | None, args.every_frames),
                    random_count=cast(int | None, args.random_count),
                    random_seed=cast(int, args.seed),
                    start_time_s=cast(float | None, args.start_time),
                    end_time_s=cast(float | None, args.end_time),
                    clips_path=cast(Path | None, args.clips),
                    write_clips=cast(bool, args.write_clips),
                    label_group=cast(str, args.label_group),
                    group_id=cast(str | None, args.group_id),
                    settings=settings,
                )
            if args.dataset_command == "split":
                return _run_dataset_split(
                    tuple(cast(list[Path], args.manifests)),
                    output_path=cast(Path, args.output),
                    split_by=cast(str, args.by),
                    train_ratio=cast(float, args.train),
                    validation_ratio=cast(float, args.validation),
                    test_ratio=cast(float, args.test),
                    random_seed=cast(int, args.seed),
                )
            parser.print_help()
            return EXIT_OK
        if args.command == "ball":
            if args.ball_command == "create-annotation-template":
                return _run_ball_annotation_template(
                    cast(Path, args.split_manifest),
                    dataset_version=cast(str, args.dataset_version),
                    output_path=cast(Path, args.output),
                )
            if args.ball_command == "review":
                return _run_ball_review(
                    cast(Path, args.split_manifest),
                    annotations_path=cast(Path, args.annotations),
                    dataset_version=cast(str | None, args.dataset_version),
                    prediction_paths=tuple(cast(list[Path], args.predictions)),
                    port=cast(int, args.port),
                    open_browser=not cast(bool, args.no_open),
                )
            if args.ball_command == "train":
                return _run_train_ball(
                    cast(Path, args.config),
                    output_dir=cast(Path, args.output_dir),
                )
            if args.ball_command == "detect":
                return _run_detect_ball(
                    cast(Path, args.video),
                    config_path=cast(Path, args.config),
                    weights_path=cast(Path, args.weights),
                    strategy_name=cast(str, args.strategy),
                    output_dir=cast(Path, args.output_dir),
                    calibration_path=cast(Path | None, args.calibration),
                    device=cast(str, args.device),
                )
            if args.ball_command == "evaluate":
                return _run_evaluate_ball(
                    cast(Path, args.config),
                    weights_path=cast(Path, args.weights),
                    strategy_name=cast(str, args.strategy),
                    partition=cast(str, args.partition),
                    output_dir=cast(Path, args.output_dir),
                    device=cast(str, args.device),
                )
            if args.ball_command == "compare":
                return _run_compare_ball(
                    cast(Path, args.config),
                    weights_path=cast(Path, args.weights),
                    partition=cast(str, args.partition),
                    strategy_names=(
                        tuple(cast(list[str], args.strategies))
                        if args.strategies is not None
                        else None
                    ),
                    output_dir=cast(Path, args.output_dir),
                    device=cast(str, args.device),
                )
            parser.print_help()
            return EXIT_OK
        if args.command == "track-ball":
            return _run_track_ball(
                cast(Path, args.video),
                detections_path=cast(Path, args.detections),
                calibration_path=cast(Path, args.calibration),
                output_dir=cast(Path, args.output_dir),
                settings=settings,
            )
        parser.error(f"unsupported command: {args.command}")
    except PickleballVisionError as error:
        print(f"error [{error.code}]: {error}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except Exception:
        logger = logging.getLogger("pickleball_vision.cli")
        logger.exception("unexpected_error", extra={"context": {"code": ErrorCode.INTERNAL.value}})
        print(
            f"error [{ErrorCode.INTERNAL}]: an unexpected error occurred",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR
