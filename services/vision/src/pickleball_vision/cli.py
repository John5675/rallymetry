"""Command-line interface for Pickleball Vision."""

from __future__ import annotations

import argparse
import asyncio
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
from pickleball_vision.bounce_detection_workflow import detect_bounces_in_video
from pickleball_vision.calibration_workflow import calibrate_video
from pickleball_vision.config import Settings
from pickleball_vision.contact_detection_workflow import detect_contacts_in_video
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
from pickleball_vision.hitter_identification_workflow import identify_hitters_in_video
from pickleball_vision.local_worker import run_local_worker
from pickleball_vision.logging import configure_logging
from pickleball_vision.match_analytics_workflow import analyze_match
from pickleball_vision.match_annotation_ui import serve_match_annotation
from pickleball_vision.media import (
    AudioExtractionOptions,
    MediaTimeline,
    extract_audio,
    inspect_media,
)
from pickleball_vision.person_detection_pipeline import detect_people_in_video
from pickleball_vision.player_analysis_workflow import analyze_players_in_video
from pickleball_vision.player_isolation_workflow import isolate_primary_players
from pickleball_vision.player_tracking_workflow import (
    track_players_in_video,
    validate_portable_player_profile,
)
from pickleball_vision.rally_segmentation_workflow import segment_rallies_in_video
from pickleball_vision.shot_dataset import build_shot_training_dataset
from pickleball_vision.shot_pretraining import pretrain_shot_representation
from pickleball_vision.shot_reconstruction_workflow import reconstruct_shots_in_video
from pickleball_vision.shot_review_overlay import apply_ai_shot_review_overlay
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
    worker_parser = subparsers.add_parser(
        "worker",
        help="poll MongoDB and process queued matches on this computer",
    )
    worker_parser.add_argument(
        "--once",
        action="store_true",
        help="claim at most one queued job and then exit",
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

    annotate_match_parser = subparsers.add_parser(
        "annotate-match",
        help="launch a resumable local editor for multimodal human ground truth",
    )
    annotate_match_parser.add_argument("video", type=Path, help="path to a local video file")
    annotate_match_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="versioned annotation JSON to create or reopen",
    )
    annotate_match_parser.add_argument(
        "--audio-events",
        type=Path,
        help="optional Prompt 10 audio-events.json for waveform/transient context",
    )
    annotate_match_parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="loopback annotation server port; use 0 to select an available port",
    )
    annotate_match_parser.add_argument(
        "--no-open",
        action="store_true",
        help="do not open the default browser automatically",
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

    validate_profile_parser = subparsers.add_parser(
        "validate-player-profile",
        help="check reviewed player anchors before expensive full-video inference",
    )
    validate_profile_parser.add_argument("video", type=Path, help="path to a local video file")
    validate_profile_parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="reviewed court calibration JSON",
    )
    validate_profile_parser.add_argument(
        "--assignments",
        type=Path,
        required=True,
        help="reviewed player-assignments.json with portable image anchors",
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
    track_parser.add_argument(
        "--portable-profile",
        action="store_true",
        help=(
            "rebind enriched manual image anchors to fresh detections in a temporary "
            "hosted-media workspace"
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

    segment_rallies_parser = subparsers.add_parser(
        "segment-rallies",
        help="infer rally intervals from structured trajectory and optional supporting signals",
    )
    segment_rallies_parser.add_argument(
        "video",
        type=Path,
        help="source video represented by the input artifacts",
    )
    segment_rallies_parser.add_argument(
        "--ball-tracks",
        type=Path,
        required=True,
        help="frame-complete ball_tracks.json from track-ball",
    )
    segment_rallies_parser.add_argument(
        "--player-tracks",
        type=Path,
        help="optional source-compatible tracks.json for player-reset confidence support",
    )
    segment_rallies_parser.add_argument(
        "--audio-events",
        type=Path,
        help="optional audio-events.json; transients support confidence but not boundaries",
    )
    segment_rallies_parser.add_argument(
        "--annotations",
        type=Path,
        help="optional human match annotations used only for post-inference evaluation",
    )
    segment_rallies_parser.add_argument(
        "--annotations-complete",
        action="store_true",
        help="treat all unannotated source time as reviewed negative evaluation coverage",
    )
    segment_rallies_parser.add_argument(
        "--evaluation-partition",
        choices=("development", "validation", "test"),
        default="validation",
        help="provenance label only; the command never tunes thresholds automatically",
    )
    segment_rallies_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for rallies.json, rally-debug.mp4, and rally-evaluation.json",
    )
    detect_bounces_parser = subparsers.add_parser(
        "detect-bounces",
        help="detect visual-first bounce candidates with optional synchronized audio support",
    )
    detect_bounces_parser.add_argument(
        "video",
        type=Path,
        help="source video represented by the input artifacts",
    )
    detect_bounces_parser.add_argument(
        "--ball-tracks",
        type=Path,
        required=True,
        help="frame-complete ball_tracks.json from track-ball",
    )
    detect_bounces_parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="court calibration used only after visual plane-contact plausibility",
    )
    detect_bounces_parser.add_argument(
        "--rallies",
        type=Path,
        help="optional source-compatible rallies.json for confidence support only",
    )
    detect_bounces_parser.add_argument(
        "--audio-events",
        type=Path,
        help="optional audio-events.json; transients cannot create bounce candidates",
    )
    detect_bounces_parser.add_argument(
        "--annotations",
        type=Path,
        help="optional human annotations used only for post-inference evaluation",
    )
    detect_bounces_parser.add_argument(
        "--annotations-complete",
        action="store_true",
        help="treat all unannotated source time as reviewed negative evaluation coverage",
    )
    detect_bounces_parser.add_argument(
        "--evaluation-partition",
        choices=("development", "validation", "test"),
        default="validation",
        help="provenance label only; the command never tunes thresholds automatically",
    )
    detect_bounces_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for bounces.json, bounce-debug.mp4, and bounce-evaluation.json",
    )
    detect_contacts_parser = subparsers.add_parser(
        "detect-contacts",
        help="detect visual-first paddle contacts with optional synchronized audio support",
    )
    detect_contacts_parser.add_argument(
        "video",
        type=Path,
        help="source video represented by the input artifacts",
    )
    detect_contacts_parser.add_argument(
        "--ball-tracks",
        type=Path,
        required=True,
        help="frame-complete ball_tracks.json from track-ball",
    )
    detect_contacts_parser.add_argument(
        "--player-tracks",
        type=Path,
        required=True,
        help="source-compatible logical tracks.json used to rank candidate players",
    )
    detect_contacts_parser.add_argument(
        "--rallies",
        type=Path,
        help="optional source-compatible rallies.json for confidence support only",
    )
    detect_contacts_parser.add_argument(
        "--bounces",
        type=Path,
        help="optional bounces.json for event-state support and coincident-bounce exclusion",
    )
    detect_contacts_parser.add_argument(
        "--audio-events",
        type=Path,
        help="optional audio-events.json; transients cannot create contact candidates",
    )
    detect_contacts_parser.add_argument(
        "--annotations",
        type=Path,
        help="optional human annotations used only for post-inference evaluation",
    )
    detect_contacts_parser.add_argument(
        "--annotations-complete",
        action="store_true",
        help="treat all unannotated source time as reviewed negative evaluation coverage",
    )
    detect_contacts_parser.add_argument(
        "--evaluation-partition",
        choices=("development", "validation", "test"),
        default="validation",
        help="provenance label only; the command never tunes thresholds automatically",
    )
    detect_contacts_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for contacts.json, contact-debug.mp4, and contact-evaluation.json",
    )
    identify_hitters_parser = subparsers.add_parser(
        "identify-hitters",
        help="resolve contact candidates to logical players or UNKNOWN",
    )
    identify_hitters_parser.add_argument(
        "video",
        type=Path,
        help="source video represented by the contact and player-track artifacts",
    )
    identify_hitters_parser.add_argument(
        "--contacts",
        type=Path,
        required=True,
        help="source-compatible contacts.json from detect-contacts",
    )
    identify_hitters_parser.add_argument(
        "--player-tracks",
        type=Path,
        required=True,
        help="exact logical tracks.json used to generate the contact artifact",
    )
    identify_hitters_parser.add_argument(
        "--annotations",
        type=Path,
        help="optional human contact player labels used only for evaluation",
    )
    identify_hitters_parser.add_argument(
        "--evaluation-partition",
        choices=("development", "validation", "test"),
        default="validation",
        help="provenance label only; the command never tunes thresholds automatically",
    )
    identify_hitters_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for hitters.json, hitter-debug.mp4, and hitter-evaluation.json",
    )
    reconstruct_shots_parser = subparsers.add_parser(
        "reconstruct-shots",
        help="reconstruct rally-local shots and apply documented rule-based classes",
    )
    reconstruct_shots_parser.add_argument("video", type=Path, help="source match video")
    reconstruct_shots_parser.add_argument(
        "--ball-tracks",
        type=Path,
        required=True,
        help="frame-complete ball_tracks.json",
    )
    reconstruct_shots_parser.add_argument(
        "--rallies",
        type=Path,
        required=True,
        help="source-compatible automatic rallies.json",
    )
    reconstruct_shots_parser.add_argument(
        "--contacts",
        type=Path,
        required=True,
        help="source-compatible contacts.json",
    )
    reconstruct_shots_parser.add_argument(
        "--bounces",
        type=Path,
        help="optional source-compatible bounces.json for landing linkage",
    )
    reconstruct_shots_parser.add_argument(
        "--hitters",
        type=Path,
        required=True,
        help="source-compatible hitters.json",
    )
    reconstruct_shots_parser.add_argument(
        "--player-tracks",
        type=Path,
        required=True,
        help="exact tracks.json referenced by contacts and hitters",
    )
    reconstruct_shots_parser.add_argument(
        "--annotations",
        type=Path,
        help="optional human shotType labels used only for evaluation",
    )
    reconstruct_shots_parser.add_argument(
        "--evaluation-partition",
        choices=("development", "validation", "test"),
        default="validation",
        help="provenance label only; thresholds are never tuned automatically",
    )
    reconstruct_shots_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for shots.json, shot-debug.mp4, and shot-evaluation.json",
    )
    analyze_match_parser = subparsers.add_parser(
        "analyze-match",
        help="compute deterministic match analytics from structured domain artifacts",
    )
    analyze_match_parser.add_argument("video", type=Path, help="source match video")
    analyze_match_parser.add_argument(
        "--rallies",
        type=Path,
        required=True,
        help="source-compatible automatic rallies.json",
    )
    analyze_match_parser.add_argument(
        "--shots",
        type=Path,
        required=True,
        help="source-compatible reconstructed shots.json",
    )
    analyze_match_parser.add_argument(
        "--player-positions",
        type=Path,
        required=True,
        help="source-compatible structured player_positions.json",
    )
    analyze_match_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new match-analytics.json output path",
    )

    shot_model_parser = subparsers.add_parser(
        "shot-model",
        help="audit multi-axis shot labels and pretrain temporal representations",
    )
    shot_model_subparsers = shot_model_parser.add_subparsers(dest="shot_model_command")
    shot_dataset_parser = shot_model_subparsers.add_parser(
        "build-dataset",
        help="apply auditable corrections and enforce semantic-training gates",
    )
    shot_dataset_parser.add_argument(
        "source_dataset",
        type=Path,
        help="AI-adjudicated multievent dataset JSON",
    )
    shot_dataset_parser.add_argument(
        "--corrections",
        type=Path,
        help="optional correction layer; the source dataset is never overwritten",
    )
    shot_dataset_parser.add_argument("--output-dir", type=Path, required=True)
    shot_dataset_parser.add_argument(
        "--minimum-train-per-class",
        type=int,
        default=10,
        help="minimum train support for every claimed semantic-axis class",
    )
    shot_dataset_parser.add_argument(
        "--minimum-held-out-per-class",
        type=int,
        default=5,
        help="minimum validation and test support per claimed class",
    )
    shot_pretrain_parser = shot_model_subparsers.add_parser(
        "pretrain-representation",
        help="pretrain a temporal encoder on licensed ball/racket annotations only",
    )
    shot_pretrain_parser.add_argument("--config", type=Path, required=True)
    shot_pretrain_parser.add_argument("--output-dir", type=Path, required=True)
    shot_review_parser = shot_model_subparsers.add_parser(
        "apply-review",
        help="attach AI visual review evidence for an exact reviewed source video",
    )
    shot_review_parser.add_argument("video", type=Path, help="exact local source video")
    shot_review_parser.add_argument(
        "--shots",
        type=Path,
        required=True,
        help="machine-generated shots.json to preserve and enrich",
    )
    shot_review_parser.add_argument(
        "--review-index",
        type=Path,
        help="optional reviewed-label index (default: bundled eight-video review)",
    )
    shot_review_parser.add_argument("--output", type=Path, required=True)
    shot_review_parser.add_argument(
        "--maximum-timing-delta-ms",
        type=float,
        default=250.0,
        help="maximum contact-time difference for one-to-one review matching",
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
    return MediaTimeline(
        audio_video_offset_ms=settings.media.audio_video_offset_ms,
        fusion_tolerance_ms=settings.media.fusion_tolerance_ms,
    )


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


def _run_annotate_match(
    video_path: Path,
    *,
    output_path: Path,
    audio_events_path: Path | None,
    port: int,
    open_browser: bool,
    settings: Settings,
) -> int:
    artifacts = serve_match_annotation(
        video_path,
        output_path=output_path,
        timeline=_media_timeline(settings),
        audio_events_path=audio_events_path,
        port=port,
        open_browser=open_browser,
        on_started=lambda url: print(f"Match annotation interface: {url}", file=sys.stderr),
    )
    logging.getLogger("pickleball_vision.cli").info(
        "match_annotation_stopped",
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


def _run_validate_player_profile(
    video_path: Path,
    *,
    calibration_path: Path,
    assignments_path: Path,
    settings: Settings,
) -> int:
    result = validate_portable_player_profile(
        video_path,
        calibration_path=calibration_path,
        assignments_path=assignments_path,
        person_settings=settings.person_detection,
        isolation_settings=settings.player_isolation,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "portable_player_profile_validated",
        extra={"context": result},
    )
    _print_json(result)
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
    allow_portable_profile: bool,
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
        allow_portable_profile=allow_portable_profile,
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


def _run_segment_rallies(
    video_path: Path,
    *,
    ball_tracks_path: Path,
    player_tracks_path: Path | None,
    audio_events_path: Path | None,
    annotations_path: Path | None,
    annotations_complete: bool,
    evaluation_partition: str,
    output_dir: Path,
    settings: Settings,
) -> int:
    artifacts = segment_rallies_in_video(
        video_path,
        ball_tracks_path=ball_tracks_path,
        player_tracks_path=player_tracks_path,
        audio_events_path=audio_events_path,
        annotations_path=annotations_path,
        annotations_complete=annotations_complete,
        evaluation_partition=evaluation_partition,
        output_dir=output_dir,
        settings=settings.rally_segmentation,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "rallies_segmented",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_detect_bounces(
    video_path: Path,
    *,
    ball_tracks_path: Path,
    calibration_path: Path,
    rallies_path: Path | None,
    audio_events_path: Path | None,
    annotations_path: Path | None,
    annotations_complete: bool,
    evaluation_partition: str,
    output_dir: Path,
    settings: Settings,
) -> int:
    artifacts = detect_bounces_in_video(
        video_path,
        ball_tracks_path=ball_tracks_path,
        calibration_path=calibration_path,
        rallies_path=rallies_path,
        audio_events_path=audio_events_path,
        annotations_path=annotations_path,
        annotations_complete=annotations_complete,
        evaluation_partition=evaluation_partition,
        output_dir=output_dir,
        settings=settings.bounce_detection,
        timeline=_media_timeline(settings),
    )
    logging.getLogger("pickleball_vision.cli").info(
        "bounces_detected",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_detect_contacts(
    video_path: Path,
    *,
    ball_tracks_path: Path,
    player_tracks_path: Path,
    rallies_path: Path | None,
    bounces_path: Path | None,
    audio_events_path: Path | None,
    annotations_path: Path | None,
    annotations_complete: bool,
    evaluation_partition: str,
    output_dir: Path,
    settings: Settings,
) -> int:
    artifacts = detect_contacts_in_video(
        video_path,
        ball_tracks_path=ball_tracks_path,
        player_tracks_path=player_tracks_path,
        rallies_path=rallies_path,
        bounces_path=bounces_path,
        audio_events_path=audio_events_path,
        annotations_path=annotations_path,
        annotations_complete=annotations_complete,
        evaluation_partition=evaluation_partition,
        output_dir=output_dir,
        settings=settings.contact_detection,
        timeline=_media_timeline(settings),
    )
    logging.getLogger("pickleball_vision.cli").info(
        "contacts_detected",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_identify_hitters(
    video_path: Path,
    *,
    contacts_path: Path,
    player_tracks_path: Path,
    annotations_path: Path | None,
    evaluation_partition: str,
    output_dir: Path,
    settings: Settings,
) -> int:
    artifacts = identify_hitters_in_video(
        video_path,
        contacts_path=contacts_path,
        player_tracks_path=player_tracks_path,
        annotations_path=annotations_path,
        evaluation_partition=evaluation_partition,
        output_dir=output_dir,
        settings=settings.hitter_identification,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "hitters_identified",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_reconstruct_shots(
    video_path: Path,
    *,
    ball_tracks_path: Path,
    rallies_path: Path,
    contacts_path: Path,
    bounces_path: Path | None,
    hitters_path: Path,
    player_tracks_path: Path,
    annotations_path: Path | None,
    evaluation_partition: str,
    output_dir: Path,
    settings: Settings,
) -> int:
    artifacts = reconstruct_shots_in_video(
        video_path,
        ball_tracks_path=ball_tracks_path,
        rallies_path=rallies_path,
        contacts_path=contacts_path,
        bounces_path=bounces_path,
        hitters_path=hitters_path,
        player_tracks_path=player_tracks_path,
        annotations_path=annotations_path,
        evaluation_partition=evaluation_partition,
        output_dir=output_dir,
        settings=settings.shot_classification,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "shots_reconstructed",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_analyze_match(
    video_path: Path,
    *,
    rallies_path: Path,
    shots_path: Path,
    player_positions_path: Path,
    output_path: Path,
    settings: Settings,
) -> int:
    artifacts = analyze_match(
        video_path,
        rallies_path=rallies_path,
        shots_path=shots_path,
        player_positions_path=player_positions_path,
        output_path=output_path,
        settings=settings.match_analytics,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "match_analytics_computed",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_build_shot_dataset(
    source_dataset_path: Path,
    *,
    corrections_path: Path | None,
    output_dir: Path,
    minimum_train_per_class: int,
    minimum_held_out_per_class: int,
) -> int:
    artifacts = build_shot_training_dataset(
        source_dataset_path,
        corrections_path=corrections_path,
        output_dir=output_dir,
        minimum_train_examples_per_class=minimum_train_per_class,
        minimum_held_out_examples_per_class=minimum_held_out_per_class,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "shot_training_dataset_audited",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_pretrain_shot_representation(
    config_path: Path,
    *,
    output_dir: Path,
) -> int:
    artifacts = pretrain_shot_representation(config_path, output_dir=output_dir)
    logging.getLogger("pickleball_vision.cli").info(
        "shot_temporal_representation_pretrained",
        extra={"context": artifacts.as_dict()},
    )
    _print_json(artifacts.as_dict())
    return EXIT_OK


def _run_apply_shot_review(
    video_path: Path,
    *,
    shots_path: Path,
    review_index_path: Path | None,
    output_path: Path,
    maximum_timing_delta_ms: float,
) -> int:
    artifacts = apply_ai_shot_review_overlay(
        video_path,
        shots_path=shots_path,
        output_path=output_path,
        review_index_path=review_index_path,
        maximum_timing_delta_ms=maximum_timing_delta_ms,
    )
    logging.getLogger("pickleball_vision.cli").info(
        "shot_ai_review_overlay_applied",
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
        if args.command == "worker":
            return asyncio.run(run_local_worker(once=cast(bool, args.once)))
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
        if args.command == "annotate-match":
            return _run_annotate_match(
                cast(Path, args.video),
                output_path=cast(Path, args.output),
                audio_events_path=cast(Path | None, args.audio_events),
                port=cast(int, args.port),
                open_browser=not cast(bool, args.no_open),
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
        if args.command == "validate-player-profile":
            return _run_validate_player_profile(
                cast(Path, args.video),
                calibration_path=cast(Path, args.calibration),
                assignments_path=cast(Path, args.assignments),
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
                allow_portable_profile=cast(bool, args.portable_profile),
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
        if args.command == "segment-rallies":
            return _run_segment_rallies(
                cast(Path, args.video),
                ball_tracks_path=cast(Path, args.ball_tracks),
                player_tracks_path=cast(Path | None, args.player_tracks),
                audio_events_path=cast(Path | None, args.audio_events),
                annotations_path=cast(Path | None, args.annotations),
                annotations_complete=cast(bool, args.annotations_complete),
                evaluation_partition=cast(str, args.evaluation_partition),
                output_dir=cast(Path, args.output_dir),
                settings=settings,
            )
        if args.command == "detect-bounces":
            return _run_detect_bounces(
                cast(Path, args.video),
                ball_tracks_path=cast(Path, args.ball_tracks),
                calibration_path=cast(Path, args.calibration),
                rallies_path=cast(Path | None, args.rallies),
                audio_events_path=cast(Path | None, args.audio_events),
                annotations_path=cast(Path | None, args.annotations),
                annotations_complete=cast(bool, args.annotations_complete),
                evaluation_partition=cast(str, args.evaluation_partition),
                output_dir=cast(Path, args.output_dir),
                settings=settings,
            )
        if args.command == "detect-contacts":
            return _run_detect_contacts(
                cast(Path, args.video),
                ball_tracks_path=cast(Path, args.ball_tracks),
                player_tracks_path=cast(Path, args.player_tracks),
                rallies_path=cast(Path | None, args.rallies),
                bounces_path=cast(Path | None, args.bounces),
                audio_events_path=cast(Path | None, args.audio_events),
                annotations_path=cast(Path | None, args.annotations),
                annotations_complete=cast(bool, args.annotations_complete),
                evaluation_partition=cast(str, args.evaluation_partition),
                output_dir=cast(Path, args.output_dir),
                settings=settings,
            )
        if args.command == "identify-hitters":
            return _run_identify_hitters(
                cast(Path, args.video),
                contacts_path=cast(Path, args.contacts),
                player_tracks_path=cast(Path, args.player_tracks),
                annotations_path=cast(Path | None, args.annotations),
                evaluation_partition=cast(str, args.evaluation_partition),
                output_dir=cast(Path, args.output_dir),
                settings=settings,
            )
        if args.command == "reconstruct-shots":
            return _run_reconstruct_shots(
                cast(Path, args.video),
                ball_tracks_path=cast(Path, args.ball_tracks),
                rallies_path=cast(Path, args.rallies),
                contacts_path=cast(Path, args.contacts),
                bounces_path=cast(Path | None, args.bounces),
                hitters_path=cast(Path, args.hitters),
                player_tracks_path=cast(Path, args.player_tracks),
                annotations_path=cast(Path | None, args.annotations),
                evaluation_partition=cast(str, args.evaluation_partition),
                output_dir=cast(Path, args.output_dir),
                settings=settings,
            )
        if args.command == "analyze-match":
            return _run_analyze_match(
                cast(Path, args.video),
                rallies_path=cast(Path, args.rallies),
                shots_path=cast(Path, args.shots),
                player_positions_path=cast(Path, args.player_positions),
                output_path=cast(Path, args.output),
                settings=settings,
            )
        if args.command == "shot-model":
            if args.shot_model_command == "build-dataset":
                return _run_build_shot_dataset(
                    cast(Path, args.source_dataset),
                    corrections_path=cast(Path | None, args.corrections),
                    output_dir=cast(Path, args.output_dir),
                    minimum_train_per_class=cast(int, args.minimum_train_per_class),
                    minimum_held_out_per_class=cast(int, args.minimum_held_out_per_class),
                )
            if args.shot_model_command == "pretrain-representation":
                return _run_pretrain_shot_representation(
                    cast(Path, args.config),
                    output_dir=cast(Path, args.output_dir),
                )
            if args.shot_model_command == "apply-review":
                return _run_apply_shot_review(
                    cast(Path, args.video),
                    shots_path=cast(Path, args.shots),
                    review_index_path=cast(Path | None, args.review_index),
                    output_path=cast(Path, args.output),
                    maximum_timing_delta_ms=cast(float, args.maximum_timing_delta_ms),
                )
            parser.print_help()
            return EXIT_OK
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
