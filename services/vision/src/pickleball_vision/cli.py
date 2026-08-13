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
from pickleball_vision.calibration_workflow import calibrate_video
from pickleball_vision.config import Settings
from pickleball_vision.errors import ErrorCode, PickleballVisionError
from pickleball_vision.logging import configure_logging
from pickleball_vision.person_detection_pipeline import detect_people_in_video
from pickleball_vision.video import extract_frame, inspect_video, sample_frames

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


def _run_inspect(video_path: Path) -> int:
    metadata = inspect_video(video_path)
    logging.getLogger("pickleball_vision.cli").info(
        "video_inspected",
        extra={"context": {"path": str(metadata.path)}},
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
            return _run_inspect(cast(Path, args.video))
        if args.command == "extract-frame":
            return _run_extract_frame(
                cast(Path, args.video),
                timestamp=cast(float, args.timestamp),
                output_path=cast(Path, args.output),
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
