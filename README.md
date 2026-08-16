# Pickleball Vision

Pickleball Vision is a local-first computer-vision project for turning recorded
doubles pickleball matches into inspectable, structured match data. The long-term
goal includes court and player tracking, ball trajectories, rally and shot events,
match analytics, and AI-assisted coaching.

The current milestone is **Custom pickleball detector**. The local
CLI can inspect video plus optional synchronized audio, extract lossless analysis
audio, calibrate the court, detect people broadly, derive court-aware candidates,
manually assign the four logical match roles, and track those identities separately
from transient ByteTrack IDs. It can now derive separate raw/smoothed court positions,
movement metrics, heatmaps, and top-down review video. It can also extract
full-resolution ball-annotation frames and lossless review clips, then assign
leakage-safe dataset splits. It now supports versioned single-class ball training,
high-resolution/ROI/tiled raw inference, fixed-split evaluation, and strategy
comparison, plus a resumable local interface for human correction of training
annotations. Ball tracking and pickleball event inference remain unimplemented.

## Repository map

- `services/vision/`: typed Python package and `pickleball-vision` CLI
- `docs/`: architecture, coordinate, annotation, and analytics contracts
- `ml/`: local datasets plus versioned training, evaluation, and experiment workspaces
- `scripts/`: future repository-level utilities
- `sample-data/`: local-only media and small documented examples
- `output/`: generated local artifacts; never source-controlled
- `PLANS.md`: ordered milestones and the single current milestone
- `AGENTS.md`: durable rules for humans and coding agents

Spring Boot and Next.js are intentionally absent. Product services will be added
only after the local computer-vision pipeline is proven.

## Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) (recommended)

The locked Python environment supplies the FFmpeg libraries and extraction binary;
a separate system FFmpeg installation is not required.

## Set up the vision service

From the repository root:

```bash
cd services/vision
uv sync --locked --extra dev
```

Run the Foundation health check:

```bash
uv run pickleball-vision doctor
```

Inspect and extract from a local video:

```bash
uv run pickleball-vision inspect /absolute/path/to/match.mp4
uv run pickleball-vision extract-audio /absolute/path/to/match.mp4 \
  --output ../../output/match-audio.wav
uv run pickleball-vision extract-frame /absolute/path/to/match.mp4 \
  --timestamp 30.5 \
  --output ../../output/frame-at-30.5s.jpg
uv run pickleball-vision sample-frames /absolute/path/to/match.mp4 \
  --count 12 \
  --output-dir ../../output/sample-frames
uv run pickleball-vision calibrate /absolute/path/to/match.mp4 \
  --timestamp 30.5 \
  --output ../../output/calibration.json
uv run pickleball-vision detect-people /absolute/path/to/match.mp4 \
  --calibration ../../output/calibration.json \
  --output-dir ../../output/person-detection
uv run pickleball-vision isolate-players /absolute/path/to/match.mp4 \
  --detections ../../output/person-detection/detections.json \
  --calibration ../../output/calibration.json \
  --timestamp 30.5 \
  --output-dir ../../output/player-isolation
uv run pickleball-vision track-players /absolute/path/to/match.mp4 \
  --calibration ../../output/calibration/calibration.json \
  --output-dir ../../output/player-tracking
uv run pickleball-vision analyze-players /absolute/path/to/match.mp4 \
  --calibration ../../output/calibration/calibration.json \
  --output-dir ../../output/player-analysis \
  --position-corrections ../../output/player-tracking/player-position-corrections.json
uv run pickleball-vision dataset extract-frames /absolute/path/to/match.mp4 \
  --output-dir ../../ml/datasets/match-unlabeled \
  --every 30 \
  --label-group unlabeled
uv run pickleball-vision ball train \
  --config ../../ml/training/ball-detector.example.json \
  --output-dir ../../ml/experiments/pickleball-yolo26s-v1
uv run pickleball-vision ball review ../../ml/datasets/match-splits.json \
  --annotations ../../ml/datasets/match-annotations.json \
  --dataset-version match-v1 \
  --predictions ../../output/ball-detection/match/detections.json
```

Command results are emitted as JSON on standard output. Diagnostic structured
logs go to standard error. Video inputs keep their local provenance; do not put a
private URL in the repository or CLI command history.

`inspect` includes optional audio codec/rate/channel/duration and available stream
start times. `extract-audio` preserves source rate and channels by default, writes
PCM WAV, and records conversion and sample-to-source-time mapping in
`match-audio.wav.metadata.json`. Explicit conversion is available when needed:

```bash
uv run pickleball-vision extract-audio /absolute/path/to/match.mp4 \
  --output ../../output/match-audio-mono-16khz.wav \
  --sample-rate 16000 \
  --channels 1
```

The calibration command opens a local window for named court-landmark clicks and
writes `calibration.json`, `calibration-overlay.jpg`, and `court-topdown.jpg`.
See [the manual calibration guide](docs/court-calibration.md) for controls and
quality checks.

Configuration uses environment variables prefixed with `PICKLEBALL_VISION_`:

```bash
PICKLEBALL_VISION_LOG_LEVEL=DEBUG \
PICKLEBALL_VISION_LOG_FORMAT=console \
PICKLEBALL_VISION_OUTPUT_DIR=../../output \
uv run pickleball-vision doctor
```

Set `PICKLEBALL_VISION_AUDIO_VIDEO_OFFSET_MS` to a finite positive or negative
correction when measured evidence shows a source A/V offset; its default is zero.
See [the media timeline contract](docs/media-timeline.md).

Person inference is configured with `PERSON_MODEL`, `PERSON_DEVICE`,
`PERSON_MIN_CONFIDENCE`, `PERSON_IMAGE_SIZE`, `PERSON_IOU_THRESHOLD`, and
`PERSON_MAX_DETECTIONS`, under the same prefix. See
[the person-detection guide](docs/person-detection.md) for defaults and review
guidance. Defaults are safe for local development. Do not put secrets in
command-line arguments or committed configuration.

Primary-player isolation uses `ISOLATION_*` geometry, short-gap association, and
candidate eligibility settings documented in
[the primary-player isolation guide](docs/primary-player-isolation.md). Its local
window assigns `ME`, `PARTNER`, `OPPONENT_1`, and `OPPONENT_2`; pass the previous
`player-assignments.json` with `--assignments` to review or correct those choices.

Persistent tracking uses ByteTrack as transient motion evidence, then independently
resolves the four logical roles with the manual anchors, court side, court membership,
movement plausibility, clothing appearance, and conservative gap reacquisition. An
optional sibling `player-names.json` maps the four stable roles to display names; use
`--player-names` for a custom location. See
[the persistent-player tracking guide](docs/player-tracking.md). The standard command
finds detections through `player-assignments.json`; `--assignments` and `--detections`
remain available for nonstandard artifact locations.

Release 0.1 player analysis consumes the structured `tracks.json` artifact and
retains raw bottom-center ground points separately from optional bounded manual
court-plane corrections and bounded smoothed coordinates. It generates
movement/occupancy metrics, per-player heatmaps, a source overlay, and a top-down
animation. See [the analytics definitions](docs/analytics-definitions.md) for exact
formulas, region boundaries, quality gates, correction format, and limitations.

Ball dataset extraction supports deterministic cadence, seeded random sampling,
half-open time ranges, named clip/rally groups, optional synchronized lossless review
clips, and positive/negative/unlabeled curation queues. Dataset splits keep whole
videos, clips, or groups together. See
[the ball dataset tooling guide](docs/ball-dataset-tooling.md) and
[annotation policy](docs/annotation-guide.md).

Custom ball training requires every fixed-split frame to have explicit human-reviewed
box annotations; unreviewed frames are rejected rather than treated as negatives.
Inference supports high-resolution full-frame, calibrated court ROI, tiled, and
court-tiled strategies while retaining source-pixel proposals and detections without
temporal tracking. See [the custom detector guide](docs/ball-detector.md).

## Verify the setup

Run these exact commands from the repository root:

```bash
cd services/vision
uv sync --locked --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pickleball-vision doctor
```

Read [PLANS.md](PLANS.md) before starting work. Only its current milestone may be
implemented.
