# Pickleball Vision

Pickleball Vision is a local-first computer-vision project for turning recorded
doubles pickleball matches into inspectable, structured match data. The long-term
goal includes court and player tracking, ball trajectories, rally and shot events,
match analytics, and AI-assisted coaching.

The current milestone is **Person detection**. The local CLI can inspect video
metadata, extract and sample source-resolution frames, manually fit a multi-point
court-plane homography, and run broad pretrained person detection. Person
detections are observations only; the pipeline does not decide which four people
are match participants yet.

## Repository map

- `services/vision/`: typed Python package and `pickleball-vision` CLI
- `docs/`: architecture, coordinate, annotation, and analytics contracts
- `ml/`: future dataset, training, evaluation, and experiment workspaces
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
```

Command results are emitted as JSON on standard output. Diagnostic structured
logs go to standard error. Video inputs keep their local provenance; do not put a
private URL in the repository or CLI command history.

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

Person inference is configured with `PERSON_MODEL`, `PERSON_DEVICE`,
`PERSON_MIN_CONFIDENCE`, `PERSON_IMAGE_SIZE`, `PERSON_IOU_THRESHOLD`, and
`PERSON_MAX_DETECTIONS`, under the same prefix. See
[the person-detection guide](docs/person-detection.md) for defaults and review
guidance. Defaults are safe for local development. Do not put secrets in
command-line arguments or committed configuration.

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
