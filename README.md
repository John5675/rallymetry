# Pickleball Vision

Pickleball Vision is a local-first computer-vision project for turning recorded
doubles pickleball matches into inspectable, structured match data. The long-term
goal includes court and player tracking, ball trajectories, rally and shot events,
match analytics, and AI-assisted coaching.

The latest completed milestone is **MongoDB job queue + separate analysis worker**.
Milestones 0–21 are complete. The local
CLI can inspect video plus optional synchronized audio, extract lossless analysis
audio, calibrate the court, detect people broadly, derive court-aware candidates,
manually assign the four logical match roles, and track those identities separately
from transient ByteTrack IDs. It can now derive separate raw/smoothed court positions,
movement metrics, heatmaps, and top-down review video. It can also extract
full-resolution ball-annotation frames and lossless review clips, then assign
leakage-safe dataset splits. It now supports versioned single-class ball training,
high-resolution/ROI/tiled raw inference, fixed-split evaluation, and strategy
comparison, plus a resumable local interface for human correction of training
annotations. It now reconstructs a conservative primary-match image-space ball path
with explicit observed, interpolated, and unknown states. Synchronized audio can now
be converted into inspectable raw signal features and generic transient candidates.
It also provides a resumable local editor for explicit human rally, contact, bounce,
winner, shot-type, and optional audio-context annotations. Structured automatic
rally segmentation now combines primary-ball activity, sustained motion, gaps,
serve-like onsets, adjacent-burst dead-ball handoff rejection, and optional
player/audio confidence support, then evaluates
against human rally boundaries. Visual-first bounce candidates now combine local
trajectory reversal, continuity, and optional synchronized audio support while
gating court projection behind plane-contact plausibility. Visual-first paddle-
contact candidates now combine trajectory velocity/direction discontinuity,
logical-player proximity, rally/event-state context, and optional synchronized
audio support without assigning a hitter. The separate hitter-identification stage
now combines logical-player proximity, tracking confidence, court side, visual
trajectory direction, prior credible hitter, rally order, and visual contact
confidence. It retains `UNKNOWN` when evidence is insufficient and never uses audio
to choose a player. Accepted rally-local contacts can now be reconstructed into
structured shots and assigned one of nine deliberately small, rule-based classes,
including `OTHER` and `UNKNOWN`. Deterministic analytics now consume only those
structured domain objects. Optional hosted persistence stores compact match records
through the official PyMongo Async API and large artifacts through interchangeable
local-filesystem or Vercel Blob adapters without making cloud access a CLI
prerequisite. A separate FastAPI control plane now exposes JSON match/result records
and queues durable processing-job status without running analysis in HTTP requests.
A separate outbound-only worker now atomically claims those MongoDB jobs, maintains
bounded leases and heartbeats, stages local or Blob media, invokes an explicit
operator-controlled plan of existing CLI stages, persists compact structured
results, and publishes selected artifacts through the configured store.

## Repository map

- `services/vision/`: typed Python package and `pickleball-vision` CLI
- `docs/`: architecture, coordinate, annotation, and analytics contracts
- `ml/`: local datasets plus versioned training, evaluation, and experiment workspaces
- `scripts/`: future repository-level utilities
- `sample-data/`: local-only media and small documented examples
- `output/`: generated local artifacts; never source-controlled
- `PLANS.md`: ordered milestones and the single current milestone
- `AGENTS.md`: durable rules for humans and coding agents

The locked product stack is a React/Vite/TypeScript frontend, a FastAPI product API,
MongoDB Atlas for hosted structured data and the initial small-scale job queue,
Vercel Blob for hosted media/artifacts, and a separate Python analysis worker that
invokes the existing pipeline. Heavy analysis will not run in Vercel Functions or
inside FastAPI HTTP requests. The persistence adapters, FastAPI control plane, and
analysis worker are now implemented; the browser application remains deferred. See
the
[architecture contract](docs/architecture.md) and
[hosted persistence contract](docs/persistence.md), plus the
[API contract](docs/api.md) and [worker contract](docs/worker.md).

## Optional hosted persistence

Local artifacts remain the default and require no credentials. Copy `.env.example`
as a reference, then inject real values through your shell or deployment secret
store only when a future API or worker uses the hosted adapters:

```bash
export MONGODB_URL='mongodb+srv://<user>:<password>@<cluster>/'
export MONGODB_DATABASE='pickleball_vision'
export PICKLEBALL_VISION_ARTIFACT_BACKEND='vercel_blob'
export BLOB_READ_WRITE_TOKEN='<server-side-token>'
```

Do not expose either credential to a browser. MongoDB stores compact structured
records and artifact references; videos, frames, audio waveforms, model weights,
large detections, and debug media remain in an artifact store.

For the supported Vercel project-link, private-store creation, local environment
pull, and worker startup sequence, see the
[hosted persistence setup](docs/persistence.md#provision-a-private-vercel-blob-store).

Run the API from `services/vision` after configuring MongoDB:

```bash
export CORS_ORIGINS='http://localhost:5173'
uv run uvicorn pickleball_vision.api.main:app --host 127.0.0.1 --port 8000
```

`POST /api/matches/{matchId}/process` returns `202 Accepted` with a queued job ID
after verifying the match has a `SOURCE_MEDIA` artifact. Run a single local claim
with the trusted example plan (after replacing its match/model path placeholders):

```bash
uv run pickleball-vision worker \
  --pipeline-plan ../../docs/examples/worker-pipeline-plan.json \
  --once
```

Omit `--once` for continuous single-concurrency polling. The worker only needs
outbound MongoDB Atlas and Vercel Blob access; it opens no inbound server.

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
uv run pickleball-vision analyze-audio /absolute/path/to/match.mp4 \
  --output-dir ../../output/audio-analysis
uv run pickleball-vision annotate-match /absolute/path/to/match.mp4 \
  --output ../../output/match-annotations.json \
  --audio-events ../../output/audio-analysis/audio-events.json
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
uv run pickleball-vision track-ball /absolute/path/to/match.mp4 \
  --detections ../../output/ball-detection/match/detections.json \
  --calibration ../../output/calibration/calibration.json \
  --output-dir ../../output/ball-tracking
uv run pickleball-vision segment-rallies /absolute/path/to/match.mp4 \
  --ball-tracks ../../output/ball-tracking/ball_tracks.json \
  --audio-events ../../output/audio-analysis/audio-events.json \
  --annotations ../../output/match-annotations.json \
  --output-dir ../../output/rally-segmentation
uv run pickleball-vision detect-bounces /absolute/path/to/match.mp4 \
  --ball-tracks ../../output/ball-tracking/ball_tracks.json \
  --calibration ../../output/calibration/calibration.json \
  --rallies ../../output/rally-segmentation/rallies.json \
  --audio-events ../../output/audio-analysis/audio-events.json \
  --annotations ../../output/match-annotations.json \
  --output-dir ../../output/bounce-detection
uv run pickleball-vision detect-contacts /absolute/path/to/match.mp4 \
  --ball-tracks ../../output/ball-tracking/ball_tracks.json \
  --player-tracks ../../output/player-tracking/tracks.json \
  --rallies ../../output/rally-segmentation/rallies.json \
  --bounces ../../output/bounce-detection/bounces.json \
  --audio-events ../../output/audio-analysis/audio-events.json \
  --annotations ../../output/match-annotations.json \
  --output-dir ../../output/contact-detection
uv run pickleball-vision identify-hitters /absolute/path/to/match.mp4 \
  --contacts ../../output/contact-detection/contacts.json \
  --player-tracks ../../output/player-tracking/tracks.json \
  --annotations ../../output/match-annotations.json \
  --output-dir ../../output/hitter-identification
uv run pickleball-vision reconstruct-shots /absolute/path/to/match.mp4 \
  --ball-tracks ../../output/ball-tracking/ball_tracks.json \
  --rallies ../../output/rally-segmentation/rallies.json \
  --contacts ../../output/contact-detection/contacts.json \
  --bounces ../../output/bounce-detection/bounces.json \
  --hitters ../../output/hitter-identification/hitters.json \
  --player-tracks ../../output/player-tracking/tracks.json \
  --annotations ../../output/match-annotations.json \
  --output-dir ../../output/shot-reconstruction
uv run pickleball-vision analyze-match /absolute/path/to/match.mp4 \
  --rallies ../../output/rally-segmentation/rallies.json \
  --shots ../../output/shot-reconstruction/shots.json \
  --player-positions ../../output/player-analysis/player_positions.json \
  --output ../../output/match-analytics.json
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

Audio analysis preserves channels in a synchronized PCM representation, stores raw
window features separately from generic `TRANSIENT` candidates, and creates waveform
and event-timeline review images. A transient is never classified as a paddle contact
or bounce at this stage, and no-audio videos exit successfully with an explicit
vision-only fallback. See [the audio-analysis guide](docs/audio-analysis.md).

Multimodal match ground truth is created in a loopback-only browser editor. It
supports native playback, exact frame stepping, seeking, add/edit/delete, atomic
resume, and optional waveform/transient context without treating audio as a
semantic event. See [the match annotation guide](docs/match-annotation.md).

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

Deterministic match analytics consume only the structured rally, shot, and player
position artifacts. The stage verifies source/provenance compatibility, preserves
unknown hitters and shot classes, and writes a complete `match-analytics.json`
without consulting detector tensors, YOLO output, or raw audio. The same
[analytics definitions](docs/analytics-definitions.md) specify every formula,
denominator, missing-data rule, confidence limitation, and known inaccuracy.

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

Ball trajectory reconstruction associates raw candidates using image-space motion,
acceleration plausibility, temporal persistence, confidence, and a calibrated
primary-court image envelope. It retains raw observations separately from bounded
smoothing, marks only short internal gaps as interpolated, and leaves longer gaps
unknown. Airborne ball points are never projected through court homography. See
[the ball trajectory guide](docs/ball-tracking.md).

Automatic rally segmentation consumes the structured trajectory and uses explicit
motion, gap, serve-like onset, and activity-burst signals. Compatible player tracks
and generic audio transients may support confidence but cannot create a boundary.
Materially weaker bursts immediately adjacent to stronger rally evidence are
retained as inspectable rejected candidates instead of being reported as rallies.
Human annotations are isolated to post-inference evaluation, and sparse annotation
files do not make unreviewed video time negative. See
[the rally segmentation guide](docs/rally-segmentation.md).

Multimodal bounce detection requires visual direction reversal, local trajectory
shape, and continuity before optional audio can increase confidence. Audio alone
creates no candidate. Court coordinates remain null unless the visual candidate is
plausibly on the plane and inside the projected court image; no 3D position or line
call is inferred. Evaluation compares visual-only and fused thresholds over the
same candidate set. See [the bounce detection guide](docs/bounce-detection.md).

Multimodal paddle-contact detection requires a visual trajectory velocity/direction
discontinuity before player, rally, bounce-state, or audio context can affect
confidence. Candidate-player rankings retain logical roles and proximity evidence
but never assign a hitter. Audio alone creates no contact. Evaluation compares
visual-only and fused thresholds over the same candidate set. See
[the paddle-contact detection guide](docs/contact-detection.md).

Hitter identification is a separate derived layer and never writes into source
contact candidates. It uses visual contact confidence, player proximity/position,
court side, tracking quality, trajectory direction, previous credible hitter, and
rally order. All assignment gates and weights are recorded; ambiguous cases remain
`UNKNOWN`. Human player labels are used only for post-inference accuracy evaluation.
See [the hitter-identification guide](docs/hitter-identification.md).

Shot reconstruction groups accepted contacts inside predicted rallies and links a
bounded source-trajectory segment, optional accepted bounce/landing, logical hitter,
and raw bottom-center hitter position. The ordered classifier supports only
`SERVE`, `RETURN`, `DINK`, `DROP`, `DRIVE`, `VOLLEY`, `OVERHEAD`, `OTHER`, and
`UNKNOWN`; it stores every tested feature and rule. See
[the shot-reconstruction guide](docs/shot-reconstruction.md).
The shot debug video also retains the recent image-space ball trail, with observed
and interpolated points distinguished and unknown gaps left disconnected.

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
