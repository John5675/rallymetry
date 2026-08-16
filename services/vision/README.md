# Vision service

This Python package is the local computer-vision pipeline. It currently provides
typed executable infrastructure, audio-aware FFmpeg/OpenCV media ingestion,
lossless analysis-audio extraction, manual multi-point court calibration,
pretrained person detection, court-aware manual primary-player isolation,
persistent logical-player tracking, and Release 0.1 player court-position analytics.
It also provides ball dataset extraction, review clips, leakage-safe split assignment,
versioned custom pickleball training, raw spatial inference, and fixed-split detector
evaluation, plus a local resumable human-review UI for detector annotations. Ball
tracking, audio-event detection, and later match analytics remain unimplemented.

```bash
uv sync --locked --extra dev
uv run pickleball-vision doctor
uv run pickleball-vision inspect /absolute/path/to/match.mp4
uv run pickleball-vision extract-audio /absolute/path/to/match.mp4 \
  --output ../../output/match-audio.wav
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
uv run pickleball-vision ball compare \
  --config ../../ml/training/ball-detector.example.json \
  --weights /absolute/path/to/best.pt \
  --partition validation \
  --output-dir ../../ml/evaluation/strategy-comparison
uv run pickleball-vision ball review ../../ml/datasets/match-splits.json \
  --annotations ../../ml/datasets/match-annotations.json \
  --dataset-version match-v1 \
  --predictions ../../output/ball-detection/match/detections.json
```

Calibration uses a local OpenCV window. See `docs/court-calibration.md` from the
repository root for landmark definitions and interaction controls.
Primary-player assignment also uses a local OpenCV window; see
`docs/primary-player-isolation.md` for controls, uncertainty, and artifact
contracts.
Persistent tracking consumes those assignments, learns recording-local clothing
appearance at the manual anchors, and writes raw transient tracker evidence
separately from logical roles. An optional `player-names.json` supplies display names;
see `docs/player-tracking.md`.
Release 0.1 player analysis preserves raw, manually corrected, and smoothed court
positions separately and writes qualified metrics plus heatmaps and review videos; see
`docs/analytics-definitions.md`.
Ball dataset tooling never loads a model. It preserves source/frame provenance and
keeps neighboring frames together during video/clip/group splitting; see the
[dataset guide](../../docs/ball-dataset-tooling.md) and
[annotation policy](../../docs/annotation-guide.md).
The custom detector refuses unreviewed ground truth and records dataset/model versions,
content hashes, experiment configuration, and metrics. Its full-frame, court-ROI,
tiled, and court-tiled inference modes preserve source pixels and perform no temporal
tracking; see [the detector guide](../../docs/ball-detector.md).
The detector guide also documents the loopback-only manual review UI. Yellow model
suggestions never become training labels until explicitly accepted and saved by a
human reviewer.

Audio is optional. Extraction preserves source sample rate and channels by default
and writes a timing/conversion sidecar next to the PCM WAV. See
`docs/media-timeline.md` from the repository root for the canonical source-media
timeline and `PICKLEBALL_VISION_AUDIO_VIDEO_OFFSET_MS` convention.

The package uses a `src/` layout so tests exercise the installed package rather
than accidentally importing a checkout-relative module.
