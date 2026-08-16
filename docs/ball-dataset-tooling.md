# Ball Dataset Tooling

Milestone 7 creates local annotation material and split assignments. It does not
load, train, or run a ball detector.

## Output contract

Each extraction writes a new run directory:

```text
dataset-run/
├── dataset-manifest.json
├── images/
│   ├── positive/
│   ├── negative/
│   └── unlabeled/
└── clips/                 # only with --write-clips
```

The image files retain source resolution. `dataset-manifest.json` schema version 1
records the source content SHA-256 and media streams; extraction settings; every
frame's zero-based index, timestamp, FPS time base, grouping, and relative path; and
optional clip provenance. Source media is read-only.

`positive`, `negative`, and `unlabeled` are human curation queues, not predictions.
Begin with `unlabeled` unless a human has reviewed the selected range. See
`annotation-guide.md` for the exact pickleball labeling policy.

## Cadence, random, and time-range extraction

From `services/vision`:

```bash
uv run pickleball-vision dataset extract-frames /absolute/path/to/match.mp4 \
  --output-dir ../../ml/datasets/match-cadence \
  --every 30 \
  --label-group unlabeled
```

`--every N` selects the first eligible frame and then every Nth frame. With named
clips, cadence restarts at each clip boundary.

Seeded random sampling selects unique frames and is reproducible:

```bash
uv run pickleball-vision dataset extract-frames /absolute/path/to/match.mp4 \
  --output-dir ../../ml/datasets/match-random \
  --random-count 500 \
  --seed 2026 \
  --start-time 60 \
  --end-time 300 \
  --label-group unlabeled
```

Time ranges are half-open: start is inclusive and end is exclusive. `--clips`
cannot be combined with `--start-time` or `--end-time`.

## Named clips and rally/groups

Use a JSON file when ranges need durable clip or rally boundaries:

```json
{
  "schema_version": 1,
  "record_type": "ball_dataset_clips",
  "clips": [
    {
      "clip_id": "rally-001",
      "start_time_s": 42.5,
      "end_time_s": 55.2,
      "group_id": "rally-001",
      "label_group": "unlabeled"
    },
    {
      "clip_id": "hard-negatives-001",
      "start_time_s": 55.2,
      "end_time_s": 60.0,
      "group_id": "between-rallies-001",
      "label_group": "negative"
    }
  ]
}
```

Clip IDs must be unique and ranges must not overlap at a source frame. Extract
frames and optional synchronized review media with:

```bash
uv run pickleball-vision dataset extract-frames /absolute/path/to/match.mp4 \
  --output-dir ../../ml/datasets/match-clips \
  --clips /absolute/path/to/clips.json \
  --every 5 \
  --write-clips
```

Review clips use Matroska with lossless FFV1 video. Source audio is optional; when
present, it is trimmed on the same timeline and stored as lossless PCM with source
rate/channels preserved. The manifest records the conversion and mapping from clip
time zero to source-video time.

## Leakage-safe splits

Create reference-only split assignments from one or more extraction manifests:

```bash
uv run pickleball-vision dataset split \
  ../../ml/datasets/match-a/dataset-manifest.json \
  ../../ml/datasets/match-b/dataset-manifest.json \
  --output ../../ml/datasets/split-by-video.json \
  --by video \
  --train 0.70 \
  --validation 0.15 \
  --test 0.15 \
  --seed 2026
```

`--by video` is the safest default. `--by clip` keeps every frame from a named clip
together. `--by group` requires every record to have a `group_id` and keeps each
rally/group together. The ratios must sum to 1.0. Assignment is deterministic for a
seed and balances frame counts approximately because provenance units are never
split. Images are not copied or moved.

The split manifest becomes immutable input to custom detector annotation, training,
and evaluation. Generate explicit unreviewed annotation records and follow the next
stage in [the custom detector guide](ball-detector.md); never train directly from the
positive/negative/unlabeled directory names.
