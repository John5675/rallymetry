# Custom Pickleball Detector

Milestone 8 trains and evaluates one class: `pickleball`. It does not track the ball,
interpolate missing observations, or infer trajectories, bounces, contacts, or other
events.

## Prerequisites and fixed data

Training consumes three immutable inputs:

1. a Milestone 7 `ball_dataset_split_assignments` manifest;
2. a human-reviewed `pickleball_detection_annotations` manifest; and
3. a versioned experiment JSON configuration.

The split manifest fixes train, validation, and test record IDs. Its SHA-256 and the
annotation manifest SHA-256 are written into every experiment. All three partitions
must be nonempty, no split unit may cross partitions, and every referenced frame must
have a `review_status` of `reviewed`. An empty `objects` array is a negative only after
that explicit review. Ambiguous boxes are rejected as detector ground truth.

Generate a safe, entirely unreviewed annotation template from `services/vision`:

```bash
uv run pickleball-vision ball create-annotation-template \
  ../../ml/datasets/vid-splits.json \
  --dataset-version vid-v1 \
  --output ../../ml/datasets/vid-annotations.json
```

One reviewed positive frame has this form:

```json
{
  "record_id": "sha256:SOURCE:frame:123",
  "review_status": "reviewed",
  "objects": [
    {
      "annotation_id": "ball-123-1",
      "class_name": "pickleball",
      "bounding_box": {
        "left_px": 920.0,
        "top_px": 410.0,
        "right_px": 932.0,
        "bottom_px": 422.0
      },
      "court_side": "far",
      "scope": "primary_match",
      "visibility": "blurred"
    }
  ]
}
```

`court_side` is a human annotation (`near`, `far`, or `unknown`). Evaluation never
projects an airborne ball through court homography to manufacture a side label.
Neighboring-court balls remain class `pickleball` with
`scope=neighboring_court`.

## Resumable manual review interface

The review interface is a local browser UI served only on `127.0.0.1`; it is not a
product backend. Its queue is an immutable split manifest. It can create a new
unreviewed annotation file or resume an existing one, and it writes an adjacent
`*.review-summary.json` progress report after every change.

Raw `detections.json` files are optional suggestion sources. Yellow dashed boxes are
model predictions and never become ground truth automatically. Click a suggestion to
accept it, drag to draw a missing box, or right-click a human box to remove it. Green
boxes are human annotations. For every accepted ball, record human-owned near/far
context, primary/neighboring/unknown scope, visibility, and annotation confidence.

Launch from `services/vision`:

```bash
uv run pickleball-vision ball review \
  ../../ml/datasets/long-match-splits.json \
  --annotations ../../ml/datasets/long-match-annotations.json \
  --dataset-version long-match-v1 \
  --predictions ../../output/ball-detection/long-match/detections.json
```

`--dataset-version` is required when creating the annotations file and becomes an
optional consistency check on later resume runs. Repeat `--predictions` when a split
references multiple source videos. Use `--port 0` to select any available port or
`--no-open` to print the URL without opening the browser.

Controls:

- `R` saves the current boxes as reviewed ground truth.
- `N` explicitly saves a reviewed negative with no visible balls.
- `S` saves a resumable unreviewed draft, which training will reject.
- `U` clears the frame back to unreviewed.
- Left/right arrows move through the selected queue filter.
- Delete removes the selected human box. Browser zoom and the interface zoom slider
  support tiny far-side balls.

The queue filters expose unreviewed frames, frames with or without suggestions, and
low-confidence suggestions. Review both detections and no-detection frames: otherwise
the next model will learn accepted predictions but not enough missed balls or hard
negatives. Fully occluded balls receive no box; visible neighboring-court balls still
receive a box with `scope=neighboring_court`.

For a longer recording, create explicit non-overlapping clip/rally groups before
splitting so neighboring frames remain together. Run the existing detector over the
same source to obtain suggestions, then review the new fixed split. Give the expanded
corpus a new dataset version and do not tune against the held-out test partition.

## Experiment configuration and training

Copy and edit
[`ml/training/ball-detector.example.json`](../ml/training/ball-detector.example.json).
Paths inside the JSON resolve relative to the configuration file. The configuration
owns:

- human dataset version and fixed manifest paths;
- model version and pretrained base model;
- epochs, training resolution, batch, device, workers, seed, deterministic mode,
  and patience; and
- every named inference strategy and evaluation IoU threshold.

Train from `services/vision`:

```bash
uv run pickleball-vision ball train \
  --config ../../ml/training/ball-detector.example.json \
  --output-dir ../../ml/experiments/pickleball-yolo26s-v1
```

The command creates deterministic image links and YOLO label files inside the run,
then calls the isolated Ultralytics training adapter. The run contains
`experiment.json`, `metrics.json`, `prepared-dataset/`, backend logs/plots, and
ignored `best.pt`/`last.pt` weights. Experiment metadata records configuration,
dataset and annotation hashes, code revision, Python/platform/package versions,
seed, effective device, model version, and weight hash. Failed runs retain a failed
experiment record rather than looking complete.

## Spatial inference strategies

All final boxes use original source-frame pixels. Each output also retains every
per-crop proposal before cross-crop overlap suppression.

- `full_frame`: run the entire source frame at its configured inference resolution.
- `court_roi`: crop a padded image-space rectangle around the calibrated primary
  court, then infer at high resolution.
- `tiled`: cover the entire source frame with deterministic overlapping crops.
- `court_tiled`: tile only the padded calibrated primary-court ROI.

Court calibration is used only to choose an image crop. No ball coordinate is
transformed through homography. Tiled strategies translate crop-local boxes back to
source pixels and retain all proposals plus the IDs supporting each deduplicated box.

Run raw video inference:

```bash
WEIGHTS="../../ml/experiments/pickleball-yolo26s-v1/backend/ultralytics/weights/best.pt"

uv run pickleball-vision ball detect ../../sample-data/vid.mp4 \
  --config ../../ml/training/ball-detector.example.json \
  --weights "$WEIGHTS" \
  --strategy full-1920 \
  --output-dir ../../output/ball-detection/full-1920
```

For `court-roi-1920` or `court-tiled-1280`, also pass:

```bash
--calibration ../../output/calibration/calibration.json
```

The outputs are `detections.json`, `annotated.mp4`, and `summary.json`.
`detections.json` is explicitly frame-local raw evidence: its temporal tracking,
interpolation, and event flags are false.

## Evaluation metrics

Predictions are greedily matched in descending confidence order to unused ground
truth boxes at the configured IoU threshold.

- **Precision:** matched detections divided by all detections.
- **Recall:** matched annotations divided by all visible ground-truth annotations.
- **False positives:** unmatched detections.
- **False positives/minute:** false positives divided by the union duration, in
  minutes, of fixed source clip intervals represented by the partition.
- **Detection coverage:** positive frames with at least one matched detection divided
  by all frames containing one or more ground-truth balls.
- **Near/far performance:** object recall and positive-frame coverage restricted to
  human-annotated `near` or `far` objects. Unknown-side objects remain in overall
  metrics but not either side subtotal.

Run one strategy on the fixed validation set:

```bash
uv run pickleball-vision ball evaluate \
  --config ../../ml/training/ball-detector.example.json \
  --weights "$WEIGHTS" \
  --strategy tiled-1280 \
  --partition validation \
  --output-dir ../../ml/evaluation/pickleball-yolo26s-v1/tiled-1280
```

Use the test partition only for a final selected model/strategy audit, not iterative
tuning.

## Compare strategies on identical validation clips

For ROI strategies, add the dataset source ID and calibration path under
`evaluation.calibrations_by_source_id` in the configuration. Then run:

```bash
uv run pickleball-vision ball compare \
  --config ../../ml/training/ball-detector.example.json \
  --weights "$WEIGHTS" \
  --partition validation \
  --strategies full-1920 court-roi-1920 tiled-1280 court-tiled-1280 \
  --output-dir ../../ml/evaluation/pickleball-yolo26s-v1/strategy-comparison
```

Each strategy gets its own raw `detections.json` and `metrics.json`.
`comparison.json` records the common ordered frame IDs and refuses a comparison if
they differ. Compare recall and far-side coverage first, then precision and false
positives/minute; tiny-ball recall bought with excessive neighboring-court false
positives is not automatically a better operating point.
