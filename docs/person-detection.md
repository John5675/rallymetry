# Person Detection Baseline

Milestone 3 detects all visible people broadly. A detection is source-image
evidence, not an assertion that a person is one of the four match players. People
on adjacent courts, spectators, and officials remain in `detections.json` and the
debug video.

## Model and coordinates

The default adapter uses the pretrained Ultralytics YOLO26 nano detection model
and requests only the COCO `person` class. Model tensors are translated immediately
into Pickleball Vision observation records. Bounding boxes use source-frame pixels:

- origin: top-left;
- x increases right and y increases down;
- `left_px`, `top_px`, `right_px`, `bottom_px` describe an `xyxy` rectangle; and
- frame numbers are zero-based, with timestamp `frame_number / fps`.

No box is projected through the court homography. In particular, the center of a
person box is not a physical court position. The downstream primary-player
isolation stage derives bottom-center ground estimates in a separate artifact;
this raw detector output remains unchanged.

The first model-backed run may download a `.pt` weight file. Weight files are
ignored by Git and must never be committed.

## Inference configuration

All values use the `PICKLEBALL_VISION_` prefix:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PERSON_MODEL` | `yolo26n.pt` | Ultralytics pretrained model name or local weight path |
| `PERSON_DEVICE` | `auto` | `auto`, `cpu`, `mps`, `cuda`, `cuda:N`, or a numeric CUDA index |
| `PERSON_MIN_CONFIDENCE` | `0.20` | Inclusive confidence threshold for retained people |
| `PERSON_IMAGE_SIZE` | `1280` | Square inference image size; source output coordinates are preserved |
| `PERSON_IOU_THRESHOLD` | `0.70` | Non-maximum-suppression overlap threshold |
| `PERSON_MAX_DETECTIONS` | `100` | Per-frame safety ceiling, not a participant count |

`auto` prefers CUDA, then Apple Metal Performance Shaders (MPS), then CPU. Force
`cpu` when comparing behavior across machines. A larger image size can help small
far-side players but costs substantial inference time and memory.

## Artifacts and review

`detections.json` contains the detector/configuration snapshot and all accepted
observations. `annotated.mp4` draws the same observations over every source frame.
`summary.json` reports processed-frame, detection-count, confidence, and runtime
statistics. These are detector diagnostics, never match statistics.

Review the annotated video at several rally moments. Check foreground players for
stable full-body boxes, then pause on the much smaller far-side players and inspect
whether each remains boxed through movement, overlap, and the far baseline. Also
look for false positives and all adjacent-court people: adjacent people should be
detected at this stage, not suppressed.

If far-side people are missed, compare a diagnostic run with a lower confidence
and/or larger inference image. Do not choose the setting by appearance alone;
eventually measure precision and recall against labeled frames.
