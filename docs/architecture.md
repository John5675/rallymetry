# Architecture

## Purpose

Pickleball Vision will turn recorded doubles-match video into evidence-backed,
structured match data. The first product boundary is a local, inspectable pipeline.
Backend services, asynchronous orchestration, and a web dashboard are deliberately
deferred until the vision pipeline is useful and measurable.

## Monorepo boundaries

| Area | Responsibility | Must not own |
| --- | --- | --- |
| `services/vision` | Runtime pipeline, typed schemas, local CLI | Training datasets, web UI |
| `ml` | Dataset curation, training, evaluation, experiments | Product APIs |
| `docs` | Stable contracts, definitions, labeling policy | Generated run artifacts |
| `sample-data` | Local, uncommitted test media | Private URLs or committed video |
| `output` | Local, generated results | Source-of-truth code or labels |

## Data layers

The system should evolve around explicit, versioned records rather than an opaque
end-to-end result:

1. **Source metadata** describes video plus optional synchronized audio streams and
   their canonical source-media timeline without modifying the recording.
2. **Calibration** records image-space court landmarks and plane transforms.
3. **Annotation datasets** retain source hashes, frame/time provenance, human label
   groups, object annotations, and leakage-safe split units independently of model
   output.
4. **Observations** record model detections as produced, including confidence and
   provenance.
5. **Tracks** associate observations over time without rewriting their evidence.
6. **Events** infer pickleball meaning such as bounces or contacts, with confidence
   and links to supporting observations.
7. **Analytics** consume structured tracks and events, never raw detector tensors.
8. **Presentation** explains structured results and uncertainty; it does not
   manufacture facts.

Raw detections and derived events must remain separate even if a future optimized
implementation computes them in one process. Persistent identifiers belong to the
tracking layer, not to individual detections.

The person detector is accessed through a model-independent protocol. Its
Ultralytics implementation owns model loading and tensor translation; pipeline
orchestration owns video decoding and artifact persistence; observation dataclasses
own the stable JSON representation. This prevents a pretrained-model API from
becoming the repository's domain contract.

Primary-player isolation is a derived selection layer between raw observations and
persistent tracking. `player-candidates.json` references raw detections and adds
ground/court assessments plus ephemeral short-gap associations.
`player-assignments.json` separately records four human logical roles. Neither
artifact mutates detector evidence, and ephemeral candidate IDs must not become the
persistent logical identity.

Persistent player tracking has two further layers. `tracks.json` first records raw
ByteTrack observations linked to raw detection indices. A separate resolver uses
manual anchors, calibrated court membership/side, plausible motion, and temporal
continuity to produce per-frame logical states. Recording-local appearance evidence
from the manual anchors helps distinguish same-side players through tracker-ID
changes; it remains derived evidence rather than a biometric identity. `ME`,
`PARTNER`, `OPPONENT_1`, and `OPPONENT_2` therefore do not change merely because
ByteTrack creates a new ID. Questionable transitions are review events, not silently
trusted relabeling.

Player-position analytics is a further derived layer. `player_positions.json`
consumes only the structured logical identity layer, retains each raw bottom-center
image point and raw court projection, optionally adds an explicit recording-local
manual court-position correction, and then adds a separate bounded smoothing result.
`summary.json` consumes these structured position records rather than detector output.
Missing frames and suspected identity switches remain gaps in trajectories and
metrics.

Ball dataset tooling is an offline curation boundary. It reads local source media,
writes full-resolution frame images and optional lossless review clips, and records
content hashes plus frame/time provenance in `dataset-manifest.json`. Split manifests
reference those immutable records without copying or moving images. `positive`,
`negative`, and `unlabeled` are human curation states, never detector predictions.

The custom pickleball detector consumes a fixed split manifest plus a separate,
fully human-reviewed annotation manifest. Training materialization, Ultralytics
training, and model inference are separate boundaries. Each experiment snapshots
dataset/model versions, content hashes, configuration, code/runtime provenance,
weights hashes, and metrics. Model weights and generated datasets remain local.

Spatial inference may run a high-resolution full frame, a calibrated image-space
court crop, overlapping tiles, or tiled court crop. The model adapter sees only one
image or crop at a time; orchestration restores boxes to source pixels and retains
all crop proposals before cross-crop deduplication. Calibration is used to construct
an ROI, never to project an airborne ball onto the court plane. Raw ball detections
contain no temporal identity or semantic event.

Ball trajectory reconstruction is a derived image-space tracking layer. It links at
most one primary-match candidate per frame using motion, acceleration, persistence,
confidence, and an image-space court envelope while leaving ambiguous periods
unknown. `ball_tracks.json` references selected and rejected detector IDs, retains raw
observed points separately from interpolation and smoothing, and contains no semantic
pickleball events. Calibration projects only known court geometry into the image for
relevance; airborne ball points never pass through court homography.

The local annotation-review interface is an adapter over fixed split records,
human annotation JSON, source images, and optional raw detector suggestions. It binds
only to loopback and introduces no product backend. Suggestions stay in their raw
detection artifact; only explicit human actions write annotation records. Each save
is atomic, source images remain read-only, and a separate progress summary makes
long-running review resumable and inspectable.

## Execution model

During the local-pipeline milestones, the CLI is the executable boundary.
Configuration is loaded there, structured logging is initialized there, and typed
application errors are translated to stable exit codes there. Library imports must
not configure global logging or perform I/O.

Each future pipeline stage should have declared inputs, outputs, schema versions,
configuration, and provenance. Intermediate artifacts should be serializable so a
developer can inspect a failed stage without rerunning every preceding stage.

## Dependency direction

Domain schemas should depend only on small shared primitives. Detectors may emit
observation schemas; event derivation may consume observation and tracking schemas;
analytics may consume structured match data. Reverse dependencies are forbidden.
UI- or API-specific representations must adapt domain records instead of becoming
the domain model.

## Deferred architecture

Spring Boot, Next.js, queues, remote object storage, and hosted inference are not
Foundation concerns. Their boundaries will be designed during Backend
productization, Async processing, and Web dashboard milestones after local pipeline
contracts are supported by evidence.
