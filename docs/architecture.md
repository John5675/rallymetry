# Architecture

## Purpose

Pickleball Vision will turn recorded doubles-match video into evidence-backed,
structured match data. The first product boundary is a local, inspectable pipeline.
That pipeline remains the analysis source of truth and must continue to work locally
without hosted services when possible. The remaining application architecture is
now fixed, although its product services are deferred to their roadmap milestones.

## Locked application stack

```text
React + Vite + TypeScript (browser; eventually deployed to Vercel)
        |
        | HTTPS application requests
        v
FastAPI product API
        |
        +-- official PyMongo Async API --> MongoDB Atlas (records + job status)
        +-- Render async client --------> Render Workflows queue

On-demand Render Workflow task
        |
        +-- updates domain status ------> MongoDB Atlas
        +-- reads/writes binaries ------> Vercel Blob
        +-- invokes --------------------> existing CV + audio pipeline
```

The frontend is React with Vite and TypeScript, not Next.js. The product API is
FastAPI, not Spring Boot. Hosted structured data uses MongoDB Atlas through the
official PyMongo Async API, not PostgreSQL or Motor. Hosted videos and generated
binary or large frame-level artifacts use Vercel Blob.

FastAPI request handling is a control-plane boundary. It may validate requests,
read or update compact structured records, issue safe artifact operations, and
enqueue analysis work. It must not run the heavy CV/audio pipeline inside an HTTP
request. Heavy analysis also must not run in Vercel Functions. Render provisions a
temporary task instance only when FastAPI starts `analyze_match`, then deprovisions it.

## Monorepo boundaries

| Area | Responsibility | Must not own |
| --- | --- | --- |
| `services/vision` | Runtime pipeline, typed schemas, local CLI | Training datasets, web UI |
| `ml` | Dataset curation, training, evaluation, experiments | Product APIs |
| `docs` | Stable contracts, definitions, labeling policy | Generated run artifacts |
| `sample-data` | Local, uncommitted test media | Private URLs or committed video |
| `output` | Local, generated results | Source-of-truth code or labels |

Application areas are introduced only as their milestones become current:

| Area | Responsibility | Must not own |
| --- | --- | --- |
| `apps/web` React/Vite application | Browser presentation and later human workflows | Hosted credentials, CV execution, domain truth |
| `services/vision/src/pickleball_vision/api` | FastAPI HTTP contracts, validation, compact records, artifact manifests, job submission/status | Heavy analysis inside requests |
| Render Workflow task | Stage media/setup, invoke existing pipeline, persist results, clean scratch space | Polling MongoDB, browser presentation, or synchronous request handling |
| Hosted persistence adapters | PyMongo Async and Vercel Blob integration behind project interfaces | CV/audio domain algorithms |

Milestone 19 implements the provider-neutral persistence records and optional
MongoDB/Vercel Blob adapters; see [`persistence.md`](persistence.md). Milestone 20
adds the FastAPI control plane under the vision service; see [`api.md`](api.md).
Milestone 21 adds on-demand Render execution without moving analysis into HTTP; see
[`render-workflows.md`](render-workflows.md).
Milestone 22 adds the strict TypeScript dashboard under `apps/web`; see
[`web.md`](web.md). Milestone 23 adds the Vercel SPA deployment contract, persistent
FastAPI container boundary, and dual-access Blob delivery; see
[`deployment.md`](deployment.md).

## Runtime responsibilities

### Browser application

The browser presents match status, structured results, public review media, and
deterministic analytics. Later milestones add human corrections. It calls FastAPI
through documented HTTP contracts and never
receives MongoDB credentials or Vercel Blob credentials. If a future direct upload
flow is needed, FastAPI may issue a short-lived, narrowly scoped operation; permanent
service credentials remain server-side.

User authentication is intentionally deferred until a dedicated web-access
milestone. Earlier local or hosted prototypes must not create an accidental auth
system. The design should remain proportionate to approximately six users.

### FastAPI product API

FastAPI is the product control plane. Its responsibilities are bounded request
validation, structured application reads/writes, hosted-artifact coordination, job
creation, and job-status reporting. Request completion cannot depend on finishing a
video analysis. It creates an application job, asks Render to start the configured
task, stores the returned task-run ID, and exposes status/results as the task updates them.

### On-demand analysis workflow

`analyze_match` is a Render Workflow task and separate execution unit. It retrieves
private source/setup artifacts or one explicitly submitted YouTube recording, runs the existing CV/audio pipeline, uploads an
explicit allow-list of generated media, writes compact status/results, and cleans its
job-scoped `/tmp/rallymetry/<job-id>` directory. It preserves raw-observation/
derived-event boundaries and never turns infrastructure state into match evidence.

The workflow may call stable Python interfaces or the preserved CLI boundary. Hosted
concerns must not be spread through detector, calibration, tracking, audio, or
analytics modules. The human-correction layer is append/revision oriented: machine
records remain unchanged, MongoDB stores a separate target-specific correction, and
API projections resolve only active verified corrections. This gives the dashboard
an effective semantic view while retaining predictions for audit and evaluation.

### MongoDB Atlas

MongoDB Atlas stores hosted structured application data such as match metadata,
artifact manifests, compact summaries, job records, and references to
structured analysis outputs. Python application code uses the official PyMongo Async
API. Motor is not part of the architecture.

MongoDB is not a task queue. Render Workflows owns queuing and execution. The
`processing_jobs` collection is Rallymetry's durable source of truth for domain stage,
progress, results, and errors. A partial unique index prevents two active application
jobs for the same match; no process polls job documents.

Large source videos, annotated videos, extracted audio, model weights, datasets, and
large frame-level CV/audio artifacts do not belong directly in MongoDB documents.
MongoDB stores their metadata, provenance, status, and blob references.

### Vercel Blob

Vercel Blob stores hosted source video and binary/generated artifacts. A private
store owns source/internal objects; a separate public store owns only deliberately
public `VIEWABLE_MEDIA` such as friend-viewable review videos and heatmaps. Only
FastAPI and the Render Workflow receive only the credentials needed by their
project-owned adapters. The browser receives public object URLs, never provider
credentials. Storage details cannot become CV/audio domain dependencies.

### Local pipeline

The existing `pickleball-vision` CLI and local filesystem artifacts remain supported.
Local ingestion, calibration, detection, tracking, audio analysis, and future local
pipeline stages should continue working without MongoDB Atlas, Vercel Blob, or an
internet connection when their model assets are already present. Hosted execution
adapts this pipeline; it does not replace or fork it.

The Milestone 12 annotation editor is also a local-pipeline tool. Its loopback-only
HTTP server and native browser media controls are a maintainable local UI adapter,
not the FastAPI product API or React dashboard. It writes versioned human ground
truth directly to the local artifact boundary and introduces no hosted dependency.

Milestone 13 rally segmentation is another local derived-event stage. It consumes
versioned trajectory and optional compatible player/audio evidence, writes rally
intervals separately from every raw input, and loads human annotations only after
inference for evaluation. It does not depend on the hosted API, workflow, database, or
blob adapters.

## Data placement

| Data | System of record |
| --- | --- |
| Local source media and generated artifacts | Local filesystem, as today |
| Hosted source videos and large generated artifacts | Vercel Blob |
| Match metadata, job state, compact summaries, artifact manifests | MongoDB Atlas |
| Raw and derived pipeline contracts | Existing versioned artifact schemas, stored locally or by blob reference |
| Secrets | API/workflow environment or deployment secret store; never browser data |

MongoDB documents should reference immutable or versioned artifact identities and
retain provenance. Moving an artifact from the local filesystem to hosted blob
storage must not collapse raw observations into events or make presentation records
the analytical source of truth.

## Data layers

The system should evolve around explicit, versioned records rather than an opaque
end-to-end result:

1. **Source metadata** describes video plus optional synchronized audio streams and
   their canonical source-media timeline without modifying the recording.
2. **Calibration** records image-space court landmarks and plane transforms.
3. **Annotation datasets** retain source hashes, frame/time provenance, human label
   groups, object annotations, and leakage-safe split units independently of model
   output.
4. **Observations** record model detections and raw audio signal features as
   produced, including confidence and provenance.
5. **Tracks** associate observations over time without rewriting their evidence.
6. **Events** infer pickleball meaning such as bounces or contacts, with confidence
   and links to supporting observations.
7. **Analytics** consume structured tracks and events, never raw detector tensors.
8. **Presentation** explains structured results and uncertainty; it does not
   manufacture facts.

Raw detections and derived events must remain separate even if a future optimized
implementation computes them in one process. Persistent identifiers belong to the
tracking layer, not to individual detections.

Audio analysis is a parallel observation path. It retains synchronized PCM and
channel-aware signal windows separately from derived generic transient candidates.
Those candidates are still non-semantic evidence: they do not assert a paddle
contact, bounce, rally, or primary-court source. Audio-free inputs create an explicit
unavailable result so every downstream audio consumer can select a vision-only
fallback.

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

Deterministic match analytics is the terminal local derived-data layer.
`match-analytics.json` validates and summarizes only structured rally, shot, and
player-position artifacts. Contact and bounce relationships are consumed only as
references already retained by a structured shot. It never reaches backward into
raw model tensors, YOLO records, audio waveforms, or detector adapters. Input hashes,
explicit denominators, `UNKNOWN` exclusions, coverage, and confidence limitations
keep every statistic inspectable; this artifact is suitable for future product
adapters without making those adapters part of the analytical source of truth.

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

Automatic rally segmentation is a derived event layer over the structured ball
timeline. It uses explicit ball activity, sustained image-space motion, bounded and
long gaps, serve-like motion onsets, burst spacing, and optional source-compatible
player reset evidence. Generic audio transients are optional confidence support and
cannot create a boundary. Adjacent bursts are compared using explicit motion,
coverage, persistence, and duration evidence so a weaker possible dead-ball handoff
can be excluded without being semantically classified or discarded.
`rallies.json` retains accepted and rejected candidates, supporting signals, and provenance;
it never mutates `ball_tracks.json`, `tracks.json`, or `audio-events.json`. Human
annotations are a post-inference evaluation input only, with sparse reviewed time
kept distinct from reviewed-negative complete-video coverage.

Multimodal bounce detection is a separate derived-event layer over the immutable
ball trajectory. It generates candidates only from visual reversal, motion, shape,
and continuity evidence. Optional rally intervals and generic audio transients are
confidence support; neither can create a candidate. Audio fusion remaps raw analysis
timestamps with explicit A/V offset and tolerance provenance. Court homography is
applied to a candidate ball point only after visual plane-contact plausibility and
projected-court image inclusion. Outputs retain low-confidence candidates and never
infer true 3D position or line calls.

Multimodal paddle-contact detection is the next separate event layer. It requires
visual ball-trajectory velocity/direction discontinuity before logical-player,
rally, bounce-state, or audio context can affect confidence. Logical-player boxes
support proximity ranking while bottom-center ground points remain the only physical
court-position estimate. Candidate-player rankings are not hitter assignments;
`assignedHitter` remains null. Generic audio cannot create a contact, and airborne
ball points are never projected through court homography. Human contact annotations
are isolated to post-inference temporal evaluation.

Hitter identification consumes that immutable candidate layer plus the exact
logical-player track artifact recorded in its provenance. It emits a separate
logical role or `UNKNOWN`, confidence, alternatives, and supporting visual/player
signals. Previous-hitter and rally-order context are bounded and reset after an
uncertain eligible contact. Audio makes no identity contribution, and hitter
resolution never projects the airborne ball through court homography. Human player
labels remain isolated to post-inference evaluation.

Shot reconstruction consumes those exact compatible artifacts and emits a separate
rally-local shot layer. It references rather than rewrites the frame-complete ball
trajectory, copies a landing court point only from an accepted visually plane-gated
bounce, and copies hitter location only from a bottom-center player ground point.
Its initial classifier is an ordered, configured domain rule set with a fixed small
vocabulary and explicit `UNKNOWN`; no new neural network is introduced. Human shot
labels remain post-inference evaluation inputs. See `shot-reconstruction.md`.

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

During hosted execution, FastAPI accepts short control-plane requests and starts one
Render task. Render Workflows queues and runs long analysis; MongoDB records domain
status without acting as that queue. The React/Vite application consumes API contracts
only; it never calls the workflow or
database directly.

Each future pipeline stage should have declared inputs, outputs, schema versions,
configuration, and provenance. Intermediate artifacts should be serializable so a
developer can inspect a failed stage without rerunning every preceding stage.

## Dependency direction

Domain schemas should depend only on small shared primitives. Detectors may emit
observation schemas; event derivation may consume observation and tracking schemas;
analytics may consume structured match data. Reverse dependencies are forbidden.
UI- or API-specific representations must adapt domain records instead of becoming
the domain model.

## Deployment boundary

The React/Vite frontend deploys to Vercel. FastAPI runs in the vendor-neutral
container under `services/vision` on a persistent Python-capable host. Analysis runs
on temporary on-demand Render Workflow compute.
Heavy analysis cannot execute in Vercel Functions or in the API request process.

MongoDB Atlas and Vercel Blob remain optional hosted adapters, not prerequisites for
local pipeline use. Milestone 20 provides the FastAPI control plane and creates
durable job records. Milestone 21 provides async Render triggering, duplicate-submit
protection, private source/setup staging, idempotent publication, progress updates,
and cleanup. Milestone 22 provides the browser, and Milestone
23 provides hosted delivery. Milestone 23A adds the bounded YouTube-link submission
UI while keeping retrieval in Render. Authentication and corrections remain later work.
