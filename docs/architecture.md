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
        +-- official PyMongo Async API --> MongoDB Atlas
        |                                  structured records + small job queue
        |
        +-- hosted artifact adapter ----> Vercel Blob

Separate Python analysis worker
        |
        +-- claims/updates jobs --------> MongoDB Atlas
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
request. Heavy analysis also must not run in Vercel Functions. A separately deployed
Python worker owns long-running pipeline execution.

## Monorepo boundaries

| Area | Responsibility | Must not own |
| --- | --- | --- |
| `services/vision` | Runtime pipeline, typed schemas, local CLI | Training datasets, web UI |
| `ml` | Dataset curation, training, evaluation, experiments | Product APIs |
| `docs` | Stable contracts, definitions, labeling policy | Generated run artifacts |
| `sample-data` | Local, uncommitted test media | Private URLs or committed video |
| `output` | Local, generated results | Source-of-truth code or labels |

Planned application areas are documented now but must not be created before their
milestones become current:

| Planned area | Responsibility | Must not own |
| --- | --- | --- |
| React/Vite web application | Browser presentation and later human workflows | Hosted credentials, CV execution, domain truth |
| FastAPI product API | HTTP contracts, validation, compact records, artifact coordination, job submission/status | Heavy analysis inside requests |
| Python analysis worker | Claim jobs, stage media, invoke existing pipeline, persist results | Browser presentation or synchronous request handling |
| Hosted persistence adapters | PyMongo Async and Vercel Blob integration behind project interfaces | CV/audio domain algorithms |

Exact directories and deployment configuration belong to the corresponding product
milestones. Milestone 11 introduces no application scaffolding.

## Runtime responsibilities

### Browser application

The browser eventually presents match status, structured results, review media, and
human corrections. It calls FastAPI through documented HTTP contracts. It never
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
video analysis. The API returns a durable job identifier and exposes status/results
after the worker processes that job.

### Analysis worker

The analysis worker is a separate Python process and deployment unit. It retrieves
source media through an adapter, runs the existing CV/audio pipeline, uploads large
generated artifacts through an adapter, and writes compact status and result records.
It preserves the pipeline's raw-observation/derived-event boundaries and never turns
an infrastructure job state into match evidence.

The worker may call stable Python interfaces or the preserved CLI boundary. Hosted
concerns must not be spread through detector, calibration, tracking, audio, or
analytics modules.

### MongoDB Atlas

MongoDB Atlas stores hosted structured application data such as match metadata,
artifact manifests, compact summaries, job records, job leases, and references to
structured analysis outputs. Python application code uses the official PyMongo Async
API. Motor is not part of the architecture.

For the expected small scale, MongoDB may also implement the initial job queue using
atomic claims, explicit states, attempt metadata, and expiring worker leases. Redis
and Celery are intentionally absent unless measured throughput, latency, or delivery
requirements later demonstrate that MongoDB is insufficient.

Large source videos, annotated videos, extracted audio, model weights, datasets, and
large frame-level CV/audio artifacts do not belong directly in MongoDB documents.
MongoDB stores their metadata, provenance, status, and blob references.

### Vercel Blob

Vercel Blob stores hosted source video and binary/generated artifacts, including
review media, visualizations, and large structured artifacts when appropriate. Only
FastAPI and the analysis worker access Blob through project-owned interfaces and
adapters. Storage provider details and credentials cannot become CV/audio domain
dependencies or browser configuration.

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
inference for evaluation. It does not depend on the future API, worker, database, or
blob adapters.

## Data placement

| Data | System of record |
| --- | --- |
| Local source media and generated artifacts | Local filesystem, as today |
| Hosted source videos and large generated artifacts | Vercel Blob |
| Match metadata, job state, compact summaries, artifact manifests | MongoDB Atlas |
| Raw and derived pipeline contracts | Existing versioned artifact schemas, stored locally or by blob reference |
| Secrets | Server/worker environment or deployment secret store; never browser data |

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

During hosted milestones, FastAPI accepts short control-plane requests and the
separate worker performs long-running analysis. MongoDB job state connects those
processes without importing API or storage concerns into the vision domain. The
React/Vite application consumes API contracts only; it never calls the worker or
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

The React/Vite frontend will eventually deploy to Vercel. FastAPI and the Python
analysis worker require independent Python-capable runtime boundaries; heavy analysis
cannot execute in Vercel Functions or in the API request process. Their exact hosting
provider is intentionally deferred.

MongoDB Atlas and Vercel Blob are hosted adapters, not prerequisites for local use.
No application service, database integration, worker, web application, deployment,
or authentication feature is implemented during architecture lock-in.
