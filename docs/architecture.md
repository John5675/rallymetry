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

1. **Source metadata** describes a video and its time base without modifying it.
2. **Calibration** records image-space court landmarks and plane transforms.
3. **Observations** record model detections as produced, including confidence and
   provenance.
4. **Tracks** associate observations over time without rewriting their evidence.
5. **Events** infer pickleball meaning such as bounces or contacts, with confidence
   and links to supporting observations.
6. **Analytics** consume structured tracks and events, never raw detector tensors.
7. **Presentation** explains structured results and uncertainty; it does not
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
persistent track IDs introduced by the next milestone.

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
