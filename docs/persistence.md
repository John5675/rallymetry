# Hosted persistence

## Scope

Milestone 19 adds optional storage adapters without changing the existing local
analysis pipeline. That milestone itself does not execute analysis jobs, claim queue
leases, download YouTube media, or expose cloud credentials to a browser. Milestone
20 consumes this boundary through the separate [FastAPI application API](api.md).

Hosted structured records use MongoDB Atlas through the official PyMongo Async API.
Binary and large generated artifacts use the project-owned `ArtifactStore` contract,
implemented by the local filesystem and Vercel Blob. Both adapters live under
`services/vision/src/pickleball_vision/persistence`; computer-vision and audio
algorithms do not import provider SDKs.

## Configuration

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `MONGODB_URL` | Only for MongoDB access | unset | Atlas or MongoDB connection string |
| `MONGODB_DATABASE` | No | `pickleball_vision` | Database name |
| `PICKLEBALL_VISION_ARTIFACT_BACKEND` | No | `local` | `local` or `vercel_blob` |
| `PICKLEBALL_VISION_LOCAL_ARTIFACT_ROOT` | No | `output/artifacts` | Local artifact root |
| `BLOB_READ_WRITE_TOKEN` | Only for Vercel Blob | unset | Server-side Blob credential |

The local backend needs no cloud service, credentials, or network. Diagnostic
configuration reports only whether credentials are configured; it never returns
their values. `.env.example` contains empty placeholders only. Application and
worker deployments must inject real secrets through their environment.

## MongoDB collections

Each record is match-scoped where applicable. Repeating records remain separate so
a match does not become one unbounded document.

| Collection | Contents | Principal indexes |
| --- | --- | --- |
| `matches` | Match metadata, optional `youtubeVideoId`, compact summary, versions, artifact references | unique sparse YouTube ID; update time |
| `players` | Logical match players and compact metadata | unique match/player |
| `rallies` | Structured rally records | unique match/record; match/time |
| `contacts` | Structured contact records | unique match/record; match/time |
| `bounces` | Structured bounce records | unique match/record; match/time |
| `shots` | Structured shot records | unique match/record; match/time |
| `analytics` | Deterministic metrics, calculation version, input references | unique match/analytics ID |
| `processing_jobs` | Job status and progress records | match/create time; status/update time |
| `corrections` | Versionable correction records for the later workflow | match/target |
| `artifacts` | Provider-neutral artifact manifests | unique pathname; match/time; match/category |

The processing-job record is persistence only. Atomic claiming, retries, leases,
and worker execution belong to Milestone 21.

Before a write, the adapter validates BSON-safe values, rejects byte arrays and
known inline binary fields, rejects unsafe MongoDB keys, rejects non-finite numbers,
and enforces a conservative 2 MiB application document ceiling. In particular,
MP4 data, raw frames, audio waveforms, weights, huge detection arrays, and debug
artifact bytes must be placed in an artifact store and referenced by ID.

## Artifact contract

`ArtifactStore` provides asynchronous `put`, `get`, `delete`, and `exists`
operations. A successful `put` returns an `ArtifactRecord` with:

- `artifactType`, category, provider, access, and randomized pathname
- URL when the provider supplies one
- content type and byte size
- UTC creation time and optional pipeline version
- SHA-256 checksum when available
- optional match association

Paths contain both a project-generated random token and, for Vercel Blob uploads,
a provider random suffix. Original filenames are sanitized for display only and are
not treated as access control.

| Category | Default hosted access | Public allowed? | Examples |
| --- | --- | --- | --- |
| `SOURCE_MEDIA` | `PRIVATE` | No | uploaded original/analysis input |
| `VIEWABLE_MEDIA` | `PRIVATE` | Yes, only when explicitly requested | annotated video, top-down video, heatmap, thumbnail |
| `INTERNAL_ARTIFACT` | `PRIVATE` | No | detections, tracks, audio observations, evaluation reports |

The local provider records `LOCAL` access and copies atomically under its configured
root. The Vercel provider streams uploads and downloads, keeps its token inside the
adapter, and never writes the token to artifact metadata. Callers should first store
the file, then save the returned artifact manifest in MongoDB and reference its
`artifactId` from compact match or result records.

Deleting a stored object and deleting its MongoDB manifest are deliberately separate
operations. A later application service must define authorization, retention, and
transaction/repair behavior rather than hiding a cross-provider partial failure.

## YouTube metadata

`MatchRecord.youtubeVideoId` stores only a video ID supplied by an application
caller. This milestone does not fetch YouTube URLs or media. The CV pipeline still
receives a local media file, whether that file originated from a camera, object
storage, or a separately authorized ingestion process.

## Testing

Persistence tests use an in-memory MongoDB-shaped test double, a Vercel Blob client
double, and temporary local files. They never contact Atlas or Vercel and never
require production credentials.
