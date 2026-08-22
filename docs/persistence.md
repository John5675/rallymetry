# Hosted persistence

## Scope

Milestone 19 adds optional storage adapters without changing the existing local
analysis pipeline. That milestone itself does not execute analysis jobs, download
YouTube media, or expose cloud credentials to a browser. Milestone
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
| `BLOB_READ_WRITE_TOKEN` | Only for Vercel Blob | unset | Private-store server credential |
| `PUBLIC_BLOB_READ_WRITE_TOKEN` | Only for public viewable delivery | unset | Separate public-store server credential |

The local backend needs no cloud service, credentials, or network. Diagnostic
configuration reports only whether credentials are configured; it never returns
their values. `.env.example` contains empty placeholders only. Application and
workflow deployments must inject real secrets through their environment.

### Provision Vercel Blob stores

Link the repository to a dedicated Vercel project without deploying it, create a
private store in the region nearest the analysis workflow, and pull the development
environment into the ignored `.env.local` file:

```bash
npx --yes vercel@latest login
npx --yes vercel@latest link
npx --yes vercel@latest blob create-store pickleball-vision-private \
  --access private \
  --region sfo1 \
  --yes
npx --yes vercel@latest env add PICKLEBALL_VISION_ARTIFACT_BACKEND \
  production,preview,development \
  --value vercel_blob \
  --yes \
  --no-sensitive
npx --yes vercel@latest env pull .env.local --yes
```

Vercel creates and connects `BLOB_READ_WRITE_TOKEN`; never copy its value into a
tracked file. The Python settings layer intentionally does not load dotenv files,
so local workflow development must export the pulled values before it starts:

```bash
cd services/vision
set -a
source ../../.env.local
set +a
export PIPELINE_CONFIG="$PWD/../../docs/examples/render-workflow-pipeline-plan.json"
render workflows dev -- uv run python -m pickleball_vision.workflows.app
```

Friend-viewable deployment uses a second public Blob store whose token is injected
as `PUBLIC_BLOB_READ_WRITE_TOKEN`. Blob access is store-wide and cannot be changed
per object, so the routed adapter sends explicit public `VIEWABLE_MEDIA` only to
that store. Private viewable drafts, `SOURCE_MEDIA`, and `INTERNAL_ARTIFACT` remain
on the private store. Connect the public store with the `PUBLIC_BLOB_` environment
prefix; see [`deployment.md`](deployment.md#vercel-blob).

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
| `processing_jobs` | Worker ownership/lease, status, progress, attempts, optional Render/run IDs, errors, source/result references | match/create time; status/update time; worker claim; unique active match |
| `corrections` | Immutable prediction snapshots plus revisioned human semantic corrections | unique active match/type/target; match/corrected time |
| `artifacts` | Provider-neutral artifact manifests | unique pathname; match/time; match/category |

In `mongodb_worker` mode, MongoDB coordinates the intentionally small
single-concurrency queue. Atomic claim updates retain `workerId`, `claimedAt`,
`heartbeatAt`, and `leaseExpiresAt`; bounded attempts recover a crashed worker without
allowing two live workers to own the same job. A partial unique active-match index
prevents accidental duplicate submissions. Job documents retain only compact
status/provenance and artifact references; outputs use separate collections or
artifact storage. Render Workflow execution remains an optional adapter.

Before a write, the adapter validates BSON-safe values, rejects byte arrays and
known inline binary fields, rejects unsafe MongoDB keys, rejects non-finite numbers,
and enforces a conservative 2 MiB application document ceiling. In particular,
MP4 data, raw frames, audio waveforms, weights, huge detection arrays, and debug
artifact bytes must be placed in an artifact store and referenced by ID.

### Correction integrity

One active correction may exist for a match, correction type, collection, and target
record. The document stores the original `prediction`, its confidence/version, the
current `humanCorrection`, reviewer verification, optional multimodal evidence,
revision history, and correction/removal timestamps. Persistence rejects any update
that changes the target or prediction snapshot. API removal is a soft removal so the
review history remains useful for audit and future evaluation/training export.

Corrections never modify `players`, `rallies`, `bounces`, or `shots`. Consumers build
an effective semantic view from active verified corrections while the original
machine collections remain byte-for-byte available.

## Artifact contract

`ArtifactStore` provides asynchronous `put`, `get`, `delete`, and `exists`
operations. A successful `put` returns an `ArtifactRecord` with:

- `artifactType`, category, provider, access, and randomized pathname
- URL when the provider supplies one
- content type and byte size
- UTC creation time and optional pipeline version
- optional stable `processingRunId` for idempotent workflow publication
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
root. The Vercel provider streams uploads and downloads, keeps both tokens inside a
routed adapter, and never writes either token to artifact metadata. Callers should first store
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
