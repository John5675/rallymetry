# FastAPI application API

## Boundary

The FastAPI application is a small asynchronous control plane over the hosted
persistence layer. It creates and reads compact application records, lists existing
analysis results, and submits durable job status. It does not open videos, invoke a
CLI workflow, load a model, run analytics, upload artifacts, or execute a processing
job inside an HTTP request.

The ASGI application lives at `pickleball_vision.api.main:app`. Application routes
depend on `ApplicationPersistence`; the production lifespan supplies
`MongoPersistence`, while tests supply an isolated in-memory implementation.

## Configuration

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `MONGODB_URL` | For data endpoints | unset | MongoDB Atlas connection string |
| `MONGODB_DATABASE` | No | `pickleball_vision` | MongoDB database name |
| `CORS_ORIGINS` | No | `http://localhost:5173` | Comma-separated browser origins |
| `PICKLEBALL_VISION_ARTIFACT_BACKEND` | No | `local` | Artifact backend reported by health/config |
| `PICKLEBALL_VISION_LOCAL_ARTIFACT_ROOT` | No | `output/artifacts` | Local artifact root |
| `BLOB_READ_WRITE_TOKEN` | Not needed by FastAPI trigger | unset | Workflow-side Blob credential |
| `RENDER_API_KEY` | For `/process` | unset | Server-side Render task-start credential |
| `RENDER_WORKFLOW_TASK` | With Render key | unset | `{workflow-slug}/analyze_match` identifier |

`CORS_ORIGINS` values must be complete HTTP or HTTPS origins without paths, queries,
or fragments. A wildcard may be used only by itself. CORS credentials are disabled
because this milestone does not implement authentication.

If `MONGODB_URL` is missing or startup connectivity fails, the process remains live,
`GET /health` reports `degraded`, and data endpoints return a structured `503`. This
keeps liveness inspectable without pretending hosted persistence is ready.

## Run locally

From `services/vision`:

```bash
export MONGODB_URL='mongodb+srv://<user>:<password>@<cluster>/'
export MONGODB_DATABASE='pickleball_vision'
export CORS_ORIGINS='http://localhost:5173'
export RENDER_API_KEY='<server-side-key>'
export RENDER_WORKFLOW_TASK='rallymetry-analysis/analyze_match'

uv run uvicorn pickleball_vision.api.main:app \
  --host 127.0.0.1 \
  --port 8000
```

Then inspect `http://127.0.0.1:8000/docs` or request
`http://127.0.0.1:8000/health`. Never put real credentials in committed files or
browser code.

## Endpoints

| Method | Path | Result |
| --- | --- | --- |
| `GET` | `/health` | Process and persistence readiness |
| `POST` | `/api/matches` | Create compact match metadata (`201`) |
| `GET` | `/api/matches` | Paginated matches |
| `GET` | `/api/matches/{matchId}` | One match |
| `PATCH` | `/api/matches/{matchId}` | Update title, YouTube/source, or analysis setup references |
| `GET` | `/api/matches/{matchId}/players` | Logical players |
| `GET` | `/api/matches/{matchId}/rallies` | Paginated structured rallies |
| `GET` | `/api/matches/{matchId}/shots` | Paginated structured shots |
| `GET` | `/api/matches/{matchId}/contacts` | Paginated structured paddle contacts |
| `GET` | `/api/matches/{matchId}/bounces` | Paginated structured bounces |
| `GET` | `/api/matches/{matchId}/analytics` | Latest deterministic analytics record |
| `GET` | `/api/matches/{matchId}/artifacts` | Artifact manifests, not artifact bytes |
| `GET` | `/api/matches/{matchId}/corrections` | Active human correction ledger |
| `POST` | `/api/matches/{matchId}/corrections` | Create a correction with an immutable prediction snapshot (`201`) |
| `PATCH` | `/api/matches/{matchId}/corrections/{correctionId}` | Add a human correction revision |
| `DELETE` | `/api/matches/{matchId}/corrections/{correctionId}` | Soft-remove a correction without touching its prediction (`204`) |
| `POST` | `/api/matches/{matchId}/process` | Persist a queued analysis job (`202`) |
| `GET` | `/api/jobs/{jobId}` | Durable job status |

List endpoints use `limit` (default 50, maximum 100) and `offset` where the
collection can grow. API schemas expose application IDs such as `matchId`, `jobId`,
and `recordId`; MongoDB `_id`, BSON types, driver errors, and credentials are not part
of the response contract.

Player and structured-domain responses retain their machine fields and additionally
expose verified correction projections. `payload` is always the original prediction;
`effectivePayload` is the semantic value after active verified corrections, and
`verifiedCorrections` identifies the human records that produced it. Analytics
similarly returns correction-aware `metrics`, the untouched `predictionMetrics`, and
`appliedCorrectionIds`. Unverified corrections remain visible in the correction
ledger but do not affect effective reads or analytics.

## Human correction contract

Corrections support `PLAYER_IDENTITY`, `RALLY_BOUNDARY`, `BOUNCE`, `HITTER`, and
`SHOT_TYPE`. The server derives and validates the target collection; clients cannot
redirect a correction into an arbitrary collection. Creation snapshots
`prediction`, `predictionConfidence`, and `predictionVersion` from the target record.
Later edits may change only the human layer, verification/evidence, and review note.
Each update increments `revision` and retains the superseded human value in
`history`. Removal sets the correction inactive; it does not delete or alter the
machine record.

Optional `visualEvidence` and `audioEvidence` are compact structured references or
summaries only. Video, images, waveforms, and other binaries remain artifacts.

`youtubeVideoId`, when provided, is the exact 11-character YouTube video ID using
letters, numbers, `_`, or `-`. It is metadata for browser embedding; the API does
not download the video.

## Process submission

`POST /api/matches/{matchId}/process` verifies the match and its private hosted
source/setup artifacts, atomically creates at most one active
`ProcessingJobRecord` with:

```json
{
  "jobId": "job_<random-id>",
  "matchId": "match_<id>",
  "jobType": "analyze_match",
  "status": "QUEUED",
  "stage": "QUEUED",
  "progress": 0.0,
  "attemptCount": 0,
  "sourceType": "BLOB",
  "sourceArtifactId": "artifact_<id>",
  "processingRunId": "run_<id>",
  "renderTaskRunId": "trn-...",
  "resultArtifactIds": []
}
```

The API uses the official async Render SDK to start the configured `analyze_match`
task and stores its initial task-run ID; it does not await task completion. Missing
source/setup returns `409`, and a trigger failure records `FAILED` and returns a safe
`503`. A duplicate request returns the existing active job without a second Render
run. The response is `202 Accepted` with a `Location` header. The task writes domain
progress; `GET /api/jobs/{jobId}` exposes safe stage/timestamp/attempt/error fields.
See [`render-workflows.md`](render-workflows.md).

## Errors and request logging

Expected failures share one envelope:

```json
{
  "error": {
    "code": "resource_not_found",
    "message": "match was not found",
    "details": {"resource": "match", "id": "match_missing"},
    "requestId": "..."
  }
}
```

Each response includes `X-Request-ID`. A safe caller-supplied ID is preserved;
otherwise the API creates one. Request logs contain method, path, status, duration,
and request ID. They deliberately omit request bodies, query strings, connection
strings, and Blob tokens.
