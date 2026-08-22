# Hosted deployment

## Production shape

```text
Friends -> Vercel React/Vite -> persistent FastAPI -> MongoDB Atlas
                                      |
                                      +-> QUEUED processing job

Developer Mac -> single local worker -> authorized source retrieval + CV/audio
                                      -> MongoDB + Vercel Blob
                                      -> existing website refreshes data
```

Vercel hosts only the static React/Vite application. FastAPI runs on a small
persistent Python host. Heavy analysis runs on the developer's Mac through a
single-concurrency MongoDB worker, never in FastAPI, Vercel Functions, or an HTTP
request. This avoids cloud YouTube egress blocks and requires no inbound connection
to the Mac. Jobs remain queued while the Mac is offline.

## Environment boundaries

### Frontend

```text
VITE_API_BASE_URL=https://<fastapi-host>
```

No MongoDB, Blob, or Render credential is browser-visible.

### FastAPI

```text
MONGODB_URL=mongodb+srv://...
MONGODB_DATABASE=rallymetry
CORS_ORIGINS=http://localhost:5173,https://rallymetry.vercel.app
ANALYSIS_EXECUTION_MODE=mongodb_worker
DEFAULT_ANALYSIS_PROFILE_MATCH_ID=match_...
```

FastAPI creates the durable job but does not need model files, Blob credentials, or
Render workflow credentials and does not wait for task completion.

### Developer-machine worker

```text
MONGODB_URL=mongodb+srv://...
MONGODB_DATABASE=rallymetry
PICKLEBALL_VISION_ARTIFACT_BACKEND=vercel_blob
BLOB_READ_WRITE_TOKEN=...
PUBLIC_BLOB_READ_WRITE_TOKEN=...
PIPELINE_CONFIG=<absolute path to render-workflow-pipeline-plan.json>
MODEL_DEVICE=cpu
WORKFLOW_TEMP_DIR=/tmp/rallymetry
YOUTUBE_MAX_DURATION_SECONDS=7200
YOUTUBE_MAX_BYTES=4000000000
WORKER_ID=johns-mac
WORKER_POLL_SECONDS=10
WORKER_HEARTBEAT_SECONDS=30
WORKER_LEASE_SECONDS=180
WORKER_MAX_ATTEMPTS=2
```

The worker does not need `RENDER_API_KEY`, a residential proxy, or inbound network
access. It needs outbound access to YouTube, MongoDB Atlas, and Vercel Blob. Only
process recordings the submitter owns or is authorized to analyze. Secrets live in
the ignored `.env.worker` file with mode `0600` and must never be committed.

## MongoDB Atlas

1. Use a dedicated `rallymetry` database inside the Atlas cluster. Do not reuse or
   modify unrelated application collections.
2. Create a least-privilege database user for Rallymetry and store the connection
   string as `MONGODB_URL` on FastAPI and the local worker.
3. Permit outbound connections from the selected hosts according to the Atlas
   network policy.
4. Start FastAPI once so `initialize_indexes()` creates Rallymetry's indexes,
   including the partial unique active-job index.
5. MongoDB stores compact match, job, rally, contact, bounce, shot, analytics, and
   artifact metadata. It coordinates this small single-worker queue but does not
   store videos, weights, waveforms, frame dumps, or large raw detections.

## Vercel Blob

Use two stores:

- a private store for `SOURCE_MEDIA`, calibration/assignment/model setup, and any
  deliberately retained internal artifact;
- a public store only for friend-viewable `VIEWABLE_MEDIA`.

Set the private token as `BLOB_READ_WRITE_TOKEN` and public token as
`PUBLIC_BLOB_READ_WRITE_TOKEN` on the worker. Never put either in `VITE_*`.
Artifact paths are randomized. Source recordings remain private. An unlisted YouTube
ID remains the normal browser playback source and may also be retrieved temporarily
by the local worker for an authorized one-click submission. That copy is bounded,
cleaned after processing, and never made public.

Before analysis, each match must reference either one private source artifact or one
validated YouTube video ID, plus four private setup artifacts in `analysisSetup`; see
[`render-workflows.md`](render-workflows.md#match-prerequisites-and-one-click-profile).

## FastAPI deployment

The production image is [services/vision/Dockerfile](../services/vision/Dockerfile).
Build it from `services/vision`:

```bash
docker build -t rallymetry-api .
docker run --rm -p 8000:8000 --env-file .env rallymetry-api
```

The container startup command is:

```text
uvicorn pickleball_vision.api.main:app --host 0.0.0.0 --port $PORT
```

Configure the host health check as `GET /health`. A response can be `degraded` when
MongoDB is unavailable, which is useful diagnostically but should fail deployment
readiness policy if hosted data is required. FastAPI remains a persistent web
service; do not convert it to a Vercel Function.

## Vercel frontend deployment

The root `vercel.json` owns the monorepo SPA contract:

- install command: `npm --prefix apps/web ci`
- build command: `npm --prefix apps/web run build`
- output directory: `apps/web/dist`
- rewrite non-file routes to `/index.html`

Set `VITE_API_BASE_URL` to the public FastAPI origin and deploy. Add the exact Vercel
domain (and optional custom domain) to FastAPI's comma-separated `CORS_ORIGINS`.

## macOS worker installation

From the repository root:

```bash
cp .env.worker.example .env.worker
chmod 600 .env.worker
```

Fill `MONGODB_URL` and, if they are not already supplied by the ignored Vercel CLI
`.env.local`, the two Blob tokens. Validate one queue poll in the foreground:

```bash
./scripts/run-local-worker.sh --once
```

Install the persistent per-user LaunchAgent:

```bash
./scripts/install-macos-worker.sh
launchctl print gui/$UID/com.rallymetry.worker
tail -f ~/Library/Logs/rallymetry-worker.log
```

The installer creates an isolated `uv tool` and copies the pipeline plan plus the
ignored worker environment into `~/Library/Application Support/Rallymetry`. This is
intentional: macOS background agents cannot reliably read repositories located under
the protected `Documents` directory.

The worker atomically claims one job, heartbeats while processing, and permits a
bounded stale-lease recovery. Direct private MP4 source artifacts remain the
fallback when an authorized YouTube recording cannot be retrieved. Render Workflow
support remains available as the explicit `render_workflow` execution mode, but it
is not the active no-proxy deployment path.

FastAPI translates each persisted worker stage/progress pair into a stable current
step key, friendly description, and analysis step index. The dashboard polls this
contract every five seconds, shows the most recent worker heartbeat, and never sends
raw worker logs, credentials, local paths, or command arguments to the browser.

## Smoke test

1. `GET https://<fastapi-host>/health` returns `200` with database ready.
2. The Vercel match list loads through the configured API origin with no CORS error.
3. `DEFAULT_ANALYSIS_PROFILE_MATCH_ID` points to a match with the required setup.
4. Paste a YouTube link in the match library, or call
   `POST /api/matches/import-youtube`; verify `202` quickly with `jobId`,
   `processingRunId`, `status=QUEUED`, and no `renderTaskRunId`.
5. A duplicate request returns that active job and creates no second queue entry.
6. The Mac worker log shows one atomic claim. `GET /api/jobs/<jobId>` advances
   through domain stages and exposes heartbeat/lease timestamps.
7. On completion, structured endpoints return rallies/shots/analytics and the
   artifact endpoint exposes only intended public viewable URLs.
8. Refresh the already deployed website. The completed match appears without a new
   Vercel deployment.
9. Confirm `output/worker-tmp/<jobId>` no longer exists after both success and a
   forced failure.

The production plan also runs the bundled eight-video AI visual-review overlay
after shot reconstruction. The overlay is source-hash gated, never changes machine
predictions, and is published only as additional shot evidence. Analytics remains
connected to the original `shots/shots.json`; the dashboard reads
`shots/reviewed-shots.json` so a matching reviewed source can show the separate AI
visual-review best guess. Unrelated uploads cannot inherit those labels.

## Rollback and failure diagnosis

Set `ANALYSIS_EXECUTION_MODE=mongodb_worker` for the local-worker path. A stopped Mac
leaves jobs queued; an interrupted claimed job becomes reclaimable after its lease
expires. Bounded failures retain `FAILED`, `failedStage`, `errorCode`, and a safe
message in MongoDB. Inspect local logs without copying credentials into error fields
or browser logs.
