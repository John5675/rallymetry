# Hosted deployment

## Production shape

```text
Friends -> Vercel React/Vite -> persistent FastAPI -> MongoDB Atlas
                                      |
                                      +-> Render Workflows queue
                                             |
                                      on-demand analyze_match
                                             |
                                      MongoDB + Vercel Blob
                                             |
                                      existing website refreshes data
```

Vercel hosts only the static React/Vite application. FastAPI runs on a small
persistent Python host. Heavy analysis runs only in an on-demand Render Workflow
task, never in FastAPI, Vercel Functions, or an always-on polling process.

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
RENDER_API_KEY=...
RENDER_WORKFLOW_TASK=rallymetry-analysis/analyze_match
DEFAULT_ANALYSIS_PROFILE_MATCH_ID=match_...
```

FastAPI needs the Render key only to call the async task-start API. It does not need
model files and does not wait for task completion.

### Render Workflow

```text
MONGODB_URL=mongodb+srv://...
MONGODB_DATABASE=rallymetry
PICKLEBALL_VISION_ARTIFACT_BACKEND=vercel_blob
BLOB_READ_WRITE_TOKEN=...
PUBLIC_BLOB_READ_WRITE_TOKEN=...
PIPELINE_CONFIG=<absolute path to render-workflow-pipeline-plan.json>
MODEL_DEVICE=cpu
WORKFLOW_TEMP_DIR=/tmp/rallymetry
RENDER_WORKFLOW_PLAN=pro
RENDER_WORKFLOW_TIMEOUT_SECONDS=21600
YOUTUBE_MAX_DURATION_SECONDS=7200
YOUTUBE_MAX_BYTES=4000000000
```

The workflow does not need `RENDER_API_KEY`. Secrets belong in deployment secret
stores and must never be committed.

## MongoDB Atlas

1. Use a dedicated `rallymetry` database inside the Atlas cluster. Do not reuse or
   modify unrelated application collections.
2. Create a least-privilege database user for Rallymetry and store the connection
   string as `MONGODB_URL` on FastAPI and the workflow service.
3. Permit outbound connections from the selected hosts according to the Atlas
   network policy.
4. Start FastAPI once so `initialize_indexes()` creates Rallymetry's indexes,
   including the partial unique active-job index.
5. MongoDB stores compact match, job, rally, contact, bounce, shot, analytics, and
   artifact metadata. It does not store videos, weights, waveforms, frame dumps, or
   large raw detections, and it is not polled as a task queue.

## Vercel Blob

Use two stores:

- a private store for `SOURCE_MEDIA`, calibration/assignment/model setup, and any
  deliberately retained internal artifact;
- a public store only for friend-viewable `VIEWABLE_MEDIA`.

Set the private token as `BLOB_READ_WRITE_TOKEN` and public token as
`PUBLIC_BLOB_READ_WRITE_TOKEN` on the workflow. Never put either in `VITE_*`.
Artifact paths are randomized. Source recordings remain private. An unlisted YouTube
ID remains the normal browser playback source and may also be downloaded directly by
the on-demand Render task for one-click submissions. That copy is bounded, temporary,
and never made public.

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

## Render Workflow deployment

Render Workflows are configured separately in the Render Dashboard rather than as a
Background Worker:

1. Choose **New > Workflow** and connect this repository.
2. Root directory: `services/vision`.
3. Build command: `uv sync --locked`.
4. Start command: `uv run python -m pickleball_vision.workflows.app`.
5. Add the workflow environment above.
6. Confirm `analyze_match` registers with plan `pro`, timeout `21600`, and one retry.
7. Copy the task identifier `{workflow-slug}/analyze_match` into FastAPI's
   `RENDER_WORKFLOW_TASK`.
8. Put a least-privilege Render API key in FastAPI's `RENDER_API_KEY`, redeploy the
   API once, and verify the key is absent from frontend environment output.

The workflow queues work and provisions/deprovisions task compute. There is no
Rallymetry polling command to keep running. Full local and hosted workflow commands,
retry behavior, status semantics, and debugging are in
[`render-workflows.md`](render-workflows.md).

## Smoke test

1. `GET https://<fastapi-host>/health` returns `200` with database ready.
2. The Vercel match list loads through the configured API origin with no CORS error.
3. `DEFAULT_ANALYSIS_PROFILE_MATCH_ID` points to a match with the required setup.
4. Paste a YouTube link in the match library, or call
   `POST /api/matches/import-youtube`; verify `202` quickly with `jobId`,
   `processingRunId`, and `renderTaskRunId`.
5. A duplicate request returns that active job and creates no second Render run.
6. Render Dashboard shows one `analyze_match` run. `GET /api/jobs/<jobId>` advances
   through domain stages.
7. On completion, structured endpoints return rallies/shots/analytics and the
   artifact endpoint exposes only intended public viewable URLs.
8. Refresh the already deployed website. The completed match appears without a new
   Vercel deployment.
9. Confirm `/tmp/rallymetry/<jobId>` no longer exists after both success and a forced
   failure.

## Rollback and failure diagnosis

Disabling `RENDER_WORKFLOW_TASK`/`RENDER_API_KEY` makes `/process` return a controlled
`workflow_unavailable` error while read-only match pages remain available. A failed
task records `FAILED`, `failedStage`, `errorCode`, and a safe message in MongoDB.
Use `renderTaskRunId` to inspect full Render logs. Never copy provider credentials
into error fields or browser logs.
