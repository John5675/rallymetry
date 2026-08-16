# AGENTS.md

These instructions apply to the entire repository. Read this file, `PLANS.md`,
and the relevant design document before changing code.

## Hard architectural rules

- Implement only the current milestone in `PLANS.md`.
- Prefer correctness and inspectability over premature optimization.
- Keep raw computer-vision detections separate from derived pickleball events.
- Never assume the four most confident person detections are the four players.
- Player court position must use an estimated ground-contact point, initially the
  bottom-center of the bounding box.
- Never use the center of a person bounding box as their physical court position.
- Court homography only applies to points lying on the court plane.
- Never project an airborne ball through the court homography as though it were
  touching the court.
- Ball observations must distinguish observed positions from interpolated
  positions.
- Never fabricate long missing ball trajectories.
- Every uncertain ML-derived event should retain confidence.
- LLMs must never invent match statistics.
- Analytics should operate on structured match data rather than raw model
  outputs.
- Never split individual neighboring video frames randomly across dataset
  partitions. Split by whole video, explicit clip, or rally/group.
- An unlabeled dataset frame is not a negative example.
- A visible neighboring-court pickleball remains a pickleball annotation with
  explicit scope; do not turn it into a visual negative.
- Detector experiments must retain dataset/model versions, fixed validation/test
  record IDs, configuration, metrics, and weights provenance.
- Near/far ball evaluation uses human annotation context; never infer an airborne
  ball's court side by projecting it through court homography.
- Model-suggested ball boxes remain predictions until a human explicitly reviews
  and accepts them; never silently promote suggestions to ground truth.
- Manual annotation work must be resumable and written atomically without changing
  source images or raw detector output.
- Audio is optional.
- The vision pipeline must work when audio is missing.
- Preserve audio/video synchronization.
- Keep raw audio observations separate from semantic pickleball events.
- Never infer a paddle contact or bounce solely from an audio transient.
- Neighboring courts may create unrelated pickleball sounds.
- Audio may increase or decrease confidence in a visually plausible event.
- Audio should not override contradictory visual evidence.
- Preserve original source media.
- Do not destructively modify source recordings.
- Preserve original channel information where practical.
- Record any resampling/channel conversion performed for analysis.
- Event fusion must support a configurable A/V timing offset and tolerance.
- All audio-dependent downstream stages must support a vision-only fallback.
- Do not introduce Spring Boot.
- Do not introduce PostgreSQL.
- Do not introduce Next.js.
- Use FastAPI for the product API.
- Use MongoDB Atlas for hosted structured data.
- Use the official PyMongo Async API rather than Motor.
- Use Vercel Blob for hosted binary and generated artifacts.
- Do not store large videos or frame-level CV artifacts directly in MongoDB.
- Keep heavy analysis outside the API process and outside HTTP request handling.
- Use a separate Python analysis worker for heavy CV and audio processing.
- MongoDB may act as the small-scale job queue.
- Do not add Redis or Celery unless demonstrated requirements justify them.
- Preserve the existing CLI and local pipeline.
- The local pipeline must continue working without MongoDB, Vercel, or internet
  connectivity when possible.
- Keep hosted integrations behind interfaces and adapters rather than embedding
  them throughout CV or audio code.
- Never expose MongoDB credentials or Vercel Blob credentials to the browser.
- Do not require user authentication until the dedicated web-access milestone.
- Prefer the simplest solution suitable for approximately six users.
- Videos, private YouTube URLs, secrets, large datasets, and trained model weights
  must not be committed.
- Add or update tests for behavior-changing work.
- Run tests, lint, and type-checking before completing tasks.

## Repository boundaries

- `services/vision` owns local runtime code and the CLI.
- `ml` owns dataset, training, evaluation, and experiment assets or code once the
  relevant milestone is current.
- `docs` owns stable contracts and definitions. Update the relevant document when
  a contract changes.
- `sample-data` and `output` contain local artifacts only. Keep their committed
  README files, but do not force-add ignored media or generated output.
- Do not add backend or web applications before their milestones become current.

## Engineering conventions

- Support Python 3.11+ and use type hints on public and internal interfaces.
- Keep imports side-effect free. Configure logging at the executable boundary.
- Raise typed application errors for expected failures; preserve the original
  exception as the cause when wrapping it.
- Use `pathlib.Path` for filesystem paths and explicit units in names or schemas.
- Store time as integer frame indices and/or seconds with an explicit time base.
- Keep detector output immutable or append-only. Derive tracks and events into
  separate schemas so provenance is retained.
- Keep experiments reproducible: record inputs, configuration, code revision,
  metrics, and artifacts without committing large generated files.

## Required verification

From `services/vision`, run:

```bash
uv sync --locked --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pickleball-vision doctor
```
