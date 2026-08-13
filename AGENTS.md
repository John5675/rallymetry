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
