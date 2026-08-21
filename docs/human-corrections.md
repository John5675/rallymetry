# Human correction workflow

Milestone 24 adds a semantic review layer around machine-derived player identity,
rally boundary, bounce, hitter, and shot-type records. It does not retrain a model or
rewrite any raw/derived prediction.

## Record lifecycle

Creation resolves a real match-scoped target and snapshots the target's prediction,
confidence, and best available model/pipeline version. A reviewer supplies a compact
`humanCorrection`, optional note and multimodal evidence references, and a verified
flag. Updates increment the revision and retain the superseded human value. Removal
marks the record inactive rather than deleting the correction or target.

Exactly one active correction exists per match, correction type, collection, and
target. This permits separate hitter and shot-type corrections for the same shot.
The correction collection is suitable for future evaluation/training export because
each example retains prediction provenance, target identity, reviewer result, and
timestamps. It must not be called human ground truth unless it is verified.

## Effective reads

- `payload` and base player fields are machine output.
- `effectivePayload` / `effectivePlayer` apply active verified human semantics.
- `verifiedCorrections` identifies the applied records.
- analytics reads preserve `predictionMetrics` and return correction-aware `metrics`.
- unverified and removed corrections never affect analytics.

## Supported values

- Player identity: corrected logical identity and/or display identity.
- Rally boundary: nonnegative start/end frames and/or media timestamps, with end not
  preceding start.
- Bounce: whether the candidate is a primary-match bounce, with optional corrected
  frame/timestamp/court position.
- Hitter: corrected logical player ID, including `UNKNOWN` when evidence is weak.
- Shot type: one of the initial documented shot classes, including `UNKNOWN`.

Audio and visual evidence fields contain only compact structured evidence or artifact
references. Audio never creates a bounce/contact by itself, and court coordinates
must already be geometrically justified structured values.
