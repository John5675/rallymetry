# 002 — Clarify processing progress

- **Status**: DONE
- **Commit**: 4fdc803
- **Severity**: MEDIUM
- **Category**: State indication and missed opportunities
- **Estimated scope**: 5 files, one small reusable component

## Problem

Long-running analysis is represented only by percentage text. Polling changes the
number abruptly, while job and submission notices appear without a visual bridge.

```tsx
// apps/web/src/pages/MatchPage.tsx:117 — current
{job !== null && job.status !== "COMPLETE" ? (
  <section className={`processing-banner processing-banner--${job.status === "FAILED" ? "failed" : "active"}`} aria-live="polite">
    <div>
      <strong>{job.stage?.replaceAll("_", " ") ?? job.status}</strong>
      <span>...</span>
    </div>
    <span>{Math.round(job.progress * 100)}%</span>
  </section>
) : null}

// apps/web/src/pages/MatchesPage.tsx:185 — current
<div className="submission-result--success">
  <CheckCircle2 aria-hidden="true" />
  <div>
    <strong>{submission.job.stage?.replaceAll("_", " ") ?? submission.job.status}</strong>
    <span>{Math.round(submission.job.progress * 100)}% - Analysis runs in the background.</span>
    <Link to={`/matches/${submission.match.matchId}`}>Open match</Link>
  </div>
</div>
```

## Target

Create an accessible `ProcessingProgress` component that clamps a `0..1` input,
exposes `role="progressbar"` with integer 0–100 ARIA values, and sets the fill's
inline `transform` directly:

```tsx
<span
  className="processing-progress__fill"
  style={{ transform: `scaleX(${clampedProgress})` }}
/>
```

The fill uses `transform 240ms linear`, originates at the left edge, and becomes
instant under reduced motion. Job and submission notices enter through CSS
`@starting-style` from `opacity: 0` and `translateY(8px)` to their settled state
using `200ms var(--ease-out)`. Reduced motion retains the opacity transition and
removes translation.

## Repo conventions to follow

- Components live in `apps/web/src/components` and use named exports.
- Component tests use Vitest and Testing Library next to the component or page.
- Shared styles live in `apps/web/src/styles.css`.
- The status is persistent and actionable, so keep it inline rather than adding a
  transient toast or Sonner dependency.

## Steps

1. Add `apps/web/src/components/ProcessingProgress.tsx` with clamping, integer ARIA
   values, a configurable accessible label, and direct fill transform.
2. Add `apps/web/src/components/ProcessingProgress.test.tsx` covering normal,
   below-zero, and above-one values.
3. Render the component beside the percentage in `MatchPage.tsx`.
4. Render the component in the successful submission result in `MatchesPage.tsx`.
5. Add progress-track, progress-fill, banner/result entry, and reduced-motion CSS
   to `apps/web/src/styles.css`.

## Boundaries

- Do NOT change polling cadence, job state semantics, API types, or backend code.
- Do NOT animate width, height, margin, padding, top, or left.
- Do NOT add Motion, NumberFlow, Sonner, or any other dependency.
- Do NOT turn error or processing status into ephemeral feedback.

## Verification

- **Mechanical**: from `apps/web`, run `npm run lint`, `npm run typecheck`,
  `npm test`, and `npm run build`; all must pass.
- **Feel check**: simulate two job progress updates and confirm the fill moves
  linearly without moving surrounding layout. Reload while processing and confirm
  the banner enters once, crisply, without bounce.
- In DevTools at 10% speed, confirm the banner's opacity and translation land
  together and the fill animates via `transform` only.
- Emulate `prefers-reduced-motion: reduce`; the banner may fade but must not move,
  and progress changes must be immediate.
- **Done when**: progress is readable visually and by assistive technology, mount
  transitions are under 300ms, and no layout property is animated.
