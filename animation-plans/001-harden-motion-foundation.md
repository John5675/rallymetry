# 001 — Harden the motion foundation

- **Status**: DONE
- **Commit**: 4fdc803
- **Severity**: HIGH
- **Category**: Easing, physicality, performance, accessibility, and cohesion
- **Estimated scope**: 1 file, small CSS-only change

## Problem

The dashboard has no shared easing tokens, uses a positional nudge for press
feedback, animates hover movement on touch devices, scales timeline markers on
keyboard focus, paints skeleton background position continuously, and removes all
motion feedback for reduced-motion users.

```css
/* apps/web/src/styles.css:464 — current */
transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;

/* apps/web/src/styles.css:775 — current */
transition: transform 120ms ease, background-color 120ms ease;

/* apps/web/src/styles.css:778 — current */
.button:active {
  transform: translateY(1px);
}

/* apps/web/src/styles.css:1014 — current */
.timeline-marker:hover,
.timeline-marker:focus-visible {
  z-index: 2;
  transform: translateX(-50%) scale(1.6);
}

/* apps/web/src/styles.css:1783 — current */
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  *, *::before, *::after {
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
  }
}
```

## Target

Add these shared tokens to `:root`:

```css
--ease-out: cubic-bezier(0.23, 1, 0.32, 1);
--ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
```

Use `transform: scale(0.97)` for button presses with a `160ms` transition.
Move pointer hover transforms into
`@media (hover: hover) and (pointer: fine)`. Timeline keyboard focus keeps its
outline and stacking order without scaling. Reduce timeline hover scale to `1.3`.
Replace the skeleton background-position animation with an opacity-only pulse.
Remove global smooth scrolling so anchor and keyboard navigation remain immediate.
Replace the blanket reduced-motion override with targeted rules that remove
movement while preserving useful color and opacity feedback.

## Repo conventions to follow

- Motion tokens live beside the existing color, shape, and shadow tokens in
  `apps/web/src/styles.css:1`.
- Existing reduced-motion handling lives at the bottom of
  `apps/web/src/styles.css`.
- Do not add a motion dependency; the project already uses native CSS for all
  predetermined motion.

## Steps

1. Add `--ease-out` and `--ease-in-out` to `:root` in
   `apps/web/src/styles.css`.
2. Remove `scroll-behavior: smooth` from `html`.
3. Replace weak built-in curves on card, image, button, timeline-filter, and
   timeline-marker transitions with the shared tokens.
4. Change `.button:active` to `transform: scale(0.97)` and keep the press at
   `160ms`.
5. Gate match-card and timeline-marker hover transforms behind fine-pointer hover
   media queries. Keep keyboard focus visible but stationary.
6. Animate skeleton opacity only; keep the spinner's transform animation.
7. Replace the blanket reduced-motion duration reset with targeted movement
   removal and retain opacity/color transitions.

## Boundaries

- Do NOT add JavaScript or dependencies.
- Do NOT animate navigation, data metrics, charts, or court maps.
- Do NOT change layout, colors, typography, or domain behavior.
- If the cited selectors no longer exist, stop and report drift.

## Verification

- **Mechanical**: from `apps/web`, run `npm run lint`, `npm run typecheck`,
  `npm test`, and `npm run build`; all must pass.
- **Feel check**: press buttons and confirm the whole control compresses subtly;
  hover cards and timeline points with a mouse and confirm movement is crisp and
  restrained; keyboard-focus a timeline point and confirm it does not move.
- In DevTools, slow animations to 10% and confirm all hover/press transforms start
  immediately and settle without a built-in-ease tail.
- Emulate `prefers-reduced-motion: reduce` and confirm position movement is gone
  while focus, color, and opacity feedback remain.
- **Done when**: the motion review finds no ungated hover transform, built-in
  deliberate easing, keyboard-triggered transform, non-GPU skeleton motion, or
  blanket reduced-motion suppression.
