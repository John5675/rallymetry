interface ProcessingProgressProps {
  value: number;
  label?: string;
}

export function ProcessingProgress({
  value,
  label = "Analysis progress",
}: ProcessingProgressProps) {
  const finiteValue = Number.isFinite(value) ? value : 0;
  const clampedValue = Math.min(1, Math.max(0, finiteValue));
  const percentage = Math.round(clampedValue * 100);

  return (
    <div
      className="processing-progress"
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={percentage}
    >
      <span
        className="processing-progress__fill"
        style={{ transform: `scaleX(${clampedValue})` }}
        aria-hidden="true"
      />
    </div>
  );
}
