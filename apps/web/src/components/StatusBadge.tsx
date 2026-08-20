interface StatusBadgeProps {
  status: string;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const normalized = status.toUpperCase();
  const tone =
    normalized === "COMPLETE"
      ? "complete"
      : normalized === "FAILED"
        ? "failed"
        : normalized === "QUEUED" || normalized === "DRAFT" || normalized === "READY"
          ? "neutral"
          : "active";
  return <span className={`status status--${tone}`}>{normalized.replaceAll("_", " ")}</span>;
}
