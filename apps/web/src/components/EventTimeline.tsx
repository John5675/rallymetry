import { AudioLines, Circle, Crosshair, Goal } from "lucide-react";
import { useMemo, useState } from "react";

import type { TimelineEvent } from "../domain";
import { formatConfidence, formatDuration } from "../domain";

interface EventTimelineProps {
  events: TimelineEvent[];
  durationSeconds: number | null;
  onSeek: (timestampSeconds: number) => void;
}

const KIND_LABELS = {
  rally: "Rally boundary",
  contact: "Contact",
  bounce: "Bounce",
  shot: "Shot",
} as const;

function MarkerIcon({ kind }: { kind: TimelineEvent["kind"] }) {
  if (kind === "rally") return <Goal aria-hidden="true" />;
  if (kind === "contact") return <AudioLines aria-hidden="true" />;
  if (kind === "bounce") return <Circle aria-hidden="true" />;
  return <Crosshair aria-hidden="true" />;
}

export function EventTimeline({ events, durationSeconds, onSeek }: EventTimelineProps) {
  const [selectedKinds, setSelectedKinds] = useState<Set<TimelineEvent["kind"]>>(
    () => new Set(["rally", "contact", "bounce", "shot"]),
  );
  const duration = Math.max(
    durationSeconds ?? 0,
    events.at(-1)?.timestampSeconds ?? 0,
    1,
  );
  const visibleEvents = useMemo(
    () => events.filter((event) => selectedKinds.has(event.kind)),
    [events, selectedKinds],
  );

  function toggle(kind: TimelineEvent["kind"]): void {
    setSelectedKinds((current) => {
      const next = new Set(current);
      if (next.has(kind)) next.delete(kind);
      else next.add(kind);
      return next;
    });
  }

  if (events.length === 0) {
    return (
      <div className="timeline-empty">
        No structured rally, contact, bounce, or shot events are available yet.
      </div>
    );
  }

  return (
    <div className="timeline">
      <div className="timeline-filters" aria-label="Timeline event filters">
        {(Object.keys(KIND_LABELS) as TimelineEvent["kind"][]).map((kind) => (
          <button
            key={kind}
            type="button"
            className={`timeline-filter timeline-filter--${kind}`}
            aria-pressed={selectedKinds.has(kind)}
            onClick={() => toggle(kind)}
          >
            <MarkerIcon kind={kind} />
            {KIND_LABELS[kind]}
          </button>
        ))}
      </div>
      <div className="timeline-track" aria-label={`Event timeline, ${formatDuration(duration)}`}>
        <span className="timeline-time timeline-time--start">0:00</span>
        <span className="timeline-time timeline-time--end">{formatDuration(duration)}</span>
        {visibleEvents.map((event) => (
          <button
            key={event.id}
            type="button"
            className={`timeline-marker timeline-marker--${event.kind}`}
            style={{ left: `${Math.min(100, (event.timestampSeconds / duration) * 100)}%` }}
            title={`${event.label} | ${formatDuration(event.timestampSeconds)} | ${formatConfidence(event.confidence)}`}
            aria-label={`Seek to ${event.label} at ${formatDuration(event.timestampSeconds)}`}
            onClick={() => onSeek(event.timestampSeconds)}
          />
        ))}
      </div>
      <p className="timeline-note">
        Select a marker to seek the active video. Event confidence is retained in each marker tooltip.
      </p>
    </div>
  );
}
