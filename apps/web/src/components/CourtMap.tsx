import { MapPinned } from "lucide-react";

import type { CourtPoint } from "../domain";
import { COURT_LENGTH_METERS, COURT_WIDTH_METERS, formatConfidence } from "../domain";

interface CourtMapProps {
  points: CourtPoint[];
}

export function CourtMap({ points }: CourtMapProps) {
  return (
    <div className="court-map-wrap">
      <div className="court-map" aria-label={`Top-down court with ${points.length} shot landings`}>
        <div className="court-line court-line--net" />
        <div className="court-line court-line--near-kitchen" />
        <div className="court-line court-line--far-kitchen" />
        <div className="court-line court-line--near-center" />
        <div className="court-line court-line--far-center" />
        {points.map((point) => (
          <button
            key={point.id}
            type="button"
            className={`court-point court-point--${point.shotType.toLowerCase()}`}
            style={{
              left: `${(point.x / COURT_WIDTH_METERS) * 100}%`,
              bottom: `${(point.y / COURT_LENGTH_METERS) * 100}%`,
            }}
            title={`${point.label}: ${point.shotType}, (${point.x.toFixed(2)}, ${point.y.toFixed(2)}) m, confidence ${formatConfidence(point.confidence)}`}
            aria-label={`${point.label}, ${point.shotType}, landing ${point.x.toFixed(2)} by ${point.y.toFixed(2)} meters`}
          />
        ))}
        <span className="court-side-label court-side-label--far">Far side</span>
        <span className="court-side-label court-side-label--near">Near side</span>
      </div>
      {points.length === 0 ? (
        <div className="court-map-empty">
          <MapPinned aria-hidden="true" />
          <span>No defensible landing coordinates available.</span>
        </div>
      ) : (
        <p className="court-caption">
          {points.length} structured landing {points.length === 1 ? "position" : "positions"}. Airborne
          points are not projected.
        </p>
      )}
    </div>
  );
}
