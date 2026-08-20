import type { DomainRecord, Player } from "../api/types";
import {
  asObject,
  formatConfidence,
  formatDuration,
  numberValue,
  playerNameById,
  stringValue,
} from "../domain";

interface ShotTableProps {
  shots: DomainRecord[];
  players: Player[];
}

export function ShotTable({ shots, players }: ShotTableProps) {
  const ordered = [...shots].sort((left, right) => {
    const leftTime = numberValue(left.payload, "contactTimestamp") ?? left.timestampSeconds ?? 0;
    const rightTime = numberValue(right.payload, "contactTimestamp") ?? right.timestampSeconds ?? 0;
    return leftTime - rightTime;
  });

  return (
    <div className="data-table-wrap">
      <table>
        <thead>
          <tr>
            <th>Shot</th>
            <th>Rally</th>
            <th>Hitter</th>
            <th>Type</th>
            <th>Time</th>
            <th>Confidence</th>
            <th>Landing (m)</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map((shot) => {
            const landing = asObject(shot.payload.landingCourtPosition);
            const x = landing === null ? null : numberValue(landing, "x");
            const y = landing === null ? null : numberValue(landing, "y");
            const timestamp =
              numberValue(shot.payload, "contactTimestamp") ?? shot.timestampSeconds;
            const confidence = numberValue(shot.payload, "confidence") ?? shot.confidence;
            return (
              <tr key={shot.recordId}>
                <td><strong>#{numberValue(shot.payload, "shotIndex") ?? "—"}</strong></td>
                <td>{stringValue(shot.payload, "rallyId") ?? "—"}</td>
                <td>{playerNameById(players, stringValue(shot.payload, "hitterId"))}</td>
                <td><span className="shot-type">{stringValue(shot.payload, "shotType") ?? "UNKNOWN"}</span></td>
                <td>{formatDuration(timestamp)}</td>
                <td>{formatConfidence(confidence)}</td>
                <td>{x === null || y === null ? "—" : `${x.toFixed(2)}, ${y.toFixed(2)}`}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {ordered.length === 0 ? <p className="table-empty">No structured shots are available.</p> : null}
    </div>
  );
}
