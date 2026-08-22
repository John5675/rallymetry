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
            const payload = shot.effectivePayload ?? shot.payload;
            const corrected = (shot.verifiedCorrections?.length ?? 0) > 0;
            const landing = asObject(payload.landingCourtPosition);
            const x = landing === null ? null : numberValue(landing, "x");
            const y = landing === null ? null : numberValue(landing, "y");
            const timestamp =
              numberValue(payload, "contactTimestamp") ?? shot.timestampSeconds;
            const confidence = numberValue(payload, "confidence") ?? shot.confidence;
            const hitter = stringValue(payload, "hitterId");
            const shotType = stringValue(payload, "shotType") ?? "UNKNOWN";
            const aiVisualReview = asObject(shot.payload.aiVisualReview);
            const aiReview = aiVisualReview === null ? null : asObject(aiVisualReview.review);
            const reviewedBestGuess =
              aiReview === null ? null : stringValue(aiReview, "legacyBestGuess");
            const reviewedConfidence =
              aiReview === null ? null : numberValue(aiReview, "legacyBestGuessConfidence");
            return (
              <tr key={shot.recordId}>
                <td><strong>#{numberValue(payload, "shotIndex") ?? "N/A"}</strong></td>
                <td>{stringValue(payload, "rallyId") ?? "N/A"}</td>
                <td>{playerNameById(players, hitter)}{corrected && hitter !== stringValue(shot.payload, "hitterId") ? <small className="human-correction-note">AI: {playerNameById(players, stringValue(shot.payload, "hitterId"))}</small> : null}</td>
                <td><span className="shot-type">{shotType}</span>{corrected && shotType !== stringValue(shot.payload, "shotType") ? <small className="human-correction-note">AI prediction: {stringValue(shot.payload, "shotType") ?? "UNKNOWN"}</small> : null}{reviewedBestGuess !== null ? <small className="ai-review-note">AI visual review: {reviewedBestGuess}{reviewedConfidence === null ? "" : ` · ${formatConfidence(reviewedConfidence)}`}</small> : null}</td>
                <td>{formatDuration(timestamp)}</td>
                <td>{formatConfidence(confidence)}</td>
                <td>{x === null || y === null ? "N/A" : `${x.toFixed(2)}, ${y.toFixed(2)}`}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {ordered.length === 0 ? <p className="table-empty">No structured shots are available.</p> : null}
    </div>
  );
}
