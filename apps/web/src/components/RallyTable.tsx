import { ArrowDownUp, Search } from "lucide-react";
import { useMemo, useState } from "react";

import type { DomainRecord } from "../api/types";
import { formatConfidence, formatDuration, numberValue, stringValue } from "../domain";

interface RallyTableProps {
  rallies: DomainRecord[];
}

type SortKey = "id" | "start" | "duration" | "confidence";

interface RallyRow {
  id: string;
  start: number | null;
  end: number | null;
  duration: number | null;
  confidence: number | null;
  signalCount: number;
}

function toRow(rally: DomainRecord): RallyRow {
  const start = numberValue(rally.payload, "startTimestamp") ?? rally.timestampSeconds;
  const end = numberValue(rally.payload, "endTimestamp");
  const signals = rally.payload.supportingSignals;
  return {
    id: stringValue(rally.payload, "rallyId") ?? rally.recordId,
    start,
    end,
    duration: start !== null && end !== null ? Math.max(0, end - start) : null,
    confidence: numberValue(rally.payload, "confidence") ?? rally.confidence,
    signalCount: Array.isArray(signals) ? signals.length : 0,
  };
}

export function RallyTable({ rallies }: RallyTableProps) {
  const [query, setQuery] = useState("");
  const [minimumConfidence, setMinimumConfidence] = useState("0");
  const [sortKey, setSortKey] = useState<SortKey>("start");
  const rows = useMemo(() => {
    const threshold = Number.parseFloat(minimumConfidence);
    return rallies
      .map(toRow)
      .filter(
        (row) =>
          row.id.toLowerCase().includes(query.toLowerCase()) &&
          (row.confidence ?? 0) >= threshold,
      )
      .sort((left, right) => {
        if (sortKey === "id") return left.id.localeCompare(right.id, undefined, { numeric: true });
        return (left[sortKey] ?? Number.POSITIVE_INFINITY) -
          (right[sortKey] ?? Number.POSITIVE_INFINITY);
      });
  }, [minimumConfidence, query, rallies, sortKey]);

  return (
    <div className="data-table-wrap">
      <div className="table-tools">
        <label className="search-control">
          <Search aria-hidden="true" />
          <span className="sr-only">Filter rallies</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Filter rally ID"
          />
        </label>
        <label>
          <span>Confidence</span>
          <select
            value={minimumConfidence}
            onChange={(event) => setMinimumConfidence(event.target.value)}
          >
            <option value="0">All</option>
            <option value="0.5">50%+</option>
            <option value="0.75">75%+</option>
            <option value="0.9">90%+</option>
          </select>
        </label>
        <label>
          <span>Sort</span>
          <select value={sortKey} onChange={(event) => setSortKey(event.target.value as SortKey)}>
            <option value="start">Start time</option>
            <option value="duration">Duration</option>
            <option value="confidence">Confidence</option>
            <option value="id">Rally ID</option>
          </select>
        </label>
      </div>
      <table>
        <thead>
          <tr>
            <th><span>Rally <ArrowDownUp aria-hidden="true" /></span></th>
            <th>Start</th>
            <th>End</th>
            <th>Duration</th>
            <th>Confidence</th>
            <th>Signals</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td><strong>{row.id}</strong></td>
              <td>{formatDuration(row.start)}</td>
              <td>{formatDuration(row.end)}</td>
              <td>{formatDuration(row.duration)}</td>
              <td>{formatConfidence(row.confidence)}</td>
              <td>{row.signalCount}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 ? <p className="table-empty">No rallies match these filters.</p> : null}
    </div>
  );
}
