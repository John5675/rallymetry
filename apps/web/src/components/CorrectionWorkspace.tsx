import { CheckCircle2, RotateCcw, Save, ShieldCheck, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

/* eslint-disable react-hooks/set-state-in-effect -- persisted target changes intentionally reset the local form draft */

import { useApi } from "../api/context";
import type {
  Correction,
  CorrectionInput,
  CorrectionType,
  DomainRecord,
  JsonValue,
  MatchDashboardData,
} from "../api/types";
import { formatConfidence, numberValue, stringValue } from "../domain";

interface CorrectionWorkspaceProps {
  data: MatchDashboardData;
  onChanged: () => void;
}

const TYPES: { value: CorrectionType; label: string }[] = [
  { value: "PLAYER_IDENTITY", label: "Player identity" },
  { value: "RALLY_BOUNDARY", label: "Rally boundary" },
  { value: "BOUNCE", label: "Bounce" },
  { value: "HITTER", label: "Hitter" },
  { value: "SHOT_TYPE", label: "Shot type" },
];
const SHOT_TYPES = [
  "SERVE",
  "RETURN",
  "DINK",
  "DROP",
  "DRIVE",
  "VOLLEY",
  "OVERHEAD",
  "OTHER",
  "UNKNOWN",
];

interface TargetOption {
  id: string;
  label: string;
  record: DomainRecord | null;
}

function targets(type: CorrectionType, data: MatchDashboardData): TargetOption[] {
  if (type === "PLAYER_IDENTITY") {
    return data.players.map((player) => ({
      id: player.playerId,
      label: `${player.displayName ?? player.playerId} · ${player.logicalIdentity ?? "unassigned"}`,
      record: null,
    }));
  }
  const records =
    type === "RALLY_BOUNDARY"
      ? data.rallies
      : type === "BOUNCE"
        ? data.bounces
        : data.shots;
  return records.map((record) => ({
    id: record.recordId,
    label:
      type === "RALLY_BOUNDARY"
        ? stringValue(record.payload, "rallyId") ?? record.recordId
        : type === "BOUNCE"
          ? `${record.recordId} · ${numberValue(record.payload, "timestampSeconds") ?? record.timestampSeconds ?? "?"}s`
          : `Shot #${numberValue(record.payload, "shotIndex") ?? "?"} · ${record.recordId}`,
    record,
  }));
}

function sourcePrediction(
  type: CorrectionType,
  targetId: string,
  data: MatchDashboardData,
): Record<string, JsonValue> {
  const existing = data.corrections.find(
    (item) => item.correctionType === type && item.targetRecordId === targetId,
  );
  if (existing !== undefined) return existing.prediction;
  if (type === "PLAYER_IDENTITY") {
    const player = data.players.find((item) => item.playerId === targetId);
    return {
      playerId: player?.playerId ?? targetId,
      logicalIdentity: player?.logicalIdentity ?? null,
      displayName: player?.displayName ?? null,
    };
  }
  const option = targets(type, data).find((item) => item.id === targetId);
  const payload = option?.record?.payload ?? {};
  if (type === "RALLY_BOUNDARY") {
    const boundary: Record<string, JsonValue> = {};
    for (const key of ["startFrame", "endFrame", "startTimestamp", "endTimestamp"]) {
      const value = payload[key];
      if (value !== undefined) boundary[key] = value;
    }
    return boundary;
  }
  if (type === "BOUNCE") {
    return {
      isBounce: true,
      frame: payload.frame ?? null,
      timestampSeconds: payload.timestampSeconds ?? option?.record?.timestampSeconds ?? null,
    };
  }
  return type === "HITTER"
    ? { hitterId: payload.hitterId ?? null }
    : { shotType: payload.shotType ?? "UNKNOWN" };
}

function pretty(value: Record<string, JsonValue>): string {
  return JSON.stringify(value, null, 2);
}

function scalarText(value: JsonValue | undefined, fallback = ""): string {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean"
    ? String(value)
    : fallback;
}

export function CorrectionWorkspace({ data, onChanged }: CorrectionWorkspaceProps) {
  const api = useApi();
  const [type, setType] = useState<CorrectionType>("SHOT_TYPE");
  const options = useMemo(() => targets(type, data), [data, type]);
  const [targetId, setTargetId] = useState("");
  const [primaryValue, setPrimaryValue] = useState("");
  const [secondaryValue, setSecondaryValue] = useState("");
  const [verified, setVerified] = useState(true);
  const [reason, setReason] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const resolvedTargetId = options.some((item) => item.id === targetId)
    ? targetId
    : options[0]?.id ?? "";

  const existing = data.corrections.find(
    (item) => item.correctionType === type && item.targetRecordId === resolvedTargetId,
  );
  const prediction = useMemo(
    () => sourcePrediction(type, resolvedTargetId, data),
    [data, resolvedTargetId, type],
  );

  useEffect(() => {
    const value = existing?.humanCorrection ?? prediction;
    if (type === "PLAYER_IDENTITY") {
      setPrimaryValue(scalarText(value.logicalIdentity ?? value.playerId));
      setSecondaryValue(scalarText(value.displayName));
    } else if (type === "RALLY_BOUNDARY") {
      setPrimaryValue(scalarText(value.startTimestamp));
      setSecondaryValue(scalarText(value.endTimestamp));
    } else if (type === "BOUNCE") {
      setPrimaryValue(scalarText(value.isBounce, "true"));
      setSecondaryValue(scalarText(value.timestampSeconds));
    } else if (type === "HITTER") {
      setPrimaryValue(scalarText(value.playerId ?? value.hitterId, "UNKNOWN"));
      setSecondaryValue("");
    } else {
      setPrimaryValue(scalarText(value.shotType, "UNKNOWN"));
      setSecondaryValue("");
    }
    setVerified(existing?.verified ?? true);
    setReason(existing?.reason ?? "");
  }, [existing, prediction, resolvedTargetId, type]);

  const humanCorrection = (): Record<string, JsonValue> => {
    if (type === "PLAYER_IDENTITY") {
      return {
        logicalIdentity: primaryValue,
        ...(secondaryValue.trim() === "" ? {} : { displayName: secondaryValue.trim() }),
      };
    }
    if (type === "RALLY_BOUNDARY") {
      return {
        startTimestamp: Number(primaryValue),
        endTimestamp: Number(secondaryValue),
      };
    }
    if (type === "BOUNCE") {
      return {
        isBounce: primaryValue === "true",
        ...(secondaryValue.trim() === "" ? {} : { timestampSeconds: Number(secondaryValue) }),
      };
    }
    if (type === "HITTER") return { playerId: primaryValue };
    return { shotType: primaryValue };
  };

  const save = async () => {
    if (resolvedTargetId === "") return;
    setSaving(true);
    setMessage(null);
    const input: CorrectionInput = {
      correctionType: type,
      targetRecordId: resolvedTargetId,
      humanCorrection: humanCorrection(),
      verified,
      ...(reason.trim() === "" ? {} : { reason: reason.trim() }),
    };
    try {
      if (existing === undefined) await api.createCorrection(data.match.matchId, input);
      else {
        await api.updateCorrection(data.match.matchId, existing.correctionId, {
          humanCorrection: input.humanCorrection,
          verified,
          ...(input.reason === undefined ? {} : { reason: input.reason }),
        });
      }
      setMessage(existing === undefined ? "Correction saved." : "Correction revision saved.");
      onChanged();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save correction.");
    } finally {
      setSaving(false);
    }
  };

  const remove = async (correction: Correction) => {
    setSaving(true);
    setMessage(null);
    try {
      await api.removeCorrection(data.match.matchId, correction.correctionId);
      setMessage("Correction removed; the AI prediction is still intact.");
      onChanged();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to remove correction.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="correction-workspace">
      <div className="correction-editor">
        <div className="correction-editor__heading">
          <ShieldCheck aria-hidden="true" />
          <div>
            <strong>Correction editor</strong>
            <p>Machine predictions stay immutable. Saving creates a separate human revision.</p>
          </div>
        </div>
        <div className="correction-fields">
          <label>
            <span>Correction</span>
            <select value={type} onChange={(event) => { setType(event.target.value as CorrectionType); setTargetId(""); }}>
              {TYPES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
            </select>
          </label>
          <label>
            <span>Prediction target</span>
            <select value={resolvedTargetId} onChange={(event) => setTargetId(event.target.value)}>
              {options.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
            </select>
          </label>
          {type === "PLAYER_IDENTITY" ? (
            <>
              <label><span>Correct logical identity</span><input value={primaryValue} onChange={(event) => setPrimaryValue(event.target.value)} /></label>
              <label><span>Correct display name</span><input value={secondaryValue} onChange={(event) => setSecondaryValue(event.target.value)} /></label>
            </>
          ) : type === "RALLY_BOUNDARY" ? (
            <>
              <label><span>Correct start (seconds)</span><input type="number" min="0" step="0.001" value={primaryValue} onChange={(event) => setPrimaryValue(event.target.value)} /></label>
              <label><span>Correct end (seconds)</span><input type="number" min="0" step="0.001" value={secondaryValue} onChange={(event) => setSecondaryValue(event.target.value)} /></label>
            </>
          ) : type === "BOUNCE" ? (
            <>
              <label><span>Is a primary-match bounce?</span><select value={primaryValue} onChange={(event) => setPrimaryValue(event.target.value)}><option value="true">Yes</option><option value="false">No</option></select></label>
              <label><span>Correct time (optional)</span><input type="number" min="0" step="0.001" value={secondaryValue} onChange={(event) => setSecondaryValue(event.target.value)} /></label>
            </>
          ) : type === "HITTER" ? (
            <label><span>Correct hitter</span><select value={primaryValue} onChange={(event) => setPrimaryValue(event.target.value)}><option value="UNKNOWN">Unknown</option>{data.players.map((player) => <option key={player.playerId} value={player.logicalIdentity ?? player.playerId}>{player.displayName ?? player.playerId}</option>)}</select></label>
          ) : (
            <label><span>Correct shot type</span><select value={primaryValue} onChange={(event) => setPrimaryValue(event.target.value)}>{SHOT_TYPES.map((shotType) => <option key={shotType}>{shotType}</option>)}</select></label>
          )}
          <label className="correction-fields__reason"><span>Review note (optional)</span><input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="What visual evidence supports this?" /></label>
        </div>
        <div className="correction-compare">
          <div className="correction-compare__prediction">
            <span>AI prediction</span>
            <pre>{pretty(prediction)}</pre>
            <small>{existing === undefined ? "Source model record" : `${formatConfidence(existing.predictionConfidence)} · ${existing.predictionVersion ?? "version unavailable"}`}</small>
          </div>
          <div className="correction-compare__human">
            <span><CheckCircle2 aria-hidden="true" /> Human correction</span>
            <pre>{pretty(humanCorrection())}</pre>
            <label className="verified-toggle"><input type="checkbox" checked={verified} onChange={(event) => setVerified(event.target.checked)} /> Verified for analytics</label>
          </div>
        </div>
        <div className="correction-actions">
          <button className="button button--primary" type="button" disabled={saving || resolvedTargetId === ""} onClick={() => void save()}><Save aria-hidden="true" /> {existing === undefined ? "Save correction" : `Save revision ${existing.revision + 1}`}</button>
          {existing === undefined ? null : <button className="button button--danger" type="button" disabled={saving} onClick={() => void remove(existing)}><Trash2 aria-hidden="true" /> Remove</button>}
          {message === null ? null : <span className="correction-message" role="status">{message}</span>}
        </div>
      </div>

      <div className="correction-ledger">
        <div className="correction-ledger__heading"><RotateCcw aria-hidden="true" /><div><strong>Correction ledger</strong><p>{data.corrections.length} active human layer{data.corrections.length === 1 ? "" : "s"}</p></div></div>
        {data.corrections.map((correction) => (
          <article key={correction.correctionId} className="correction-card">
            <div><span className="correction-card__type">{correction.correctionType.replaceAll("_", " ")}</span><strong>{correction.targetRecordId}</strong></div>
            <div className="correction-card__values"><span><small>AI</small>{pretty(correction.prediction)}</span><span><small>Corrected · r{correction.revision}</small>{pretty(correction.humanCorrection)}</span></div>
            <footer><span className={correction.verified ? "verified" : "unverified"}>{correction.verified ? "Verified" : "Draft"}</span><time dateTime={correction.correctedAt}>{new Date(correction.correctedAt).toLocaleString()}</time></footer>
          </article>
        ))}
        {data.corrections.length === 0 ? <p className="inline-empty">No corrections yet. AI predictions are currently used as-is.</p> : null}
      </div>
    </div>
  );
}
