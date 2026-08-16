"""Loopback-only browser interface for synchronized match ground-truth annotation."""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.parse import urlparse

from pickleball_vision.errors import MatchAnnotationInputError, MatchAnnotationIoError
from pickleball_vision.match_annotation import (
    MatchAnnotationArtifacts,
    MatchAnnotationStore,
)
from pickleball_vision.media import MediaTimeline

MAXIMUM_REQUEST_BYTES = 1024 * 1024
BYTE_RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")

MATCH_ANNOTATION_UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Pickleball Vision · Match Ground Truth</title>
  <style>
    :root { color-scheme: dark; --bg:#10141a; --panel:#1a2029; --line:#313b48;
      --text:#e9edf3; --muted:#9eabb9; --blue:#55a9ff; --green:#55d68b;
      --red:#ff6565; --orange:#ffb34d; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px/1.35 system-ui,sans-serif; }
    header { display:flex; align-items:center; gap:16px; padding:12px 18px; border-bottom:1px solid var(--line); }
    header h1 { margin:0; font-size:18px; }
    header .spacer { flex:1; }
    button, input, select, textarea { background:#111720; border:1px solid #3b4654; color:var(--text);
      border-radius:5px; padding:7px 9px; }
    button { cursor:pointer; } button:hover { border-color:var(--blue); }
    button.primary { background:#1564ae; border-color:#2f8be0; }
    button.danger { color:#ffb1b1; border-color:#7d3e45; }
    main { display:grid; grid-template-columns:minmax(560px, 1.7fr) minmax(390px, 1fr); min-height:calc(100vh - 55px); }
    .media-column { padding:14px; min-width:0; }
    .editor-column { border-left:1px solid var(--line); padding:14px; min-width:0; max-height:calc(100vh - 55px); overflow:auto; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; margin-bottom:12px; }
    video { width:100%; max-height:58vh; background:#000; display:block; }
    .playback { display:flex; align-items:center; gap:7px; margin-top:9px; flex-wrap:wrap; }
    .playback input[type=range] { flex:1; min-width:180px; padding:0; }
    #timeline { width:100%; height:96px; display:block; background:#111720; cursor:crosshair; border-radius:5px; }
    #waveform { width:100%; display:none; margin-top:8px; border-radius:5px; }
    .legend { color:var(--muted); margin:7px 0 0; font-size:12px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:9px; }
    label { display:flex; flex-direction:column; gap:4px; color:var(--muted); }
    label.wide { grid-column:1/-1; }
    textarea { min-height:68px; resize:vertical; }
    .actions { display:flex; gap:7px; flex-wrap:wrap; margin-top:11px; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th, td { text-align:left; padding:7px 5px; border-bottom:1px solid var(--line); }
    tbody tr { cursor:pointer; } tbody tr:hover { background:#222b37; }
    tbody tr.selected { background:#173b5e; }
    .event-list { max-height:330px; overflow:auto; }
    #status.error { color:#ff8f8f; } #status { color:var(--green); }
    .contract { color:var(--orange); }
    kbd { background:#111720; border:1px solid #45515f; border-bottom-width:2px; border-radius:4px; padding:1px 5px; }
    @media (max-width:1000px) { main { grid-template-columns:1fr; } .editor-column { border-left:0; border-top:1px solid var(--line); max-height:none; } }
  </style>
</head>
<body>
<header>
  <h1>Multimodal Match Ground Truth</h1>
  <span id="summary">Loading…</span><span class="spacer"></span>
  <span id="status"></span><button id="stop" class="danger">Save & stop server</button>
</header>
<main>
  <section class="media-column">
    <div class="panel">
      <video id="video" controls preload="metadata"></video>
      <div class="playback">
        <button id="play">Play / pause</button><button id="prev-frame">◀ frame</button>
        <button id="next-frame">frame ▶</button>
        <input id="scrub" type="range" min="0" max="1" step="0.001" value="0">
        <b id="position">frame 0 · 0.000s</b>
      </div>
    </div>
    <div class="panel">
      <canvas id="timeline" width="1500" height="96"></canvas>
      <img id="waveform" alt="Synchronized audio waveform">
      <p class="legend"><span class="contract">Orange ticks are generic Prompt 10 audio transients, not contacts or bounces.</span>
        Colored ticks are human annotations. Click the timeline to seek.</p>
    </div>
    <div class="panel">
      <b>Shortcuts:</b>
      <kbd>Space</kbd> play/pause · <kbd>,</kbd>/<kbd>.</kbd> frame step ·
      <kbd>J</kbd>/<kbd>L</kbd> seek 5s · <kbd>A</kbd> add selected type ·
      <kbd>E</kbd> update · <kbd>Delete</kbd> delete ·
      <kbd>1-7</kbd> quick-add event types
    </div>
  </section>
  <aside class="editor-column">
    <div class="panel">
      <h2 id="form-title">Add event at current frame</h2>
      <div class="grid">
        <label>Event type<select id="event-type"></select></label>
        <label>Frame<input id="frame" type="number" min="0"></label>
        <label>Player ID<input id="player" list="players" placeholder="optional"></label>
        <datalist id="players"><option>ME</option><option>PARTNER</option><option>OPPONENT_1</option><option>OPPONENT_2</option></datalist>
        <label>Team<input id="team" placeholder="optional"></label>
        <label>Shot type<input id="shot-type" list="shots" placeholder="optional"></label>
        <datalist id="shots"><option>SERVE</option><option>RETURN</option><option>DRIVE</option><option>DROP</option><option>DINK</option><option>VOLLEY</option><option>RESET</option><option>LOB</option><option>OVERHEAD</option></datalist>
        <label>Audio label<select id="audio-label"><option value="">None (normal)</option></select></label>
        <label>Annotator<input id="annotator" value="local-annotator"></label>
        <label>Court X (meters)<input id="court-x" type="number" step="0.01" placeholder="optional"></label>
        <label>Court Y (meters)<input id="court-y" type="number" step="0.01" placeholder="optional"></label>
        <label class="wide">Annotation confidence <span id="confidence-value">1.00</span>
          <input id="confidence" type="range" min="0" max="1" step="0.01" value="1"></label>
        <label class="wide">Notes<textarea id="notes" maxlength="4000" placeholder="optional"></textarea></label>
      </div>
      <div class="actions">
        <button id="use-current">Use current frame</button>
        <button id="add" class="primary">Add event</button>
        <button id="update">Update selected</button>
        <button id="delete" class="danger">Delete selected</button>
        <button id="clear">Clear selection</button>
      </div>
    </div>
    <div class="panel">
      <h2>Ground-truth events</h2>
      <div class="event-list"><table><thead><tr><th>Frame</th><th>Time</th><th>Type</th><th>Player/team</th></tr></thead><tbody id="events"></tbody></table></div>
    </div>
  </aside>
</main>
<script>
"use strict";
const $ = id => document.getElementById(id);
const video = $("video"), timeline = $("timeline"), ctx = timeline.getContext("2d");
let session = null, selectedId = null;
const eventColors = {RALLY_START:"#57d68d",RALLY_END:"#e95d73",SERVE_CONTACT:"#63b6ff",
 PADDLE_CONTACT:"#40d4de",BOUNCE:"#ffd24a",RALLY_WINNER:"#cf8cff",SHOT_TYPE:"#ff9e5d"};
function setStatus(text, error=false) { $("status").textContent=text; $("status").className=error?"error":""; }
function frameNow() { return Math.max(0, Math.min(session.video.frame_count-1, Math.round(video.currentTime*session.video.fps))); }
function videoTimeForFrame(frame) { return frame/session.video.fps; }
function seekFrame(frame) { video.pause(); video.currentTime=videoTimeForFrame(Math.max(0,Math.min(session.video.frame_count-1,frame))); updatePosition(); }
function seekSeconds(seconds) { video.currentTime=Math.max(0,Math.min(session.video.duration,seconds)); updatePosition(); }
async function request(url, options={}) { const response=await fetch(url,options); const body=await response.json().catch(()=>({})); if(!response.ok) throw new Error(body.error||`${response.status} ${response.statusText}`); return body; }
function optionalValue(id) { const value=$(id).value.trim(); return value || null; }
function escapeHtml(value) { const span=document.createElement("span");span.textContent=String(value);return span.innerHTML; }
function eventPayload() {
  const x=optionalValue("court-x"), y=optionalValue("court-y");
  if ((x===null)!==(y===null)) throw new Error("Court X and Y must both be provided or both blank.");
  const courtPosition=x===null?null:{xMeters:Number(x),yMeters:Number(y),coordinateSystem:"canonical_pickleball_court",source:"HUMAN_ANNOTATION"};
  return {type:$("event-type").value,frame:Number($("frame").value),playerId:optionalValue("player"),team:optionalValue("team"),shotType:optionalValue("shot-type"),courtPosition,
    audioLabel:$("audio-label").value||null,notes:optionalValue("notes"),annotationConfidence:Number($("confidence").value),annotator:optionalValue("annotator")||"local-annotator"};
}
function setForm(event=null) {
  selectedId=event?event.id:null; $("form-title").textContent=event?`Edit ${event.id}`:"Add event at current frame";
  $("event-type").value=event?event.type:$("event-type").value||"RALLY_START";
  $("frame").value=String(event?event.frame:frameNow()); $("player").value=event?.playerId||""; $("team").value=event?.team||"";
  $("shot-type").value=event?.shotType||""; $("audio-label").value=event?.audioLabel||""; $("annotator").value=event?.annotator||localStorage.getItem("match-annotator")||"local-annotator";
  $("court-x").value=event?.courtPosition?.xMeters??""; $("court-y").value=event?.courtPosition?.yMeters??"";
  $("notes").value=event?.notes||""; $("confidence").value=String(event?.annotationConfidence??1); $("confidence-value").textContent=Number($("confidence").value).toFixed(2);
  renderEvents();
}
function updatePosition() { if(!session)return; const frame=frameNow(); $("position").textContent=`frame ${frame} · ${video.currentTime.toFixed(3)}s`; $("scrub").value=String(video.currentTime); drawTimeline(); }
function drawTimeline() {
  if(!session)return; const w=timeline.width,h=timeline.height,d=Math.max(session.video.duration,0.001);
  ctx.fillStyle="#111720";ctx.fillRect(0,0,w,h);ctx.strokeStyle="#303a47";ctx.lineWidth=1;
  for(let i=0;i<=10;i++){const x=i*w/10;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();ctx.fillStyle="#9eabb9";ctx.font="12px system-ui";ctx.fillText(`${(i*d/10).toFixed(1)}s`,Math.min(w-45,x+3),h-5);}
  const start=session.video.mediaTimelineStartSeconds;
  for(const marker of session.audioContext.transientMarkers){const t=marker.mediaTimestampSeconds-start;if(t<0||t>d)continue;const x=t/d*w;ctx.strokeStyle="#ffad42";ctx.globalAlpha=.25+.65*marker.confidence;ctx.beginPath();ctx.moveTo(x,3);ctx.lineTo(x,28);ctx.stroke();}
  ctx.globalAlpha=1;
  for(const event of session.events){const x=event.videoTimestampSeconds/d*w;ctx.strokeStyle=eventColors[event.type]||"#fff";ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,34);ctx.lineTo(x,h-19);ctx.stroke();}
  const px=video.currentTime/d*w;ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(px,0);ctx.lineTo(px,h);ctx.stroke();
}
function renderEvents() {
  if(!session)return; $("events").innerHTML=session.events.map(event=>`<tr data-id="${event.id}" class="${event.id===selectedId?'selected':''}"><td>${event.frame}</td><td>${event.videoTimestampSeconds.toFixed(3)}s</td><td style="color:${eventColors[event.type]||'#fff'}">${event.type}</td><td>${escapeHtml(event.playerId||event.team||'--')}</td></tr>`).join("");
  for(const row of $("events").querySelectorAll("tr")){row.onclick=()=>{const event=session.events.find(item=>item.id===row.dataset.id);setForm(event);seekFrame(event.frame);};}
  const counts=session.counts; $("summary").textContent=`${counts.total} events · ${session.video.frame_count} frames · audio context ${session.audioContext.audioAnalysisAvailable?'available':'optional/unavailable'}`;
}
async function reload(selectId=null) { session=await request("/api/session"); $("scrub").max=String(session.video.duration); if(session.audioContext.waveformUrl){$("waveform").src=session.audioContext.waveformUrl;$("waveform").style.display="block";} renderEvents();drawTimeline(); if(selectId){const event=session.events.find(item=>item.id===selectId);if(event)setForm(event);} }
async function addEvent(typeOverride=null) { try {if(typeOverride)$("event-type").value=typeOverride;$("frame").value=String(frameNow());const event=await request("/api/events",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(eventPayload())});localStorage.setItem("match-annotator",event.annotator);await reload(event.id);setStatus(`Added ${event.type} at frame ${event.frame}.`);}catch(error){setStatus(error.message,true);} }
async function updateEvent() { if(!selectedId){setStatus("Select an event to update.",true);return;} try{const event=await request(`/api/events/${selectedId}`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(eventPayload())});localStorage.setItem("match-annotator",event.annotator);await reload(event.id);setStatus(`Updated ${event.id}.`);}catch(error){setStatus(error.message,true);} }
async function deleteEvent() { if(!selectedId){setStatus("Select an event to delete.",true);return;} if(!confirm(`Delete ${selectedId}?`))return; try{await request(`/api/events/${selectedId}`,{method:"DELETE"});const deleted=selectedId;await reload();setForm();setStatus(`Deleted ${deleted}.`);}catch(error){setStatus(error.message,true);} }
video.addEventListener("timeupdate",updatePosition);video.addEventListener("loadedmetadata",updatePosition);
timeline.onclick=event=>{const rect=timeline.getBoundingClientRect();seekSeconds((event.clientX-rect.left)/rect.width*session.video.duration);};
$("scrub").oninput=event=>seekSeconds(Number(event.target.value)); $("play").onclick=()=>video.paused?video.play():video.pause();
$("prev-frame").onclick=()=>seekFrame(frameNow()-1);$("next-frame").onclick=()=>seekFrame(frameNow()+1);$("use-current").onclick=()=>{$("frame").value=String(frameNow());};
$("confidence").oninput=()=>{$("confidence-value").textContent=Number($("confidence").value).toFixed(2);};
$("add").onclick=()=>addEvent();$("update").onclick=updateEvent;$("delete").onclick=deleteEvent;$("clear").onclick=()=>setForm();
$("stop").onclick=async()=>{await request("/api/shutdown",{method:"POST"});document.body.innerHTML="<main style='display:block;padding:40px'><h1>Annotations saved. Server stopped.</h1><p>You may close this tab.</p></main>";};
document.addEventListener("keydown",event=>{if(["INPUT","TEXTAREA","SELECT"].includes(event.target.tagName))return;const key=event.key.toLowerCase();if(event.code==="Space"){event.preventDefault();video.paused?video.play():video.pause();}else if(event.key===",")seekFrame(frameNow()-1);else if(event.key===".")seekFrame(frameNow()+1);else if(key==="j")seekSeconds(video.currentTime-5);else if(key==="l")seekSeconds(video.currentTime+5);else if(key==="a")addEvent();else if(key==="e")updateEvent();else if(event.key==="Delete"||event.key==="Backspace")deleteEvent();else if("1234567".includes(event.key)){const types=["RALLY_START","SERVE_CONTACT","PADDLE_CONTACT","BOUNCE","RALLY_END","RALLY_WINNER","SHOT_TYPE"];addEvent(types[Number(event.key)-1]);}});
(async()=>{try{session=await request("/api/session");for(const type of session.eventTypes)$("event-type").add(new Option(type,type));for(const label of session.audioLabels)$("audio-label").add(new Option(label,label));video.src=session.video.videoUrl;$("scrub").max=String(session.video.duration);if(session.audioContext.waveformUrl){$("waveform").src=session.audioContext.waveformUrl;$("waveform").style.display="block";}setForm();renderEvents();drawTimeline();setStatus("Ready. Every edit saves immediately.");}catch(error){setStatus(error.message,true);}})();
</script>
</body></html>"""


def _range_bounds(header: str, *, file_size: int) -> tuple[int, int]:
    match = BYTE_RANGE_PATTERN.fullmatch(header.strip())
    if match is None or file_size < 1:
        raise ValueError("unsupported byte range")
    first, last = match.groups()
    if not first and not last:
        raise ValueError("empty byte range")
    if first:
        start = int(first)
        end = int(last) if last else file_size - 1
    else:
        suffix_length = int(last)
        if suffix_length < 1:
            raise ValueError("invalid suffix byte range")
        start = max(0, file_size - suffix_length)
        end = file_size - 1
    if start < 0 or start >= file_size or end < start:
        raise ValueError("byte range lies outside file")
    return start, min(end, file_size - 1)


class _MatchAnnotationRequestHandler(BaseHTTPRequestHandler):
    """Tiny loopback adapter for the local editor; never a product API."""

    server_version = "PickleballVisionMatchAnnotation/1"

    def __init__(self, *args: Any, store: MatchAnnotationStore, **kwargs: Any) -> None:
        self.store = store
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _send_bytes(
        self,
        payload: bytes,
        *,
        content_type: str,
        status: int = 200,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: object, *, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(payload, allow_nan=False).encode("utf-8"),
            content_type="application/json; charset=utf-8",
            status=status,
        )

    def _copy_bytes(self, source: BinaryIO, count: int) -> None:
        remaining = count
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return
            remaining -= len(chunk)

    def _send_file_headers(self, path: Path, *, allow_ranges: bool) -> None:
        file_size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        if allow_ranges:
            self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send_file(self, path: Path, *, allow_ranges: bool) -> None:
        file_size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range") if allow_ranges else None
        if range_header is None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            if allow_ranges:
                self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as source:
                self._copy_bytes(source, file_size)
            return
        try:
            start, end = _range_bounds(range_header, file_size=file_size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return
        length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as source:
            source.seek(start)
            self._copy_bytes(source, length)

    def _event_id(self) -> str | None:
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["api", "events"]:
            return parts[2]
        return None

    def _request_json(self) -> object:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length < 1 or content_length > MAXIMUM_REQUEST_BYTES:
            raise MatchAnnotationInputError("request body has an invalid size")
        return json.loads(self.rfile.read(content_length).decode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._send_bytes(
                    MATCH_ANNOTATION_UI_HTML.encode("utf-8"),
                    content_type="text/html; charset=utf-8",
                )
                return
            if path == "/api/session":
                self._send_json(self.store.session_payload())
                return
            if path == "/media/video":
                self._send_file(self.store.video_path, allow_ranges=True)
                return
            if path == "/media/waveform" and self.store.audio_context.waveform_path is not None:
                self._send_file(self.store.audio_context.waveform_path, allow_ranges=False)
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            return
        except (MatchAnnotationInputError, MatchAnnotationIoError, OSError) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/media/video":
                self._send_file_headers(self.store.video_path, allow_ranges=True)
                return
            if path == "/media/waveform" and self.store.audio_context.waveform_path is not None:
                self._send_file_headers(self.store.audio_context.waveform_path, allow_ranges=False)
                return
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            return
        except OSError as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/events":
                self._send_json(self.store.add_event(self._request_json()))
                return
            if path == "/api/shutdown":
                self._send_json({"status": "stopping"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            MatchAnnotationInputError,
            MatchAnnotationIoError,
        ) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

    def do_PUT(self) -> None:
        event_id = self._event_id()
        if event_id is None:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            self._send_json(self.store.update_event(event_id, self._request_json()))
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            MatchAnnotationInputError,
            MatchAnnotationIoError,
        ) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

    def do_DELETE(self) -> None:
        event_id = self._event_id()
        if event_id is None:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            self._send_json(self.store.delete_event(event_id))
        except (MatchAnnotationInputError, MatchAnnotationIoError) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)


def serve_match_annotation(
    video_path: Path,
    *,
    output_path: Path,
    timeline: MediaTimeline | None = None,
    audio_events_path: Path | None = None,
    port: int = 8766,
    open_browser: bool = True,
    on_started: Callable[[str], None] | None = None,
) -> MatchAnnotationArtifacts:
    """Serve the local editor until stopped in the browser or with Ctrl-C."""

    if not 0 <= port <= 65535:
        raise MatchAnnotationInputError("annotation server port must be between 0 and 65535")
    store = MatchAnnotationStore(
        video_path,
        output_path=output_path,
        timeline=timeline,
        audio_events_path=audio_events_path,
    )

    def handler(*args: Any, **kwargs: Any) -> _MatchAnnotationRequestHandler:
        return _MatchAnnotationRequestHandler(*args, store=store, **kwargs)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as error:
        raise MatchAnnotationIoError(
            "127.0.0.1",
            reason=f"unable to start match annotation server: {error}",
        ) from error
    actual_port = cast(tuple[str, int], server.server_address)[1]
    url = f"http://127.0.0.1:{actual_port}/"
    if on_started is not None:
        on_started(url)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    event_count = cast(dict[str, object], store.session_payload()["counts"])["total"]
    return MatchAnnotationArtifacts(
        url=url,
        annotations_path=store.output_path,
        event_count=cast(int, event_count),
        audio_context_available=store.audio_context.analysis_available,
    )
