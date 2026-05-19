"""
routes.py — FastAPI application and HTTP/WebSocket route definitions.
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse

import carrier
from clients import http as _http_client
from config import NGROK_URL, MAKE_CALL_API_KEY, RECORDINGS_DIR, TRANSCRIPTS_DIR, AUDIO_TRANSCRIPT_WEBHOOK_URL
from urllib.parse import urlparse as _urlparse

# Pre-parse the ngrok hostname once at import time
# request.url.hostname returns "localhost" when behind ngrok — use NGROK_URL instead
_WS_HOST = _urlparse(NGROK_URL).netloc   # e.g. "a4d3-xxxx.ngrok-free.app"
from scripts import build_default_ctx
from session import pending_ctx
from call_handler import media_stream
import log_buffer as _log_buffer

log = logging.getLogger("aditi")

# Attach in-memory log buffer before anything else logs
_log_buffer.attach()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield                          # server is running
    await _http_client.aclose()   # clean up httpx on shutdown → no "Event loop is closed" warning
    log.info("HTTP client closed")


app = FastAPI(title="Aditi — Hindi EMI Collection Voice Bot", lifespan=_lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    from config import LLM_MODEL, SARVAM_VOICE
    return HTMLResponse(
        f"<h3>Aditi — Sarvam STT · {LLM_MODEL} LLM · Sarvam TTS ({SARVAM_VOICE})</h3>"
    )


@app.get("/health")
async def health() -> JSONResponse:
    from config import LLM_MODEL, SARVAM_VOICE
    return JSONResponse({"status": "ok", "llm": LLM_MODEL, "voice": SARVAM_VOICE})


@app.get("/debug")
async def debug() -> JSONResponse:
    """Diagnose config + connectivity — call this when something breaks."""
    import websockets as _ws
    from config import (
        NGROK_URL, SARVAM_STT_WS_URL, SARVAM_API_KEY,
        LLM_MODEL, SARVAM_VOICE, PORT,
        PLIVO_AUTH_ID, PLIVO_PHONE_NUMBER,
        RECORDING_CALLBACK_URL, AUDIO_TRANSCRIPT_WEBHOOK_URL,
    )

    # Check Sarvam STT reachability
    stt_ok = False
    stt_err = ""
    try:
        import asyncio as _asyncio
        conn = await _asyncio.wait_for(
            _ws.connect(
                SARVAM_STT_WS_URL,
                extra_headers={"Api-Subscription-Key": SARVAM_API_KEY},
            ),
            timeout=5.0,
        )
        await conn.close()
        stt_ok = True
    except Exception as exc:
        stt_err = str(exc)

    return JSONResponse({
        "server": {
            "ngrok_url":  NGROK_URL,
            "ws_host":    _WS_HOST,
            "port":       PORT,
        },
        "plivo": {
            "auth_id":           PLIVO_AUTH_ID[:6] + "…",
            "phone_number":      PLIVO_PHONE_NUMBER,
            "recording_callback": RECORDING_CALLBACK_URL or "(not set)",
        },
        "sarvam_stt": {
            "url": SARVAM_STT_WS_URL,
            "reachable": stt_ok,
            "error":     stt_err or None,
        },
        "llm":   LLM_MODEL,
        "voice": SARVAM_VOICE,
        "audio_transcript_webhook": AUDIO_TRANSCRIPT_WEBHOOK_URL or "(not set)",
    })


@app.websocket("/ws-ping")
async def ws_ping(ws: WebSocket) -> None:
    """WebSocket reachability test — connect to wss://HOST/ws-ping and expect 'pong'."""
    await ws.accept()
    await ws.send_text("pong")
    await ws.close()
    log.info("ws-ping: connection OK")


@app.post("/make-call")
async def make_call(
    request: Request,
    x_api_key: str = Header(default=""),
) -> JSONResponse:
    if MAKE_CALL_API_KEY and x_api_key != MAKE_CALL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Api-Key")

    body = await request.json()
    # Accept phone_number as primary; fall back to legacy "to" key
    to = (body.get("phone_number") or body.get("to") or "").strip()
    if not to:
        raise HTTPException(status_code=422, detail="`phone_number` is required")

    # Build ctx: defaults → caller overrides (int variants derived inside build_default_ctx)
    ctx = {**build_default_ctx(), **{k: str(v) for k, v in body.items()}}
    ctx["phone_number"] = to  # ensure it's always set

    # Re-derive int variants if caller supplied new formatted values
    from scripts import _to_int_str
    if body.get("emi_overdue_amt"):
        ctx["emi_amount_int"] = _to_int_str(ctx["emi_overdue_amt"])
        ctx["emi_amount"]     = ctx["emi_overdue_amt"]
    if body.get("min_partial"):
        ctx["min_partial_int"] = _to_int_str(ctx["min_partial"])
    if body.get("emi_overdue_date"):
        ctx["emi_due_date"] = ctx["emi_overdue_date"]

    call_sid = await carrier.make_call(to, f"{NGROK_URL}/outgoing-call")
    pending_ctx[call_sid] = ctx
    # Evict oldest entries if pending_ctx grows too large (e.g. unanswered calls
    # that never hit /outgoing-call and never get consumed by call_handler).
    if len(pending_ctx) > 200:
        oldest = next(iter(pending_ctx))
        pending_ctx.pop(oldest, None)
        log.warning("pending_ctx overflow — evicted oldest key %s", oldest)
    log.info("Outbound call initiated — SID=%s to=%s", call_sid, to)
    return JSONResponse({"call_sid": call_sid})


@app.api_route("/outgoing-call", methods=["GET", "POST"])
async def outgoing_call(request: Request) -> HTMLResponse:
    # Plivo sends both RequestUUID and CallUUID in the POST body when the call is answered.
    # make_call() stored pending_ctx under request_uuid, so we rekey it to CallUUID
    # so call_handler.py can find the context via sess.call_sid.
    # IMPORTANT: use RequestUUID for direct lookup — never iterate all keys (race condition
    # with concurrent calls causes wrong contexts to be assigned to wrong calls).
    try:
        form = await request.form()
        call_uuid    = form.get("CallUUID",    "")
        request_uuid = form.get("RequestUUID", "")
        if call_uuid and request_uuid and request_uuid in pending_ctx:
            pending_ctx[call_uuid] = pending_ctx.pop(request_uuid)
            log.info("pending_ctx rekeyed: %s → %s", request_uuid, call_uuid)
        elif call_uuid and not request_uuid:
            # Fallback: RequestUUID missing (shouldn't happen) — safe single-key check
            if call_uuid not in pending_ctx and len(pending_ctx) == 1:
                rid = next(iter(pending_ctx))
                pending_ctx[call_uuid] = pending_ctx.pop(rid)
                log.info("pending_ctx fallback rekey: %s → %s", rid, call_uuid)
    except Exception as exc:
        log.debug("outgoing-call rekey skipped: %s", exc)

    ws_url = f"wss://{_WS_HOST}/media-stream"
    log.info("connect_response ws_url=%s", ws_url)
    return HTMLResponse(carrier.connect_response(ws_url), media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream_route(ws: WebSocket) -> None:
    await media_stream(ws)


@app.api_route("/recording-callback", methods=["GET", "POST"])
async def recording_callback(request: Request) -> JSONResponse:
    """
    Plivo POSTs here when a recording is ready.
    Appends recording metadata as a JSON line in the call's transcript file.
    Set RECORDING_CALLBACK_URL=https://<ngrok>/recording-callback in .env
    """
    import json as _json
    from datetime import datetime, timezone
    from session import recording_pending

    # Try every possible encoding Plivo might use
    data: dict = {}
    try:
        # 1. Query-string params (Plivo sometimes uses GET-style params on POST)
        data = dict(request.query_params)
        if not data:
            ct = request.headers.get("content-type", "")
            if "json" in ct:
                data = await request.json()
            else:
                # form-encoded (application/x-www-form-urlencoded)
                form = await request.form()
                data = dict(form)
    except Exception as exc:
        log.warning("Recording callback parse error: %s", exc)

    # Always log raw body + headers for debugging
    try:
        raw_body = await request.body()
    except Exception:
        raw_body = b""
    log.info(
        "Recording callback — headers=%s  query=%s  body=%s",
        dict(request.headers),
        dict(request.query_params),
        raw_body[:500],
    )
    log.info("Recording callback parsed data: %s", data)

    # Plivo sometimes wraps the entire payload as a JSON string in a 'response' field
    if "response" in data and isinstance(data["response"], str):
        import json as _j
        try:
            data = {**data, **_j.loads(data["response"])}
            log.info("Recording callback unwrapped 'response' field: %s", data)
        except Exception:
            pass

    # Plivo field names vary — handle both casings
    call_uuid    = data.get("CallUUID")    or data.get("call_uuid",     "")
    recording_id = data.get("RecordingID") or data.get("recording_id",  "")
    record_url   = data.get("RecordUrl")   or data.get("record_url",    "") \
                or data.get("RecordingUrl") or data.get("recording_url", "")
    duration     = data.get("RecordingDuration") or data.get("recording_duration", "")

    log.info("Recording ready — call=%s duration=%ss url=%s", call_uuid, duration, record_url)

    if not record_url:
        log.warning("Recording callback received but no URL found in payload — skipping")
        return JSONResponse({"status": "ok"})

    from datetime import datetime, timezone

    ts_str = datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z"
    transcript_path = recording_pending.pop(call_uuid, "")

    row = {
        "ts":            ts_str,
        "event":         "recording_ready",
        "call_sid":      call_uuid,
        "recording_id":  recording_id,
        "recording_url": record_url,
        "duration_sec":  duration,
    }

    # ── 1. Append to transcript JSONL ──────────────────────────────────────────
    if transcript_path:
        try:
            with open(transcript_path, "a", encoding="utf-8") as fh:
                fh.write(_json.dumps(row, ensure_ascii=False) + "\n")
            log.info("Recording URL saved to transcript: %s", transcript_path)
        except OSError as exc:
            log.warning("Could not write recording to transcript: %s", exc)
    else:
        log.warning("No transcript found for call_sid=%s", call_uuid)

    # ── 2. Save recording metadata to recordings/<call_sid>.json ──────────────
    # This file is the canonical link between a recording and its transcript.
    # Match key: call_sid  (transcript filename also contains call_sid)
    try:
        from pathlib import Path as _Path
        _Path(RECORDINGS_DIR).mkdir(parents=True, exist_ok=True)
        rec_meta = {
            "call_sid":        call_uuid,
            "recording_id":    recording_id,
            "recording_url":   record_url,
            "duration_sec":    duration,
            "ts":              ts_str,
            "transcript_file": transcript_path,
        }
        rec_path = f"{RECORDINGS_DIR}/{call_uuid}.json"
        with open(rec_path, "w", encoding="utf-8") as fh:
            fh.write(_json.dumps(rec_meta, ensure_ascii=False, indent=2) + "\n")
        log.info("Recording metadata saved: %s", rec_path)
    except OSError as exc:
        log.warning("Could not write recording metadata: %s", exc)

    # ── 3. Push recording fields to push_data webhook ─────────────────────────
    from config import CALL_SUMMARY_WEBHOOK_URL
    from call_webhook import push_call_summary_webhook
    if CALL_SUMMARY_WEBHOOK_URL and call_uuid:
        rec_start_ms  = data.get("recording_start_ms", "")
        rec_end_ms    = data.get("recording_end_ms",   "")
        rec_dur_ms    = data.get("recording_duration_ms", "")
        wh_rec = {
            "call_sid":            call_uuid,
            "recording_url":       record_url,
            "recording_id":        recording_id,
            "duration_sec":        duration,
            "recording_start_ms":  rec_start_ms,
            "recording_end_ms":    rec_end_ms,
            "recording_duration_ms": rec_dur_ms,
            "ts":                  ts_str,
            "event":               "recording_ready",
        }
        await push_call_summary_webhook(CALL_SUMMARY_WEBHOOK_URL, wh_rec)
        log.info("Recording fields pushed to push_data webhook — call=%s", call_uuid)

    return JSONResponse({"status": "ok"})


# ─────────────────────────────────────────────────────────────────────────────
# Helper: push combined recording URL to the n8n webhook
# Payload is intentionally minimal — just the audio link + call identity.
# The recording is a single combined MP3 (both caller + bot, record_channel=both).
# ─────────────────────────────────────────────────────────────────────────────
async def _push_audio_transcript(
    call_sid: str,
    recording_url: str,
    duration: str,
    ts_str: str,
    transcript_path: str,
) -> None:
    from pathlib import Path as _Path

    # Build a stable filename for n8n binary storage.
    # If transcript is known, reuse its stem so calls are easy to correlate.
    stem = _Path(transcript_path).stem if transcript_path else call_sid
    file_name = f"{stem}_combined.mp3"
    try:
        # 1) Download Plivo-hosted combined MP3
        dl = await _http_client.get(recording_url, timeout=60.0)
        dl.raise_for_status()
        audio_bytes = dl.content
        if not audio_bytes:
            log.warning("audio_and_transcripts webhook skipped: empty audio for call=%s", call_sid)
            return

        # 2) Upload binary to n8n as multipart/form-data (same shape as curl --form file=@...)
        form_fields = {
            "call_sid": call_sid,
            "duration_sec": duration,
            "ts": ts_str,
        }
        files = {
            "file": (file_name, audio_bytes, "audio/mpeg"),
        }
        r = await _http_client.post(
            AUDIO_TRANSCRIPT_WEBHOOK_URL,
            data=form_fields,
            files=files,
            timeout=90.0,
        )
        log.info(
            "audio_and_transcripts webhook upload → HTTP %d call=%s bytes=%d",
            r.status_code,
            call_sid,
            len(audio_bytes),
        )
    except Exception as exc:
        log.warning("audio_and_transcripts webhook failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Transcript viewer — list + detail pages  (beautiful dark UI)
# ─────────────────────────────────────────────────────────────────────────────

# Outcome → (label, bg-color, text-color)  — light-mode palette
_OUTCOME_STYLE: dict[str, tuple[str, str, str]] = {
    "ptp_confirmed":          ("PTP",            "#dcfce7", "#15803d"),
    "ptp":                    ("PTP",            "#dcfce7", "#15803d"),
    "payment_today_confirmed":("Paid Today",     "#d1fae5", "#065f46"),
    "payment_confirm":        ("Paid Today",     "#d1fae5", "#065f46"),
    "partial_confirmed":      ("Partial",        "#dbeafe", "#1d4ed8"),
    "partial":                ("Partial",        "#dbeafe", "#1d4ed8"),
    "cannot_pay_callback":    ("Cannot Pay",     "#fee2e2", "#b91c1c"),
    "cannot_pay":             ("Cannot Pay",     "#fee2e2", "#b91c1c"),
    "callback_scheduled":     ("Callback",       "#ede9fe", "#6d28d9"),
    "callback":               ("Callback",       "#ede9fe", "#6d28d9"),
    "already_paid_noted":     ("Already Paid",   "#fef9c3", "#854d0e"),
    "already_paid":           ("Already Paid",   "#fef9c3", "#854d0e"),
    "deceased":               ("Deceased",       "#f1f5f9", "#475569"),
    "no_response":            ("No Response",    "#f1f5f9", "#64748b"),
    "silence_timeout":        ("No Response",    "#f1f5f9", "#64748b"),
    "orchestrator_failure":   ("Error",          "#fee2e2", "#b91c1c"),
    "carrier_disconnect":     ("Disconnected",   "#f1f5f9", "#64748b"),
}

def _outcome_badge(raw: str, size: str = "sm") -> str:
    """Return a styled HTML badge for an outcome/hangup_reason string."""
    key = (raw or "").lower().replace(" ", "_")
    label, bg, fg = _OUTCOME_STYLE.get(key, ("❓ " + (raw or "Unknown"), "#1e293b", "#64748b"))
    pad = "3px 10px" if size == "lg" else "2px 9px"
    fs  = "13px"     if size == "lg" else "11px"
    return (
        f'<span style="background:{bg};color:{fg};padding:{pad};border-radius:20px;'
        f'font-size:{fs};font-weight:600;white-space:nowrap">{label}</span>'
    )

def _fmt_ts_from_filename(stem: str) -> str:
    """Parse YYYYMMDD_HHMMSS from transcript filename stem."""
    import re as _re
    m = _re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", stem)
    if m:
        y, mo, d, h, mi, s = m.groups()
        months = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
        try:
            mn = months[int(mo) - 1]
        except Exception:
            mn = mo
        return f"{d} {mn} {y}  {h}:{mi}:{s}"
    return stem[:19]

def _parse_transcript(path: "Path") -> dict:  # type: ignore[name-defined]
    """Read a JSONL transcript and return a summary dict for list/detail views."""
    import json as _json
    out = {
        "call_sid": "", "phone": "", "customer": "", "loan_id": "", "emi": "",
        "emi_date": "", "hangup_reason": "", "state": "", "summary": "",
        "recording_url": "", "duration": "", "events": [],
    }
    try:
        for ln in path.read_text(encoding="utf-8").strip().splitlines():
            try:
                row = _json.loads(ln)
            except Exception:
                continue
            out["events"].append(row)
            evt = row.get("event", "")
            if evt == "call_start":
                out["call_sid"]  = row.get("sid", "") or out["call_sid"]
                out["phone"]     = row.get("phone", "")    or out["phone"]
                out["customer"]  = row.get("customer", "") or out["customer"]
                out["loan_id"]   = row.get("loan_id", "")  or out["loan_id"]
                out["emi"]       = row.get("emi", "")      or out["emi"]
                out["emi_date"]  = row.get("emi_date", "") or out["emi_date"]
            elif evt == "hangup":
                out["hangup_reason"] = row.get("reason", "") or out["hangup_reason"]
                out["state"]         = row.get("state", "")  or out["state"]
            elif evt == "call_summary":
                out["summary"] = row.get("summary", "") or out["summary"]
            elif evt == "recording_ready":
                out["recording_url"] = row.get("recording_url", "") or out["recording_url"]
                out["duration"]      = str(row.get("duration_sec", "")) or out["duration"]
            # Fallback for call_sid from any event
            if not out["call_sid"]:
                out["call_sid"] = row.get("sid", "")
    except OSError:
        pass
    return out

_TRANSCRIPT_CSS = """\
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,Arial,sans-serif;background:#f4f6f9;color:#1e293b;min-height:100vh}
a{color:#2563eb;text-decoration:none}
a:hover{text-decoration:underline}
"""

@app.get("/transcripts", response_class=HTMLResponse)
async def transcripts_list() -> HTMLResponse:
    """Beautiful list of all call transcripts, newest first."""
    from pathlib import Path as _Path

    files = sorted(
        _Path(TRANSCRIPTS_DIR).glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if _Path(TRANSCRIPTS_DIR).exists() else []

    cards_html = ""
    for f in files:
        meta = _parse_transcript(f)
        dt_str   = _fmt_ts_from_filename(f.stem)
        outcome  = meta["state"] or meta["hangup_reason"]
        badge    = _outcome_badge(outcome)
        customer = meta["customer"] or "—"
        phone    = meta["phone"]    or "—"
        loan_id  = meta["loan_id"]  or "—"
        emi      = f"₹{meta['emi']}" if meta["emi"] else "—"
        duration = f"{meta['duration']}s" if meta["duration"] else ""
        summary  = meta["summary"][:110] + "…" if len(meta.get("summary","")) > 110 else meta.get("summary","")

        rec_cell = (
            '<span class="rec-dot" title="Recording available"></span>'
            f'<span class="dur">{duration}</span>'
            if meta["recording_url"] else
            f'<span class="dur">{duration}</span>' if duration else
            '<span style="color:#e2e8f0">—</span>'
        )

        cards_html += f"""
          <tr onclick="location.href='/transcripts/{f.name}'">
            <td class="dt">{dt_str}</td>
            <td class="name">{customer}</td>
            <td class="mono">{phone}</td>
            <td class="mono">{loan_id}</td>
            <td>{emi}</td>
            <td>{badge}</td>
            <td class="summary-col" title="{summary}">{summary or '—'}</td>
            <td>{rec_cell}</td>
          </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Call Transcripts — Aditi</title>
  <style>
    {_TRANSCRIPT_CSS}
    .wrap{{max-width:1050px;margin:0 auto;padding:32px 24px}}
    /* top bar */
    .topbar{{display:flex;align-items:center;justify-content:space-between;
             margin-bottom:24px;flex-wrap:wrap;gap:12px}}
    .topbar-left{{display:flex;align-items:baseline;gap:12px}}
    h1{{font-size:20px;font-weight:600;color:#0f172a;letter-spacing:-.3px}}
    .count{{font-size:13px;color:#64748b}}
    .topbar-right{{display:flex;gap:8px}}
    .btn{{font-size:12px;color:#475569;padding:5px 12px;border-radius:6px;
          border:1px solid #cbd5e1;background:#fff}}
    .btn:hover{{background:#f1f5f9;text-decoration:none;color:#1e293b}}
    /* table */
    .tbl-wrap{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden}}
    table{{width:100%;border-collapse:collapse;font-size:13px}}
    thead th{{background:#f8fafc;padding:10px 14px;text-align:left;font-size:11px;
              font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.05em;
              border-bottom:1px solid #e2e8f0}}
    tbody tr{{border-bottom:1px solid #f1f5f9;cursor:pointer;transition:background .1s}}
    tbody tr:last-child{{border-bottom:none}}
    tbody tr:hover{{background:#f8fafc}}
    td{{padding:12px 14px;color:#334155;vertical-align:middle}}
    td.dt{{font-size:12px;color:#64748b;white-space:nowrap;font-family:monospace}}
    td.name{{font-weight:500;color:#0f172a}}
    td.mono{{font-family:monospace;font-size:12px;color:#475569}}
    td.summary-col{{font-size:12px;color:#64748b;max-width:260px;
                    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
    .badge{{display:inline-block;padding:2px 9px;border-radius:4px;
            font-size:11px;font-weight:600;white-space:nowrap}}
    .rec-dot{{display:inline-block;width:7px;height:7px;border-radius:50%;
              background:#22c55e;margin-right:5px;vertical-align:middle}}
    .dur{{font-size:11px;color:#94a3b8}}
    .empty{{text-align:center;padding:60px 20px;color:#94a3b8;font-size:14px}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="topbar-left">
        <h1>Call Transcripts</h1>
        <span class="count">{len(files)} record{'s' if len(files) != 1 else ''}</span>
      </div>
      <div class="topbar-right">
        <a class="btn" href="/logs">Logs</a>
        <a class="btn" href="/health">Health</a>
      </div>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>Date &amp; Time</th>
            <th>Customer</th>
            <th>Phone</th>
            <th>Loan ID</th>
            <th>EMI Due</th>
            <th>Outcome</th>
            <th>Summary</th>
            <th>Rec.</th>
          </tr>
        </thead>
        <tbody>
          {'<tr><td colspan="8" class="empty">No transcripts found. Transcripts appear here after calls complete.</td></tr>' if not files else cards_html}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/transcripts/{filename}", response_class=HTMLResponse)
async def transcript_detail(filename: str) -> HTMLResponse:
    """Detailed call view with chat bubbles, metadata header, and audio player."""
    from pathlib import Path as _Path

    if not filename.endswith(".jsonl") or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = _Path(TRANSCRIPTS_DIR) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")

    meta = _parse_transcript(path)
    events   = meta["events"]
    dt_str   = _fmt_ts_from_filename(path.stem)
    outcome  = meta["state"] or meta["hangup_reason"]
    badge_lg = _outcome_badge(outcome, size="lg")

    # ── Build chat bubbles ─────────────────────────────────────────────────────
    bubbles_html = ""
    summary_fields: dict = {}

    for row in events:
        evt  = row.get("event", "")
        text = (row.get("text") or "").strip()
        ts   = row.get("ts", "")[:19].replace("T", " ")

        if evt == "call_start":
            bubbles_html += (
                f'<div class="sys-pill">📞 Call connected &nbsp;·&nbsp; '
                f'<span class="mono">{ts}</span></div>'
            )

        elif evt == "bot" and text:
            safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            bubbles_html += f"""
            <div class="msg-row agent-row">
              <div class="av av-agent" title="Aditi (AI)">AI</div>
              <div class="msg-wrap">
                <div class="bubble bubble-agent">{safe}</div>
                <div class="msg-ts">{ts}</div>
              </div>
            </div>"""

        elif evt in ("user", "user_turn") and text:
            safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            bubbles_html += f"""
            <div class="msg-row user-row">
              <div class="msg-wrap">
                <div class="bubble bubble-user">{safe}</div>
                <div class="msg-ts" style="text-align:right">{ts}</div>
              </div>
              <div class="av av-user" title="Customer">👤</div>
            </div>"""

        elif evt == "hangup":
            reason = row.get("reason", "")
            bubbles_html += (
                f'<div class="sys-pill">📵 Call ended &nbsp;·&nbsp; '
                f'<span class="mono">{_outcome_badge(reason)}</span>'
                f'&nbsp;·&nbsp; <span class="mono">{ts}</span></div>'
            )

        elif evt == "call_summary":
            summary_fields = {k: v for k, v in row.items()
                               if k not in ("ts", "event", "state", "sid") and v not in (None, "")}

        elif evt in ("recording_uploaded", "recording_ready"):
            pass   # handled separately via meta dict

    # ── Summary card rows ──────────────────────────────────────────────────────
    _FIELD_LABELS = {
        "summary":           "Summary",
        "target_date":       "Target Date",
        "partial_amount":    "Partial Amount",
        "cannot_pay_reason": "Cannot Pay Reason",
        "already_paid_date": "Already Paid Date",
        "already_paid_mode": "Payment Mode",
        "callback_time":     "Callback Time",
    }
    sum_rows_html = ""
    for k, v in summary_fields.items():
        label = _FIELD_LABELS.get(k, k.replace("_", " ").title())
        safe_v = str(v).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        sum_rows_html += (
            f"<tr><td class='sum-key'>{label}</td>"
            f"<td class='sum-val'>{safe_v}</td></tr>"
        )

    summary_card = f"""
    <div class="sum-card">
      <div class="sum-title">Call Summary</div>
      <table class="sum-tbl"><tbody>{sum_rows_html}</tbody></table>
    </div>""" if sum_rows_html else ""

    # ── Recording player ───────────────────────────────────────────────────────
    recording_card = ""
    if meta["recording_url"]:
        dur_label = f"{meta['duration']}s" if meta["duration"] else ""
        recording_card = f"""
    <div class="rec-card">
      <div class="rec-title">Call Recording
        <span style="font-weight:400;color:#94a3b8;margin-left:8px">{dur_label}</span>
      </div>
      <audio controls>
        <source src="{meta['recording_url']}" type="audio/mpeg">
      </audio>
      <a class="dl-link" href="{meta['recording_url']}" target="_blank">Download MP3</a>
    </div>"""

    # ── Metadata header fields ─────────────────────────────────────────────────
    def _hf(label: str, value: str, mono: bool = False) -> str:
        if not value or value == "—":
            return ""
        cls = "mf-mono" if mono else ""
        return (
            f'<div class="mf"><span class="mf-lbl">{label}</span>'
            f'<span class="mf-val {cls}">{value}</span></div>'
        )

    header_fields = (
        _hf("Customer",  meta["customer"] or "—")
        + _hf("Phone",   meta["phone"]    or "—",  mono=True)
        + _hf("Loan ID", meta["loan_id"]  or "—",  mono=True)
        + _hf("EMI Due", (f"₹{meta['emi']}" if meta["emi"] else "—"))
        + _hf("Call SID",meta["call_sid"] or "—",  mono=True)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Call Detail — Aditi</title>
  <style>
    {_TRANSCRIPT_CSS}
    .wrap{{max-width:800px;margin:0 auto;padding:28px 20px 60px}}
    .back{{font-size:13px;color:#475569;display:inline-flex;align-items:center;gap:4px;
           margin-bottom:20px}}
    .back:hover{{color:#1e293b;text-decoration:none}}
    /* header card */
    .hdr{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;
          padding:18px 20px;margin-bottom:14px}}
    .hdr-top{{display:flex;align-items:center;justify-content:space-between;
              flex-wrap:wrap;gap:10px;margin-bottom:14px}}
    .hdr-dt{{font-size:12px;color:#64748b;font-family:monospace}}
    .hdr-meta{{display:flex;flex-wrap:wrap;gap:6px 28px}}
    .mf{{display:flex;flex-direction:column;gap:1px}}
    .mf-lbl{{font-size:10px;text-transform:uppercase;letter-spacing:.05em;
              color:#94a3b8;font-weight:600}}
    .mf-val{{font-size:13px;color:#334155;font-weight:500}}
    .mf-mono{{font-family:monospace;font-size:12px;font-weight:400}}
    /* section label */
    .sec-label{{font-size:11px;font-weight:600;text-transform:uppercase;
                letter-spacing:.06em;color:#94a3b8;margin:18px 0 8px}}
    /* recording */
    .rec-card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;
               margin-bottom:14px}}
    .rec-title{{font-size:12px;font-weight:600;color:#475569;margin-bottom:10px}}
    audio{{width:100%;accent-color:#2563eb}}
    .dl-link{{font-size:11px;color:#64748b;margin-top:6px;display:block}}
    /* chat */
    .chat{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;
           padding:20px;display:flex;flex-direction:column;gap:14px}}
    .msg-row{{display:flex;align-items:flex-end;gap:8px}}
    .agent-row{{justify-content:flex-start}}
    .user-row{{justify-content:flex-end}}
    .av{{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
         justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;border:1px solid #e2e8f0}}
    .av-agent{{background:#eff6ff;color:#1d4ed8}}
    .av-user{{background:#f0fdf4;color:#15803d}}
    .msg-wrap{{display:flex;flex-direction:column;max-width:75%}}
    .bubble{{padding:10px 14px;border-radius:12px;font-size:13.5px;
             line-height:1.6;word-break:break-word;border:1px solid transparent}}
    .bubble-agent{{background:#eff6ff;color:#1e3a8a;border-color:#dbeafe;
                   border-bottom-left-radius:3px}}
    .bubble-user{{background:#f0fdf4;color:#14532d;border-color:#dcfce7;
                  border-bottom-right-radius:3px}}
    .msg-ts{{font-size:10px;color:#cbd5e1;margin-top:3px;padding:0 2px}}
    .sys-pill{{text-align:center;font-size:11px;color:#94a3b8;background:#f8fafc;
               border:1px solid #e2e8f0;padding:4px 14px;border-radius:20px;
               align-self:center;margin:2px auto}}
    /* summary table */
    .sum-card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;
               padding:14px 16px;margin-top:14px}}
    .sum-title{{font-size:12px;font-weight:600;color:#475569;margin-bottom:10px}}
    .sum-tbl{{width:100%;border-collapse:collapse;font-size:13px}}
    .sum-tbl td{{padding:8px 10px;border-bottom:1px solid #f1f5f9;vertical-align:top}}
    .sum-tbl tr:last-child td{{border-bottom:none}}
    .sum-key{{color:#64748b;width:38%;font-size:12px;font-weight:500}}
    .sum-val{{color:#334155;line-height:1.5}}
    .badge{{display:inline-block;padding:2px 9px;border-radius:4px;
            font-size:11px;font-weight:600}}
    .mono{{font-family:monospace;font-size:12px}}
  </style>
</head>
<body>
  <div class="wrap">
    <a class="back" href="/transcripts">&#8592; All Calls</a>

    <div class="hdr">
      <div class="hdr-top">
        <span class="hdr-dt">{dt_str}</span>
        {badge_lg}
      </div>
      <div class="hdr-meta">
        {header_fields}
      </div>
    </div>

    {recording_card}

    <div class="sec-label">Conversation</div>
    <div class="chat">
      {bubbles_html or '<div class="sys-pill">No conversation events recorded.</div>'}
    </div>

    {summary_card}
  </div>
</body>
</html>"""
    return HTMLResponse(html)


# ─────────────────────────────────────────────────────────────────────────────
# Live log viewer
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/logs", response_class=HTMLResponse)
async def logs_page() -> HTMLResponse:
    """
    Live in-memory log viewer. Shows last 600 log lines + server state.
    Hit /logs in your browser to diagnose issues without server SSH access.
    """
    import json as _json
    from pathlib import Path as _Path
    from datetime import datetime, timezone
    from session import pending_ctx, recording_pending
    from config import CALL_SUMMARY_WEBHOOK_URL, AUDIO_TRANSCRIPT_WEBHOOK_URL

    lines = _log_buffer.get_lines()

    # ── colour map ────────────────────────────────────────────────────────────
    _colours = {
        "DEBUG":    "#475569",
        "INFO":     "#94a3b8",
        "WARNING":  "#fbbf24",
        "ERROR":    "#f87171",
        "CRITICAL": "#ef4444",
    }

    # Keywords that get highlighted regardless of level
    _hl = {
        "webhook OK":           "#4ade80",
        "webhook failed":       "#f87171",
        "webhook":              "#818cf8",
        "CALL_VARS":            "#4ade80",
        "call vars error":      "#f87171",
        "Hangup":               "#fbbf24",
        "carrier_disconnect":   "#fbbf24",
        "unexpected_disconnect":"#f87171",
        "ptp_confirmed":        "#4ade80",
        "cannot_pay":           "#fb923c",
        "STT":                  "#67e8f9",
        "TTS":                  "#a78bfa",
        "Plivo":                "#60a5fa",
        "rekeyed":              "#4ade80",
        "pending_ctx":          "#fbbf24",
        "audio_and_transcripts":"#818cf8",
        "post_data":            "#818cf8",
        "Outbound call":        "#4ade80",
    }

    def _row(entry: dict) -> str:
        msg  = entry["msg"]
        lvl  = entry["level"]
        col  = _colours.get(lvl, "#94a3b8")
        # highlight keywords
        for kw, kc in _hl.items():
            if kw.lower() in msg.lower():
                col = kc
                break
        safe = (msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        return f'<div style="color:{col};padding:2px 0;border-bottom:1px solid #1e293b;font-size:12px;font-family:monospace;white-space:pre-wrap;word-break:break-all">{safe}</div>'

    log_html = "".join(_row(e) for e in reversed(lines)) if lines else \
        '<div style="color:#475569;padding:20px;text-align:center">No logs captured yet — make a call first.</div>'

    # ── recent transcripts ────────────────────────────────────────────────────
    recent_files = sorted(
        _Path(TRANSCRIPTS_DIR).glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )[:8] if _Path(TRANSCRIPTS_DIR).exists() else []

    tx_rows = ""
    for f in recent_files:
        call_sid = state = hangup_reason = ""
        try:
            for ln in f.read_text(encoding="utf-8").strip().splitlines():
                row = _json.loads(ln)
                if not call_sid:
                    call_sid = row.get("sid", "")
                if row.get("event") == "hangup":
                    hangup_reason = row.get("reason", "")
                    state = row.get("state", "")
        except Exception:
            pass
        outcome = state or hangup_reason or "—"
        tx_rows += f"<tr><td style='color:#94a3b8;font-size:11px'>{f.name[:38]}</td><td style='color:#64748b;font-size:11px;font-family:monospace'>{call_sid[:20]}</td><td><span style='background:#312e81;color:#a5b4fc;padding:2px 8px;border-radius:20px;font-size:10px'>{outcome}</span></td></tr>"

    tx_table = f"""
    <table style="width:100%;border-collapse:collapse;background:#111827;border-radius:8px;overflow:hidden;margin-top:8px">
      <thead><tr>
        <th style="background:#1e293b;padding:8px 12px;text-align:left;font-size:10px;color:#475569;text-transform:uppercase">File</th>
        <th style="background:#1e293b;padding:8px 12px;text-align:left;font-size:10px;color:#475569;text-transform:uppercase">Call SID</th>
        <th style="background:#1e293b;padding:8px 12px;text-align:left;font-size:10px;color:#475569;text-transform:uppercase">Outcome</th>
      </tr></thead>
      <tbody>{tx_rows or '<tr><td colspan="3" style="text-align:center;padding:20px;color:#334155">No transcripts yet</td></tr>'}</tbody>
    </table>""" if recent_files else '<p style="color:#334155;font-size:13px">No transcript files found.</p>'

    # ── server state ──────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    state_rows = [
        ("Time (UTC)", now),
        ("pending_ctx keys", str(len(pending_ctx)) + (" — " + ", ".join(list(pending_ctx.keys())[:5]) if pending_ctx else " (empty ✅)")),
        ("recording_pending keys", str(len(recording_pending))),
        ("Transcript files", str(len(list(_Path(TRANSCRIPTS_DIR).glob("*.jsonl")))) if _Path(TRANSCRIPTS_DIR).exists() else "0"),
        ("push_data webhook", CALL_SUMMARY_WEBHOOK_URL or "(not set ⚠️)"),
        ("audio webhook", AUDIO_TRANSCRIPT_WEBHOOK_URL or "(not set ⚠️)"),
        ("Log lines captured", str(len(lines))),
    ]
    state_html = "".join(
        f"<tr><td style='color:#64748b;font-size:12px;padding:6px 10px;border-bottom:1px solid #1e293b;width:35%'>{k}</td>"
        f"<td style='font-size:12px;padding:6px 10px;border-bottom:1px solid #1e293b;word-break:break-all'>{v}</td></tr>"
        for k, v in state_rows
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="refresh" content="15">
  <title>Aditi — Live Logs</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#0a0f1e;color:#e2e8f0;padding:20px}}
    h2{{font-size:14px;font-weight:600;color:#94a3b8;margin:20px 0 8px;text-transform:uppercase;letter-spacing:.06em}}
    .card{{background:#111827;border-radius:10px;padding:14px 16px}}
    .header{{display:flex;align-items:center;gap:12px;margin-bottom:20px}}
    .header h1{{font-size:18px;font-weight:700;color:#f1f5f9}}
    .badge{{background:#1e293b;color:#64748b;font-size:11px;padding:2px 10px;border-radius:20px}}
    .refresh{{margin-left:auto;font-size:12px;color:#818cf8;text-decoration:none;border:1px solid #312e81;padding:4px 12px;border-radius:6px}}
    .log-box{{background:#0a0f1e;border:1px solid #1e293b;border-radius:8px;padding:12px;max-height:520px;overflow-y:auto}}
    a{{color:#818cf8;text-decoration:none}}
  </style>
</head>
<body>
  <div class="header">
    <h1>🔍 Aditi — Live Logs</h1>
    <span class="badge">auto-refresh 15s</span>
    <a class="refresh" href="/logs">⟳ Refresh now</a>
  </div>

  <h2>Server State</h2>
  <div class="card">
    <table style="width:100%;border-collapse:collapse">
      <tbody>{state_html}</tbody>
    </table>
  </div>

  <h2>Recent Transcripts (last 8)</h2>
  <div class="card">{tx_table}</div>

  <h2>Log Output (newest first · last {len(lines)} lines)</h2>
  <div class="card">
    <div class="log-box">{log_html}</div>
  </div>

  <p style="margin-top:14px;font-size:11px;color:#334155">
    Colour key:
    <span style="color:#4ade80">■ success/webhook OK</span> &nbsp;
    <span style="color:#f87171">■ error</span> &nbsp;
    <span style="color:#fbbf24">■ warning/hangup</span> &nbsp;
    <span style="color:#818cf8">■ webhook call</span> &nbsp;
    <span style="color:#67e8f9">■ STT</span> &nbsp;
    <span style="color:#a78bfa">■ TTS</span>
  </p>
</body>
</html>"""
    return HTMLResponse(html)
