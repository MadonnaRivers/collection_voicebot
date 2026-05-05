"""
viewer.py — Standalone transcript viewer.

Runs on port 8050 (separate from the main bot on 5050).
Start with:  npm start   OR   python viewer.py

Routes:
  GET /                          → redirect to /transcripts
  GET /transcripts               → list all calls
  GET /transcripts/<file>.jsonl  → chat-bubble view for a single call
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

load_dotenv()

TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR", "transcripts")
VIEWER_PORT     = int(os.getenv("VIEWER_PORT", "8050"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("viewer")

app = FastAPI(title="Aditi — Transcript Viewer")


# ── redirect / → /transcripts ────────────────────────────────────────────────
@app.get("/")
async def root():
    return RedirectResponse(url="/transcripts")


# ── list all calls ────────────────────────────────────────────────────────────
@app.get("/transcripts", response_class=HTMLResponse)
async def transcripts_list() -> HTMLResponse:
    files = sorted(
        Path(TRANSCRIPTS_DIR).glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if Path(TRANSCRIPTS_DIR).exists() else []

    rows_html = ""
    for f in files:
        call_sid = hangup_reason = recording_url = state = phone = ""
        try:
            for ln in f.read_text(encoding="utf-8").strip().splitlines():
                try:
                    row = json.loads(ln)
                    if not call_sid:
                        call_sid = row.get("sid", "")
                    if row.get("event") == "hangup":
                        hangup_reason = row.get("reason", "")
                        state         = row.get("state", "")
                    if row.get("event") == "recording_ready":
                        recording_url = row.get("recording_url", "")
                    if row.get("event") == "call_summary":
                        phone = row.get("phone_number", "")
                except Exception:
                    pass
        except OSError:
            pass

        outcome = state or hangup_reason or "—"
        badge_color = {
            "ptp": "#166534", "payment_confirm": "#14532d",
            "already_paid": "#1e3a5f", "partial": "#713f12",
            "cannot_pay": "#7f1d1d", "callback": "#312e81",
            "no_response": "#374151",
        }.get(outcome, "#1e293b")

        rec_badge = (
            f'<a href="{recording_url}" target="_blank" class="rec-link">▶ play</a>'
            if recording_url else
            '<span class="no-rec">no recording</span>'
        )
        # Human-readable date from filename  e.g. 20250505_142301_...
        name_parts = f.stem.split("_")
        date_str = ""
        if len(name_parts) >= 2:
            try:
                d, t = name_parts[0], name_parts[1]
                date_str = f"{d[6:8]}/{d[4:6]}/{d[:4]} {t[:2]}:{t[2:4]}:{t[4:]}"
            except Exception:
                pass

        rows_html += f"""
        <tr onclick="location.href='/transcripts/{f.name}'">
          <td class="dt">{date_str}</td>
          <td class="sid">{call_sid[:22] or '—'}</td>
          <td>{phone or '—'}</td>
          <td><span class="badge" style="background:{badge_color}">{outcome}</span></td>
          <td>{rec_badge}</td>
        </tr>"""

    empty = '<tr><td colspan="5" class="empty">No transcripts yet. Make a call first.</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Aditi Transcripts</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0 }}
    body {{
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0a0f1e;
      color: #cbd5e1;
      min-height: 100vh;
      padding: 32px 24px;
    }}
    .header {{
      display: flex;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 24px;
    }}
    h1 {{ font-size: 20px; font-weight: 700; color: #f1f5f9 }}
    .count {{
      font-size: 13px;
      color: #64748b;
      background: #1e293b;
      padding: 2px 10px;
      border-radius: 20px;
    }}
    .refresh {{
      margin-left: auto;
      font-size: 12px;
      color: #818cf8;
      cursor: pointer;
      text-decoration: none;
      border: 1px solid #312e81;
      padding: 4px 12px;
      border-radius: 6px;
    }}
    .refresh:hover {{ background: #1e1b4b }}
    .wrap {{ overflow-x: auto }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #111827;
      border-radius: 12px;
      overflow: hidden;
    }}
    th {{
      background: #1e293b;
      padding: 10px 16px;
      text-align: left;
      font-size: 11px;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: .06em;
      font-weight: 600;
    }}
    td {{
      padding: 11px 16px;
      font-size: 13px;
      border-bottom: 1px solid #1e293b;
      vertical-align: middle;
    }}
    tr:last-child td {{ border-bottom: none }}
    tbody tr {{ cursor: pointer; transition: background .12s }}
    tbody tr:hover td {{ background: #162032 }}
    .dt  {{ color: #94a3b8; font-size: 12px; white-space: nowrap }}
    .sid {{ color: #475569; font-size: 11px; font-family: monospace }}
    .badge {{
      color: #e2e8f0;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      white-space: nowrap;
    }}
    .rec-link {{
      color: #4ade80;
      font-size: 12px;
      text-decoration: none;
      border: 1px solid #166534;
      padding: 3px 10px;
      border-radius: 6px;
    }}
    .rec-link:hover {{ background: #14532d }}
    .no-rec {{ color: #334155; font-size: 11px }}
    .empty {{ text-align: center; padding: 60px; color: #334155; font-size: 14px }}
    a {{ color: inherit }}
  </style>
</head>
<body>
  <div class="header">
    <h1>📋 Aditi — Call Transcripts</h1>
    <span class="count">{len(files)} calls</span>
    <a class="refresh" href="/transcripts">⟳ Refresh</a>
  </div>
  <div class="wrap">
    <table>
      <thead>
        <tr>
          <th>Date / Time</th>
          <th>Call SID</th>
          <th>Phone</th>
          <th>Outcome</th>
          <th>Recording</th>
        </tr>
      </thead>
      <tbody>
        {rows_html if files else empty}
      </tbody>
    </table>
  </div>
</body>
</html>"""
    return HTMLResponse(html)


# ── single call chat view ─────────────────────────────────────────────────────
@app.get("/transcripts/{filename}", response_class=HTMLResponse)
async def transcript_detail(filename: str) -> HTMLResponse:
    if not filename.endswith(".jsonl") or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = Path(TRANSCRIPTS_DIR) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")

    events = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        try:
            events.append(json.loads(line))
        except Exception:
            pass

    call_sid = hangup_reason = state = recording_url = duration = phone = ""
    summary_fields: dict = {}
    bubbles_html = ""

    for row in events:
        evt  = row.get("event", "")
        text = (row.get("text") or "").strip()
        ts   = row.get("ts", "")[:19].replace("T", " ")

        if evt == "call_start":
            call_sid = row.get("sid", "")
            bubbles_html += f'<div class="sys-evt">📞 Call started &nbsp;·&nbsp; {ts}</div>'

        elif evt == "bot" and text:
            bubbles_html += f"""
            <div class="row agent-row">
              <div class="av agent-av">AI</div>
              <div class="msg">
                <div class="bubble agent-bubble">{text}</div>
                <div class="ts">{ts}</div>
              </div>
            </div>"""

        elif evt in ("user", "user_turn") and text:
            bubbles_html += f"""
            <div class="row user-row">
              <div class="msg">
                <div class="bubble user-bubble">{text}</div>
                <div class="ts right">{ts}</div>
              </div>
              <div class="av user-av">👤</div>
            </div>"""

        elif evt == "hangup":
            hangup_reason = row.get("reason", "")
            state         = row.get("state", "")
            bubbles_html += f'<div class="sys-evt">📵 Ended &nbsp;·&nbsp; {hangup_reason} &nbsp;·&nbsp; {ts}</div>'

        elif evt == "call_summary":
            summary_fields = {k: v for k, v in row.items()
                              if k not in ("ts", "event", "state", "sid")}
            phone = str(summary_fields.pop("phone_number", "") or "")

        elif evt == "recording_ready":
            recording_url = row.get("recording_url", "")
            duration      = str(row.get("duration_sec", ""))

    # summary table
    sum_rows = "".join(
        f"<tr><td class='sk'>{k}</td><td>{v}</td></tr>"
        for k, v in summary_fields.items()
        if v not in (None, "", False)
    )

    recording_block = f"""
    <div class="card" id="recording-card">
      <div class="card-title">🎙️ Recording &nbsp;<span class="dim">{duration}s · combined (both sides)</span></div>
      <audio controls>
        <source src="{recording_url}" type="audio/mpeg">
        <a href="{recording_url}" target="_blank">Download MP3</a>
      </audio>
    </div>""" if recording_url else '<div class="card dim-card">🎙️ Recording not available yet</div>'

    summary_block = f"""
    <div class="card">
      <div class="card-title">📋 Call Summary</div>
      <table class="stbl"><tbody>{sum_rows}</tbody></table>
    </div>""" if sum_rows else ""

    # parse a nice date from filename
    name_parts = Path(filename).stem.split("_")
    date_str = filename
    if len(name_parts) >= 2:
        try:
            d, t = name_parts[0], name_parts[1]
            date_str = f"{d[6:8]}/{d[4:6]}/{d[:4]}  {t[:2]}:{t[2:4]}:{t[4:]}"
        except Exception:
            pass

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{date_str}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0 }}
    body {{
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0a0f1e;
      color: #cbd5e1;
      padding: 20px;
      max-width: 820px;
      margin: 0 auto;
    }}
    .back {{
      display: inline-block;
      color: #818cf8;
      font-size: 13px;
      text-decoration: none;
      margin-bottom: 14px;
    }}
    .back:hover {{ text-decoration: underline }}
    .call-header {{
      margin-bottom: 18px;
    }}
    .call-header h1 {{
      font-size: 16px;
      font-weight: 700;
      color: #f1f5f9;
    }}
    .call-header .meta {{
      font-size: 12px;
      color: #475569;
      margin-top: 4px;
    }}
    .chat-box {{
      background: #111827;
      border-radius: 14px;
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .row {{
      display: flex;
      align-items: flex-end;
      gap: 10px;
    }}
    .agent-row {{ justify-content: flex-start }}
    .user-row  {{ justify-content: flex-end }}
    .av {{
      width: 34px; height: 34px;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 700;
      flex-shrink: 0;
    }}
    .agent-av {{ background: #312e81; color: #a5b4fc }}
    .user-av  {{ background: #14532d; color: #86efac; font-size: 17px }}
    .msg {{ display: flex; flex-direction: column }}
    .bubble {{
      padding: 11px 15px;
      border-radius: 18px;
      max-width: 560px;
      font-size: 14px;
      line-height: 1.65;
      word-break: break-word;
    }}
    .agent-bubble {{
      background: #1e3a5f;
      color: #bfdbfe;
      border-bottom-left-radius: 4px;
    }}
    .user-bubble {{
      background: #14532d;
      color: #dcfce7;
      border-bottom-right-radius: 4px;
    }}
    .ts       {{ font-size: 10px; color: #334155; margin-top: 4px; padding: 0 4px }}
    .ts.right {{ text-align: right }}
    .sys-evt {{
      align-self: center;
      font-size: 11px;
      color: #475569;
      background: #0a0f1e;
      border: 1px solid #1e293b;
      padding: 4px 14px;
      border-radius: 20px;
      margin: 0 auto;
    }}
    .card {{
      background: #111827;
      border-radius: 12px;
      padding: 16px 18px;
      margin-top: 16px;
    }}
    .card-title {{
      font-size: 13px;
      font-weight: 600;
      color: #94a3b8;
      margin-bottom: 10px;
    }}
    audio {{
      width: 100%;
      margin-top: 4px;
      border-radius: 8px;
      height: 40px;
    }}
    .stbl {{ width: 100%; border-collapse: collapse; font-size: 13px }}
    .stbl td {{ padding: 6px 8px; border-bottom: 1px solid #1e293b }}
    .stbl tr:last-child td {{ border-bottom: none }}
    .sk {{ color: #64748b; width: 40%; font-size: 12px }}
    .dim {{ font-weight: 400; color: #475569; font-size: 12px }}
    .dim-card {{ color: #334155; font-size: 13px }}
    a {{ color: #818cf8; text-decoration: none }}
    a:hover {{ text-decoration: underline }}
  </style>
</head>
<body>
  <a class="back" href="/transcripts">← All calls</a>

  <div class="call-header">
    <h1>{date_str}</h1>
    <div class="meta">
      SID: {call_sid or '—'}
      {f'&nbsp;·&nbsp; 📱 {phone}' if phone else ''}
      &nbsp;·&nbsp; Outcome: <strong>{state or hangup_reason or '—'}</strong>
    </div>
  </div>

  <div class="chat-box">
    {bubbles_html or '<div class="sys-evt">No conversation events found.</div>'}
  </div>

  {recording_block}
  {summary_block}
</body>
</html>"""
    return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn
    log.info("Transcript viewer → http://localhost:%d/transcripts", VIEWER_PORT)
    uvicorn.run(app, host="0.0.0.0", port=VIEWER_PORT, log_level="warning")
