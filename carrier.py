"""
carrier.py — Plivo carrier implementation.

Translates Plivo WebSocket media-stream frames and REST API calls into
the normalised interface used by call_handler.py and routes.py.

Normalised WebSocket event types produced by parse_ws_frame():
  "call_start"  → payload: {"stream_sid": str, "call_sid": str}
  "audio_frame" → payload: {"mulaw": bytes}      8 kHz µ-law PCM
  "mark_ack"    → payload: {}                    (unused — Plivo has no mark events)
  "ignore"      → payload: {}                    unhandled / irrelevant frame

NOTE: Plivo does not emit mark acknowledgement events, so mark_msg() returns None.
send_mark() handles None by treating audio as immediately drained.
"""
from __future__ import annotations

import base64 as _b64
import logging

import httpx

from config import PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER

log = logging.getLogger("aditi")

_BASE = f"https://api.plivo.com/v1/Account/{PLIVO_AUTH_ID}"
_AUTH = (PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)


# ─────────────────────────────────────────────────────────────────────────────
# Outbound call
# ─────────────────────────────────────────────────────────────────────────────
async def make_call(
    to: str,
    webhook_url: str,
    amd_callback_url: str = "",
    amd_detection_time_ms: int = 2000,
    hangup_url: str = "",
    time_limit: int = 0,
) -> str:
    """
    Initiate an outbound call via Plivo REST API.
    Returns the request_uuid (used as a temporary key until CallUUID is known).
    The routes.py /outgoing-call handler rekeys pending_ctx to the real CallUUID.

    If `amd_callback_url` is provided, Plivo runs async AMD (Answering Machine
    Detection) in parallel with the call — humans connect immediately, and
    the detection verdict ("human" / "machine_*") is POSTed to that URL.

    If `hangup_url` is provided, Plivo POSTs the final call status there when
    the call ends — including no-answer / busy / rejected cases that never
    connected (no media-stream, no transcript otherwise).
    """
    payload: dict = {
        "from":          PLIVO_PHONE_NUMBER,
        "to":            to,
        "answer_url":    webhook_url,
        "answer_method": "POST",
    }
    if hangup_url:
        payload["hangup_url"]    = hangup_url
        payload["hangup_method"] = "POST"   # Plivo's actual param name (not hangup_url_method)
        log.info("[HANGUP] callback enabled — %s", hangup_url)
    if time_limit and time_limit > 0:
        # Carrier-side hard backstop: Plivo hangs the call up at this duration
        # regardless of what our app is doing. Set above the app's own hard cap.
        payload["time_limit"] = str(time_limit)
        log.info("[PLIVO] carrier time_limit=%ds (hard duration backstop)", time_limit)
    if amd_callback_url:
        payload.update({
            "machine_detection":         "true",
            # async_amd=true makes Plivo run AMD analysis IN PARALLEL with
            # the answer_url WebSocket — no blocking wait for the verdict.
            # The "human" / "machine_*" verdict still arrives async via
            # machine_detection_url. Our _amd_monitor in call_handler
            # already handles a mid-call machine verdict (aborts greeting
            # + switches to voicemail mode). Saves the ~2s the default
            # synchronous AMD would have spent gating the greeting.
            "async_amd":                 "true",
            "machine_detection_time":    str(amd_detection_time_ms),
            "machine_detection_url":     amd_callback_url,
            "machine_detection_method":  "POST",
        })
        log.info("[AMD] enabled (async_amd) — detection_time=%d ms callback=%s",
                 amd_detection_time_ms, amd_callback_url)
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{_BASE}/Call/",
            auth=_AUTH,
            json=payload,
        )
        r.raise_for_status()
        body = r.json()
        log.debug("Plivo make_call response: %s", body)
        request_uuid = body["request_uuid"]
        log.info("Plivo call request_uuid: %s", request_uuid)
        return request_uuid


# ─────────────────────────────────────────────────────────────────────────────
# Recording
# ─────────────────────────────────────────────────────────────────────────────
async def start_recording(call_uuid: str, callback_url: str = "") -> str:
    """
    Start Plivo server-side recording on a live call.
    Returns the Plivo recording_id (empty string on failure).
    callback_url — Plivo will POST the recording URL here when the call ends.
    """
    payload: dict = {
        "time_limit":     3600,
        "record_channel": "both",        # record both legs (caller + bot TTS)
        "file_format":    "mp3",
    }
    if callback_url:
        payload["callback_url"]    = callback_url   # Plivo uses snake_case
        payload["callback_method"] = "POST"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(
                f"{_BASE}/Call/{call_uuid}/Record/",
                auth=_AUTH,
                json=payload,
            )
            log.info(
                "Plivo start_recording → HTTP %d  body=%s",
                r.status_code, r.text[:300],
            )
            if r.status_code >= 400:
                return ""
            body = r.json()
            rec_id = body.get("recording_id", "")
            log.info("[PLIVO] recording started — call=%s recording_id=%s", call_uuid, rec_id)
            return rec_id
        except Exception as exc:
            log.warning("Plivo start_recording error: %s", exc)
            return ""


async def get_recording(recording_id: str) -> dict:
    """
    Fetch recording metadata from Plivo by recording_id.
    Returns dict with: recording_id, record_url, recording_duration, call_uuid, etc.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(
                f"{_BASE}/Recording/{recording_id}/",
                auth=_AUTH,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.warning("Plivo get_recording error: %s", exc)
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# Hangup
# ─────────────────────────────────────────────────────────────────────────────
async def hangup(call_sid: str) -> None:
    """Terminate an active Plivo call by CallUUID."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.delete(
            f"{_BASE}/Call/{call_sid}/",
            auth=_AUTH,
        )
        if r.status_code == 404:
            log.info("Plivo hangup %s → already gone (404)", call_sid)
        elif r.status_code >= 400:
            log.warning("Plivo hangup %s → HTTP %d: %s", call_sid, r.status_code, r.text[:200])
        else:
            log.info("Plivo call %s terminated", call_sid)


# ─────────────────────────────────────────────────────────────────────────────
# Plivo XML — tells Plivo to stream bidirectional audio to our WebSocket
# ─────────────────────────────────────────────────────────────────────────────
def connect_response(ws_url: str) -> str:
    """Return Plivo XML that connects the call to the media-stream WebSocket."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Stream keepCallAlive="true" bidirectional="true" contentType="audio/x-mulaw;rate=8000">'
        f"{ws_url}"
        "</Stream>"
        "</Response>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket frame parsing  (Plivo media stream format)
# ─────────────────────────────────────────────────────────────────────────────
def parse_ws_frame(data: dict) -> tuple[str, dict]:
    """Translate a raw Plivo WebSocket media-stream JSON frame into a normalised event."""
    evt = data.get("event", "")

    if evt == "start":
        start = data.get("start", {})
        mf = start.get("mediaFormat") or {}
        return "call_start", {
            "stream_sid": start.get("streamId", ""),
            "call_sid":   start.get("callId",   ""),
            "content_type": str(mf.get("contentType") or ""),
            "sample_rate":  int(mf.get("sampleRate") or 0) if str(mf.get("sampleRate") or "").isdigit() else 0,
            "track":        str((start.get("tracks") or [""])[0] if isinstance(start.get("tracks"), list) else ""),
        }

    if evt == "media":
        media = data.get("media", {})
        # Skip outbound echo frames (audio we sent back)
        if media.get("track", "inbound") == "outbound":
            return "ignore", {}
        payload = media.get("payload", "")
        if not payload:
            return "ignore", {}
        return "audio_frame", {"mulaw": _b64.b64decode(payload)}

    if evt == "stop":
        return "ignore", {}

    return "ignore", {}


# ─────────────────────────────────────────────────────────────────────────────
# Outgoing WebSocket messages
# ─────────────────────────────────────────────────────────────────────────────
def media_msg(audio_mulaw: bytes, stream_sid: str) -> dict:
    """Build the Plivo JSON payload to send µ-law audio to the caller."""
    return {
        "event": "playAudio",
        "media": {
            "contentType": "audio/x-mulaw",
            "sampleRate":  8000,
            "payload":     _b64.b64encode(audio_mulaw).decode(),
        },
    }


def mark_msg(stream_sid: str) -> dict | None:
    """
    Plivo does not support mark/acknowledgement events.
    Returning None causes send_mark() to treat audio as immediately drained.
    """
    return None


def clear_msg(stream_sid: str) -> dict | None:
    """Return a Plivo clearAudio event to flush buffered audio on barge-in."""
    return {
        "event": "clearAudio",
        "id":    stream_sid,
    }
