"""
test_stt_tts.py — Smoke test for Sarvam TTS + STT using the project's own config.

What it does:
  1. TTS roundtrip — sends a Hindi sentence to Sarvam Bulbul v3 via the REST
     endpoint (same payload the bot uses), writes the response to a .wav file,
     reports bytes received and audio duration.
  2. STT roundtrip — converts the TTS µ-law audio to PCM16, streams it into
     Sarvam STT over the same WebSocket URL the bot uses, and prints the
     recognised transcript.

Pass criteria:
  • TTS: HTTP 200, > 1 KB of audio, duration > 0.5 s.
  • STT: WS connects, at least one transcript event received with non-empty text.

Run:
  venv/Scripts/python.exe test_stt_tts.py
"""
from __future__ import annotations
import asyncio
import audioop  # noqa: F401  # stdlib in 3.9 — used for µ-law → PCM16
import base64
import json
import struct
import sys
import time
import wave
from pathlib import Path

# Make sure project root is on path (we're already running from there but be safe)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Force UTF-8 stdout on Windows so Hindi / box-drawing chars don't crash cp1252
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import httpx
import websockets

from config import (
    SARVAM_API_KEY,
    SARVAM_STT_WS_URL,
    SARVAM_TTS_REST_URL,
    SARVAM_VOICE,
    TTS_PACE,
)

TEST_SENTENCE = (
    "नमस्ते, मैं अदिति बोल रही हूँ Easy Home Finance से। "
    "आपकी EMI pending है, बताइए कब तक payment कर पाएंगे?"
)

OUT_WAV   = Path("test_tts_output.wav")
OUT_RAW   = Path("test_tts_output.ulaw")


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 1) TTS test — single-shot REST (same payload as tts.py::tts_rest)
# ─────────────────────────────────────────────────────────────────────────────
async def test_tts() -> bytes | None:
    banner("TTS TEST — Sarvam Bulbul v3 REST")
    payload = {
        "text":                 TEST_SENTENCE,
        "target_language_code": "hi-IN",
        "speaker":              SARVAM_VOICE,
        "model":                "bulbul:v3",
        "speech_sample_rate":   8000,
        "output_audio_codec":   "mulaw",
        "enable_preprocessing": True,
        "pace":                 TTS_PACE,
    }
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type":         "application/json",
    }

    print(f"  endpoint : {SARVAM_TTS_REST_URL}")
    print(f"  voice    : {SARVAM_VOICE}    pace : {TTS_PACE}")
    print(f"  text     : {TEST_SENTENCE[:60]}…")

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(SARVAM_TTS_REST_URL, json=payload, headers=headers)
    except Exception as exc:
        print(f"  ❌ HTTP error: {exc}")
        return None
    dt = time.monotonic() - t0

    print(f"  status   : HTTP {r.status_code}    ({dt*1000:.0f} ms)")
    if r.status_code >= 400:
        print(f"  ❌ body  : {r.text[:300]}")
        return None

    try:
        data = r.json()
    except Exception as exc:
        print(f"  ❌ json   : {exc}    body={r.text[:200]}")
        return None

    b64 = (data.get("audios") or [None])[0] or data.get("audio")
    if not b64:
        print(f"  ❌ no audio in response. keys={list(data.keys())}")
        return None

    ulaw = base64.b64decode(b64)
    duration_sec = len(ulaw) / 8000.0  # 8 kHz µ-law → 1 byte = 1 sample
    print(f"  audio    : {len(ulaw):,} bytes µ-law  →  ~{duration_sec:.2f} s")

    # Save .ulaw raw + a playable .wav (PCM16 8 kHz)
    OUT_RAW.write_bytes(ulaw)
    pcm16 = audioop.ulaw2lin(ulaw, 2)
    with wave.open(str(OUT_WAV), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)         # 16-bit PCM
        wf.setframerate(8000)
        wf.writeframes(pcm16)
    print(f"  saved    : {OUT_WAV}   ({OUT_WAV.stat().st_size:,} bytes wav)")
    print(f"  saved    : {OUT_RAW}   (raw µ-law for STT roundtrip)")

    # Pass/fail
    ok = (len(ulaw) > 1024) and (duration_sec > 0.5)
    print(f"  result   : {'✅ PASS' if ok else '❌ FAIL'}")
    return ulaw if ok else None


# ─────────────────────────────────────────────────────────────────────────────
# 2) STT test — send the TTS audio back through Sarvam STT WebSocket
# ─────────────────────────────────────────────────────────────────────────────
async def test_stt(ulaw_audio: bytes) -> bool:
    banner("STT TEST — Sarvam saaras:v3 WebSocket")
    print(f"  endpoint : {SARVAM_STT_WS_URL[:90]}…")

    # Convert TTS µ-law output → PCM16 (what STT expects per the WS query params)
    pcm16 = audioop.ulaw2lin(ulaw_audio, 2)
    audio_sec = len(pcm16) / 2 / 8000.0
    print(f"  audio    : {len(pcm16):,} bytes PCM16 8kHz  (~{audio_sec:.2f} s)")

    t0 = time.monotonic()
    try:
        ws = await asyncio.wait_for(
            websockets.connect(
                SARVAM_STT_WS_URL,
                extra_headers={"Api-Subscription-Key": SARVAM_API_KEY},
                ping_interval=20,
                ping_timeout=10,
            ),
            timeout=10.0,
        )
    except Exception as exc:
        print(f"  ❌ connect: {exc}")
        return False
    print(f"  connect  : ✅ ({(time.monotonic() - t0)*1000:.0f} ms)")

    transcripts: list[str] = []
    events: list[str] = []
    stop = asyncio.Event()

    async def receiver() -> None:
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    msg = json.loads(raw)
                except Exception:
                    print(f"    [non-json] {raw[:200]}")
                    continue
                et = msg.get("event_type") or msg.get("type") or ""
                events.append(et or "?")
                # Pull transcript text from any of several possible field names
                txt = (
                    msg.get("transcript")
                    or msg.get("text")
                    or (msg.get("data") or {}).get("transcript")
                    or ((msg.get("data") or {}).get("transcript_text") or "")
                ).strip()
                if txt:
                    transcripts.append(txt)
                    print(f"    [{et or 'transcript'}]  {txt}")
                elif et == "error":
                    print(f"    [error] {json.dumps(msg, ensure_ascii=False)[:400]}")
                elif et:
                    print(f"    [{et}]")
                else:
                    # Unknown shape — dump it for debugging
                    print(f"    [?] {json.dumps(msg, ensure_ascii=False)[:300]}")
        except websockets.ConnectionClosed as exc:
            print(f"    (ws closed: code={exc.code} reason={exc.reason!r})")
        finally:
            stop.set()

    recv_task = asyncio.create_task(receiver())

    # Send audio the same way call_handler.py does: JSON envelope with base64 PCM16.
    FRAME_BYTES = 320  # 20 ms at 8 kHz 16-bit mono
    sent_frames = 0
    try:
        for i in range(0, len(pcm16), FRAME_BYTES):
            if ws.closed:
                print(f"  ⚠ ws closed mid-send after {sent_frames} frames")
                break
            chunk = pcm16[i:i + FRAME_BYTES]
            if len(chunk) < FRAME_BYTES:
                chunk = chunk + b"\x00" * (FRAME_BYTES - len(chunk))
            try:
                # NOTE: the URL query already set input_audio_codec=pcm_s16le,
                # so per-frame "encoding" only needs to satisfy the server's enum.
                # Sarvam currently requires audio/wav as the per-frame encoding tag
                # even though we're sending raw PCM16 (matches the codec query param).
                await ws.send(json.dumps({
                    "audio": {
                        "data":        base64.b64encode(chunk).decode(),
                        "sample_rate": 8000,
                        "encoding":    "audio/wav",
                    }
                }))
            except websockets.ConnectionClosed as exc:
                print(f"  ⚠ ws closed by server after {sent_frames} frames "
                      f"(code={exc.code} reason={exc.reason!r})")
                break
            sent_frames += 1
            await asyncio.sleep(0.02)  # real-time pacing
        print(f"  send     : ✅ {sent_frames} frames sent (relying on VAD end_speech)")
        # Pad ~500 ms of silence so Sarvam's VAD reliably detects end-of-speech
        silence = b"\x00" * FRAME_BYTES
        for _ in range(25):
            if ws.closed:
                break
            try:
                await ws.send(json.dumps({
                    "audio": {
                        "data":        base64.b64encode(silence).decode(),
                        "sample_rate": 8000,
                        "encoding":    "audio/wav",
                    }
                }))
            except Exception:
                break
            await asyncio.sleep(0.02)

        # Wait up to 10 s for final transcripts to flush in
        try:
            await asyncio.wait_for(stop.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            print("  (10s timeout — closing)")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        recv_task.cancel()
        try:
            await recv_task
        except Exception:
            pass

    print(f"  events   : {events or '(none)'}")
    final = " ".join(transcripts).strip()
    print(f"  text     : {final or '(empty)'}")

    ok = bool(final)
    print(f"  result   : {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
async def main() -> int:
    print(f"\nSarvam API key : {SARVAM_API_KEY[:8]}…{SARVAM_API_KEY[-4:]}  "
          f"(len={len(SARVAM_API_KEY)})")
    tts_audio = await test_tts()
    stt_ok = False
    if tts_audio:
        stt_ok = await test_stt(tts_audio)
    else:
        print("\n(skipping STT test — TTS failed, no audio to send back)")

    banner("SUMMARY")
    print(f"  TTS : {'✅' if tts_audio else '❌'}")
    print(f"  STT : {'✅' if stt_ok else '❌'}")
    return 0 if (tts_audio and stt_ok) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
