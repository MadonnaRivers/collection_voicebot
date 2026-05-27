"""
stt.py — Sarvam Speech-to-Text over WebSocket streaming (realtime, VAD-segmented).

We keep local per-call VAD segmentation in this module so existing call flow stays
the same, but each call uses one persistent Sarvam STT WebSocket session instead of
repeated REST requests. This avoids stt-rt req/min bottlenecks at high concurrency.

Callbacks:
  on_transcript(text)   — fires when an utterance is transcribed.
  on_speech_start()     — fires after STT_BARGE_IN_MIN_FRAMES sustained
                          VAD-speech frames (used to trigger barge-in).
"""
from __future__ import annotations
import asyncio
import audioop
import base64
import io
import logging
import wave
from typing import Awaitable, Callable

from sarvamai import AsyncSarvamAI

from config import (
    SARVAM_API_KEY,
    SARVAM_STT_MODEL,
    SARVAM_STT_LANGUAGE,
    STT_SILENCE_HANGOVER_MS,
    STT_MIN_UTTERANCE_MS,
    STT_MAX_UTTERANCE_SEC,
    STT_BARGE_IN_MIN_FRAMES,
)

log = logging.getLogger("aditi")

_FRAME_MS = 20  # caller feeds 20 ms PCM16 frames @ 8 kHz
_HANGOVER_FRAMES = max(1, STT_SILENCE_HANGOVER_MS // _FRAME_MS)
_MIN_FRAMES      = max(1, STT_MIN_UTTERANCE_MS    // _FRAME_MS)
_MAX_FRAMES      = max(_MIN_FRAMES, int(STT_MAX_UTTERANCE_SEC * 1000) // _FRAME_MS)


def _pcm8k_to_wav16k(pcm8k: bytes) -> bytes:
    """Resample 8 kHz PCM16 to 16 kHz and wrap as a WAV file in memory."""
    pcm16k, _ = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, None)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16k)
    return buf.getvalue()


class RestStt:
    """Per-call VAD-segmented STT over one persistent WebSocket session."""

    def __init__(
        self,
        on_transcript: Callable[[str], Awaitable[None]],
        on_speech_start: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._on_transcript   = on_transcript
        self._on_speech_start = on_speech_start
        self._buf: list[bytes] = []
        self._frames_since_speech = 0
        self._in_speech = False
        self._closed = False
        self._speech_run = 0
        self._speech_start_fired = False
        self._connect_lock = asyncio.Lock()
        self._client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
        self._ws_cm = None
        self._ws = None
        self._recv_task: asyncio.Task | None = None

    async def _ensure_ws(self) -> bool:
        if self._ws is not None:
            return True
        if self._closed:
            return False
        async with self._connect_lock:
            if self._ws is not None:
                return True
            try:
                kwargs: dict[str, str] = {
                    "language_code": SARVAM_STT_LANGUAGE,
                    "model": SARVAM_STT_MODEL,
                    "sample_rate": "16000",
                    "input_audio_codec": "wav",
                    "vad_signals": "false",
                    "flush_signal": "true",
                }
                if SARVAM_STT_MODEL.startswith("saaras"):
                    kwargs["mode"] = "transcribe"
                self._ws_cm = self._client.speech_to_text_streaming.connect(**kwargs)
                self._ws = await self._ws_cm.__aenter__()
                # Start the single continuous receiver that dispatches
                # transcripts as they arrive — never block a flush on recv().
                self._recv_task = asyncio.create_task(self._recv_loop())
                log.info("[STT] streaming websocket connected")
            except Exception as exc:
                self._ws = None
                self._ws_cm = None
                log.error("[STT] streaming connect failed: %s", exc)
                return False
        return True

    async def _recv_loop(self) -> None:
        """Continuously read the WS and dispatch final transcripts. One per call."""
        ws = self._ws
        while not self._closed and ws is not None and ws is self._ws:
            try:
                msg = await ws.recv()
            except Exception as exc:
                if not self._closed:
                    log.info("[STT] recv loop ended: %s", exc)
                break
            try:
                msg_type = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else None)
                msg_data = getattr(msg, "data", None) or (msg.get("data") if isinstance(msg, dict) else None)
                if msg_type == "error":
                    err = getattr(msg_data, "error", None) or str(msg_data)
                    log.error("[STT] websocket error: %s", err)
                    continue
                if msg_type in ("events", "event"):
                    continue
                transcript = (
                    (getattr(msg_data, "transcript", None) if msg_data is not None else None)
                    or (msg_data.get("transcript") if isinstance(msg_data, dict) else None)
                    or getattr(msg, "transcript", None)
                    or (msg.get("transcript") if isinstance(msg, dict) else None)
                    or ""
                ).strip()
                if not transcript:
                    continue
                log.info("[STT] transcribed: %.80s", transcript)
                try:
                    await self._on_transcript(transcript)
                except Exception as exc:
                    log.error("[STT] on_transcript handler error: %s", exc)
            except Exception as exc:
                log.debug("[STT] recv parse error: %s", exc)

    async def _close_ws(self) -> None:
        ws_cm = self._ws_cm
        self._ws = None
        self._ws_cm = None
        task = self._recv_task
        self._recv_task = None
        if task is not None and not task.done():
            task.cancel()
        if ws_cm is None:
            return
        try:
            await ws_cm.__aexit__(None, None, None)
        except Exception as exc:
            log.debug("[STT] websocket close error: %s", exc)

    async def feed(self, pcm16_frame: bytes, is_speech: bool) -> None:
        if self._closed:
            return

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._speech_run = 0
                self._speech_start_fired = False
            self._buf.append(pcm16_frame)
            self._frames_since_speech = 0
            self._speech_run += 1

            # Sustained-speech guard before barge-in (filters single noise bursts).
            if (not self._speech_start_fired
                    and self._speech_run >= STT_BARGE_IN_MIN_FRAMES
                    and self._on_speech_start is not None):
                self._speech_start_fired = True
                try:
                    await self._on_speech_start()
                except Exception as exc:
                    log.debug("on_speech_start error: %s", exc)

            # Force-flush long monologues (Sarvam REST caps at 30 s).
            if len(self._buf) >= _MAX_FRAMES:
                asyncio.create_task(self._flush())
            return

        # Silent frame after speech
        if self._in_speech:
            self._buf.append(pcm16_frame)
            self._frames_since_speech += 1
            self._speech_run = 0
            if self._frames_since_speech >= _HANGOVER_FRAMES:
                asyncio.create_task(self._flush())

    async def _flush(self) -> None:
        """Send the buffered utterance to the WS and return immediately.
        Transcripts come back asynchronously via _recv_loop — we never block
        here waiting for a reply (that was the 6 s stall bug)."""
        if not self._in_speech:
            return
        self._in_speech = False
        self._frames_since_speech = 0
        self._speech_run = 0
        self._speech_start_fired = False
        frames = self._buf
        self._buf = []
        if len(frames) < _MIN_FRAMES:
            return

        dur_ms = len(frames) * _FRAME_MS
        pcm = b"".join(frames)
        try:
            wav = _pcm8k_to_wav16k(pcm)
        except Exception as exc:
            log.error("[STT] wav encode failed: %s", exc)
            return

        if not await self._ensure_ws():
            return

        log.info("[STT] transcribing — %d ms utterance", dur_ms)
        audio_b64 = base64.b64encode(wav).decode("ascii")
        try:
            await self._ws.transcribe(audio=audio_b64, encoding="audio/wav", sample_rate=16000)
            await self._ws.flush()
        except Exception as exc:
            log.warning("[STT] websocket send/flush error: %s", exc)
            await self._close_ws()

    async def close(self) -> None:
        self._closed = True
        if self._in_speech and self._buf:
            await self._flush()
        await self._close_ws()
