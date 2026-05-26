"""
stt.py — Sarvam Speech-to-Text over REST (realtime, VAD-segmented).

Sarvam's REST endpoint accepts short audio (under 30 s). We feed it
realtime by chopping the live audio into utterances using the local
WebRTC VAD already running in call_handler, then POST each utterance
the moment the speaker pauses.

Callbacks:
  on_transcript(text)   — fires when an utterance is transcribed.
  on_speech_start()     — fires after STT_BARGE_IN_MIN_FRAMES sustained
                          VAD-speech frames (used to trigger barge-in).
"""
from __future__ import annotations
import asyncio
import audioop
import io
import logging
import time
import wave
from typing import Awaitable, Callable

import httpx

from clients import http
from config import (
    SARVAM_API_KEY,
    SARVAM_STT_REST_URL,
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
    """Per-call VAD-segmented STT. Feed 20 ms frames + VAD verdict via feed()."""

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

        log.info("[STT] transcribing — %d ms utterance", dur_ms)
        t0 = time.monotonic()

        # Retry once on transient (network / 5xx / 429).
        r = None
        last_err: str | None = None
        for attempt in range(2):
            try:
                r = await http.post(
                    SARVAM_STT_REST_URL,
                    headers={"api-subscription-key": SARVAM_API_KEY},
                    files={"file": ("utt.wav", wav, "audio/wav")},
                    data={"model": SARVAM_STT_MODEL, "language_code": SARVAM_STT_LANGUAGE},
                    timeout=httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=10.0),
                )
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                log.warning("[STT] REST attempt %d/2 error: %s", attempt + 1, last_err)
                r = None
                if attempt == 0:
                    await asyncio.sleep(0.8)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
                log.warning("[STT] REST attempt %d/2 transient: %s", attempt + 1, last_err)
                if attempt == 0:
                    await asyncio.sleep(0.8)
                    continue
            break

        if r is None:
            log.error("[STT] REST exhausted retries: %s", last_err)
            return
        if r.status_code >= 400:
            log.error("[STT] HTTP %d: %s", r.status_code, r.text[:200])
            return

        try:
            data = r.json()
        except Exception as exc:
            log.error("[STT] JSON parse: %s", exc)
            return

        transcript = (data.get("transcript") or "").strip()
        latency_ms = int((time.monotonic() - t0) * 1000)
        if not transcript:
            log.info("[STT] empty transcript (%d ms)", latency_ms)
            return
        log.info("[STT] transcribed in %d ms", latency_ms)
        try:
            await self._on_transcript(transcript)
        except Exception as exc:
            log.error("[STT] on_transcript handler error: %s", exc)

    async def close(self) -> None:
        self._closed = True
        if self._in_speech and self._buf:
            await self._flush()
