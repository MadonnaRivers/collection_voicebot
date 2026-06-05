"""
tts.py — Sarvam Bulbul v3 TTS helpers.

Real-time strategy:
  • tts_stream()          — HTTP chunked streaming (lowest latency, primary path).
  • tts_rest()            — Single-shot REST fallback when streaming fails.
  • tts_stream_pipelined()— Splits long text into sentences and pipelines them:
                            starts playing sentence-1 while sentence-2 is being
                            fetched, reducing time-to-first-audio for long scripts.

Concurrency:
  • _TTS_SEM caps simultaneous Sarvam requests across all calls.
    Set TTS_CONCURRENCY in .env (default 15). Raise this when on a higher Sarvam plan.
  • 429 responses trigger exponential backoff + retry before giving up.
"""
from __future__ import annotations
import asyncio
import base64
import json
import logging
import os
import re
from typing import Awaitable, Callable

from clients import http
from config import SARVAM_API_KEY, SARVAM_VOICE, TTS_PACE, SARVAM_TTS_STREAM_URL, SARVAM_TTS_REST_URL

log = logging.getLogger("aditi")

# ─────────────────────────────────────────────────────────────────────────────
# Global TTS concurrency limiter
# Caps simultaneous Sarvam requests so rate limits aren't hit in bulk.
# Default 200 — sized for Sarvam Pro plan running 100 concurrent calls
# (~2 in-flight sentences per call from pipelined sentence streaming).
# Lower TTS_CONCURRENCY in .env on starter plans (e.g. 30) to avoid 429s.
# ─────────────────────────────────────────────────────────────────────────────
_TTS_CONCURRENCY = int(os.getenv("TTS_CONCURRENCY", "200"))
_TTS_SEM: asyncio.Semaphore | None = None

def _sem() -> asyncio.Semaphore:
    global _TTS_SEM
    if _TTS_SEM is None:
        _TTS_SEM = asyncio.Semaphore(_TTS_CONCURRENCY)
    return _TTS_SEM

# Backoff delays (seconds) between 429 retries — released outside semaphore
_429_BACKOFFS = (0.5, 1.5, 3.5)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _tts_payload(text: str) -> dict:
    return {
        "text":                 text,
        "target_language_code": "hi-IN",
        "speaker":              SARVAM_VOICE,
        "model":                "bulbul:v3",
        "speech_sample_rate":   8000,
        "output_audio_codec":   "mulaw",
        "enable_preprocessing": True,
        "pace":                 TTS_PACE,
    }


def _tts_headers() -> dict[str, str]:
    return {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}


def _strip_wav_header(audio: bytes) -> bytes:
    if audio.startswith(b"RIFF"):
        idx = audio.find(b"data")
        if idx != -1:
            return audio[idx + 8:]
    return audio


# ─────────────────────────────────────────────────────────────────────────────
# Core streaming TTS
# ─────────────────────────────────────────────────────────────────────────────
async def tts_stream(
    text: str,
    on_chunk: Callable[[bytes], Awaitable[None]],
    abort: list[bool],
) -> bool:
    """Stream TTS audio chunks to on_chunk. Returns True if audio was produced.
    Acquires global semaphore before each attempt; retries up to 3× on 429."""
    sem = _sem()
    for attempt in range(len(_429_BACKOFFS) + 1):
        if abort[0]:
            return False
        async with sem:
            try:
                async with http.stream(
                    "POST", SARVAM_TTS_STREAM_URL,
                    json=_tts_payload(text), headers=_tts_headers(),
                ) as resp:
                    if resp.status_code == 429:
                        body = await resp.aread()
                        log.warning(
                            "TTS stream 429 (attempt %d/%d): %s",
                            attempt + 1, len(_429_BACKOFFS) + 1, body[:100],
                        )
                        # fall through to backoff below
                    elif resp.status_code >= 400:
                        body = await resp.aread()
                        log.error("TTS HTTP %d: %s", resp.status_code, body[:200])
                        return False
                    else:
                        # ── Normal streaming path ──────────────────────────
                        container: str | None = None
                        wav_buf = b""
                        wav_started = False
                        got_audio = False

                        async for chunk in resp.aiter_bytes(1024):
                            if abort[0]:
                                return True
                            if not chunk:
                                continue
                            if container is None:
                                probe = chunk[:12]
                                if probe.startswith(b"RIFF") and b"WAVE" in probe:
                                    container = "wav"
                                elif probe[:1] in (b"{", b"["):
                                    container = "json"
                                else:
                                    container = "raw"

                            if container == "raw":
                                got_audio = True
                                await on_chunk(chunk)
                            elif container == "json":
                                tail = await resp.aread()
                                try:
                                    data = json.loads((chunk + tail).decode("utf-8", errors="ignore"))
                                    b64  = (data.get("audios") or [None])[0] or data.get("audio")
                                    if b64:
                                        got_audio = True
                                        await on_chunk(base64.b64decode(b64))
                                except Exception as exc:
                                    log.warning("TTS JSON decode: %s", exc)
                                return got_audio
                            else:
                                if not wav_started:
                                    wav_buf += chunk
                                    idx = wav_buf.find(b"data")
                                    if idx == -1:
                                        continue
                                    start = idx + 8
                                    audio = wav_buf[start:]
                                    wav_buf = b""
                                    wav_started = True
                                    if audio:
                                        got_audio = True
                                        await on_chunk(audio)
                                else:
                                    got_audio = True
                                    await on_chunk(chunk)

                        return got_audio

            except Exception as exc:
                log.error("TTS stream error: %s", exc)
                return False

        # 429 backoff — semaphore released while we wait so others can proceed
        if attempt < len(_429_BACKOFFS):
            await asyncio.sleep(_429_BACKOFFS[attempt])

    log.error("TTS stream: all %d attempts exhausted (429)", len(_429_BACKOFFS) + 1)
    return False


async def tts_rest(text: str) -> bytes:
    """Single-shot REST TTS — returns raw µ-law audio bytes.
    Acquires global semaphore; retries up to 3× on 429 with backoff."""
    sem = _sem()
    for attempt in range(len(_429_BACKOFFS) + 1):
        async with sem:
            r = await http.post(SARVAM_TTS_REST_URL, json=_tts_payload(text), headers=_tts_headers())
            if r.status_code == 429:
                log.warning(
                    "TTS REST 429 (attempt %d/%d): %s",
                    attempt + 1, len(_429_BACKOFFS) + 1, r.text[:100],
                )
                # fall through to backoff below
            elif r.status_code >= 400:
                raise RuntimeError(f"TTS REST {r.status_code}: {r.text[:200]}")
            else:
                data = r.json()
                b64  = (data.get("audios") or [None])[0] or data.get("audio")
                if not b64:
                    raise RuntimeError("TTS REST: no audio in response")
                return base64.b64decode(b64)

        # 429 backoff — outside semaphore
        if attempt < len(_429_BACKOFFS):
            await asyncio.sleep(_429_BACKOFFS[attempt])

    raise RuntimeError(f"TTS REST: all {len(_429_BACKOFFS) + 1} attempts exhausted (429)")


# ─────────────────────────────────────────────────────────────────────────────
# Pipelined TTS — splits long text into sentences for lower latency
# ─────────────────────────────────────────────────────────────────────────────
_SENTENCE_SPLIT = re.compile(r'(?<=[।.!?])\s+')

# ─────────────────────────────────────────────────────────────────────────────
# Inter-sentence silence trimming
# Sarvam pads each sentence's audio with ~1-2s trailing silence + leading silence.
# This makes natural sentence breaks feel like 2-3s gaps. We trim aggressively
# to keep breaks at ~150ms while still sounding natural.
# µ-law silence bytes: 0xFF (G.711 standard "zero") and 0x7F.
# 8 kHz sample rate → 8 samples per ms → keep ~150ms = 1200 trailing bytes max.
# ─────────────────────────────────────────────────────────────────────────────
_MULAW_SILENCE = (0xFF, 0x7F, 0x00)
_KEEP_TRAILING_SILENCE_BYTES = 1200   # ~150ms at 8kHz µ-law


def _trim_sentence_silence(audio: bytes) -> bytes:
    """Trim leading + most-of-trailing silence from µ-law sentence audio."""
    if not audio:
        return audio
    n = len(audio)

    # Trim leading silence (samples are 0xFF/0x7F/0x00)
    start = 0
    while start < n and audio[start] in _MULAW_SILENCE:
        start += 1

    # Trim most of trailing silence, keep ~150ms for natural feel
    end = n
    while end > start and audio[end - 1] in _MULAW_SILENCE:
        end -= 1
    end = min(n, end + _KEEP_TRAILING_SILENCE_BYTES)

    return audio[start:end]


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries (Devanagari danda + Latin punctuation)."""
    parts = _SENTENCE_SPLIT.split(text.strip())
    # Merge very short fragments (< 20 chars) with the next sentence
    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1]) < 20:
            merged[-1] += " " + part
        else:
            merged.append(part)
    return [p.strip() for p in merged if p.strip()]


async def tts_speak(
    text: str,
    on_chunk: Callable[[bytes], Awaitable[None]],
    abort: list[bool],
) -> bool:
    """
    Bot-turn entry point. Use streaming for low time-to-first-audio; fall back
    to one-shot REST if the stream returns nothing.
    """
    text = (text or "").strip()
    if not text or abort[0]:
        return False
    if await tts_stream(text, on_chunk, abort):
        return True
    if abort[0]:
        return False
    log.warning("[TTS] stream produced no audio — falling back to REST")
    try:
        audio = await tts_rest(text)
    except Exception as exc:
        log.error("[TTS] REST fallback failed: %s", exc)
        return False
    if not audio or abort[0]:
        return False
    await on_chunk(_strip_wav_header(audio))
    return True


async def tts_stream_pipelined(
    text: str,
    on_chunk: Callable[[bytes], Awaitable[None]],
    abort: list[bool],
) -> bool:
    """
    Pipeline TTS for long scripts:
      - Fetch sentence N+1 concurrently while sentence N is playing.
      - Reduces time-to-first-audio for multi-sentence scripts by ~40-60%.
      - Falls back to single tts_stream() for short texts (1 sentence).
    """
    sentences = _split_sentences(text)

    if len(sentences) <= 1:
        # Short text — plain streaming is already optimal
        return await tts_stream(text, on_chunk, abort)

    got_audio = False

    async def _fetch(sentence: str) -> bytes | None:
        """Pre-fetch a sentence into memory (for pipeline lookahead)."""
        buf: list[bytes] = []
        async def _collect(chunk: bytes) -> None:
            buf.append(chunk)
        ok = await tts_stream(sentence, _collect, abort)
        return b"".join(buf) if ok else None

    # Kick off fetch for sentence[1] before sentence[0] even starts playing
    prefetch_task: asyncio.Task | None = None
    if len(sentences) >= 2:
        prefetch_task = asyncio.create_task(_fetch(sentences[1]))

    for i, sentence in enumerate(sentences):
        if abort[0]:
            if prefetch_task and not prefetch_task.done():
                prefetch_task.cancel()
            break

        if i == 0:
            # Stream sentence 0 live (lowest latency for first words)
            ok = await tts_stream(sentence, on_chunk, abort)
            if ok:
                got_audio = True
        else:
            # Use pre-fetched audio for sentence i — trim padding silence
            # so inter-sentence break feels natural (~150ms), not 2-3 seconds.
            audio = await prefetch_task if prefetch_task else None
            if audio:
                audio = _trim_sentence_silence(audio)
                got_audio = True
                await on_chunk(audio)
            elif not abort[0]:
                # Prefetch failed — stream live as fallback
                ok = await tts_stream(sentence, on_chunk, abort)
                if ok:
                    got_audio = True

            # Pre-fetch the NEXT sentence while this one plays
            prefetch_task = None
            if i + 1 < len(sentences) and not abort[0]:
                prefetch_task = asyncio.create_task(_fetch(sentences[i + 1]))

    if prefetch_task and not prefetch_task.done():
        prefetch_task.cancel()

    return got_audio
