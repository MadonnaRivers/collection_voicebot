"""
clients.py — Shared HTTP and LLM clients (one instance per process).
"""
from __future__ import annotations
import asyncio
import logging

import httpx
from openai import AsyncOpenAI
from config import OPENAI_API_KEY

log = logging.getLogger("aditi")

# Async HTTP client — used for TTS streaming, STT REST, webhooks, recording downloads.
# HTTP/2 multiplexes multiple Sarvam requests over a single TCP connection,
# eliminating per-request TLS handshake overhead (saves ~80-150 ms per call).
# Falls back to HTTP/1.1 if the optional `h2` package isn't installed —
# run `pip install h2` to enable HTTP/2 in production.
try:
    import h2  # noqa: F401  — presence-check for httpx[http2] extras
    _HTTP2_OK = True
except ImportError:
    _HTTP2_OK = False
    log.info("h2 package not installed — using HTTP/1.1. `pip install h2` for HTTP/2 speedup.")

http = httpx.AsyncClient(
    http2=_HTTP2_OK,
    timeout=httpx.Timeout(connect=3.0, read=30.0, write=10.0, pool=15.0),
    limits=httpx.Limits(max_keepalive_connections=500, max_connections=1500),
)

# OpenAI client for the dialogue LLM.
oai_llm = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def warmup_connections() -> None:
    """
    Pre-establish TLS connections to Sarvam (STT + TTS) and OpenAI at startup.
    This eliminates the cold-start TLS handshake (~150-250 ms) on the first call
    after the server boots — important for the opening greeting + first STT turn.

    Called once from main.py on app startup. Failures are non-fatal — calls still
    work, just with a one-time cold-start penalty on the first real request.
    """
    targets = [
        ("https://api.sarvam.ai/text-to-speech/stream", "Sarvam TTS"),
        ("https://api.sarvam.ai/speech-to-text",        "Sarvam STT"),
        ("https://api.openai.com/v1/models",            "OpenAI"),
    ]
    async def _ping(url: str, name: str) -> None:
        try:
            # HEAD/GET — server may reject auth, but the TLS handshake completes
            # and the socket goes into the keep-alive pool.
            await http.get(url, timeout=5.0)
            log.info("[WARMUP] connection pre-warmed: %s", name)
        except Exception as exc:
            log.debug("[WARMUP] %s pre-warm skipped: %s", name, exc)
    await asyncio.gather(*[_ping(u, n) for u, n in targets])
