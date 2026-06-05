"""
clients.py — Shared HTTP and LLM clients (one instance per process).
"""
from __future__ import annotations
import httpx
from openai import AsyncOpenAI
from config import OPENAI_API_KEY

# Async HTTP client — used for TTS streaming, STT REST, webhooks, recording downloads.
# Sized for 100 concurrent calls on Sarvam Pro:
#   • TTS streams      : up to 200 (TTS_CONCURRENCY semaphore caps actual usage)
#   • STT REST         : 1-2 per call when an utterance flushes
#   • LLM streams      : 1 per call (OpenAI uses its own client / pool though)
#   • Plivo REST       : start_recording, hangup, etc. — bursty
#   • n8n webhooks     : post-call pushes — bursty
# max_connections 800 leaves headroom; max_keepalive 300 keeps sockets warm
# for Sarvam (biggest win — TLS handshake is ~80 ms otherwise).
http = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=3.0, read=30.0, write=10.0, pool=15.0),
    limits=httpx.Limits(max_keepalive_connections=300, max_connections=800),
)

# OpenAI client for the dialogue LLM.
oai_llm = AsyncOpenAI(api_key=OPENAI_API_KEY)
