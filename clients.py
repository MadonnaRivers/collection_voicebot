"""
clients.py — Shared HTTP and LLM clients (one instance per process).
"""
from __future__ import annotations
import httpx
from openai import AsyncOpenAI
from config import SARVAM_API_KEY, SARVAM_LLM_BASE_URL, OPENAI_API_KEY

# Async HTTP client — used for TTS, webhook, and audio upload requests.
# max_connections=400: 100 calls × (TTS stream + LLM + webhook + recording download)
#   = ~300-400 peak concurrent connections.
# pool timeout raised to 15s so recording downloads during call-end bursts don't drop.
http = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=3.0, read=30.0, write=10.0, pool=15.0),
    limits=httpx.Limits(max_keepalive_connections=100, max_connections=400),
)

# Sarvam LLM via OpenAI-compatible endpoint (kept for future use)
oai_sarvam = AsyncOpenAI(api_key=SARVAM_API_KEY, base_url=SARVAM_LLM_BASE_URL)

# OpenAI Client
oai_llm = AsyncOpenAI(api_key=OPENAI_API_KEY)
