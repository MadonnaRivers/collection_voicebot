"""
clients.py — Shared HTTP and LLM clients (one instance per process).
"""
from __future__ import annotations
import httpx
from openai import AsyncOpenAI
from config import OPENAI_API_KEY

# Async HTTP client — used for TTS streaming, STT REST, webhooks, recording downloads.
http = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=3.0, read=30.0, write=10.0, pool=15.0),
    limits=httpx.Limits(max_keepalive_connections=100, max_connections=400),
)

# OpenAI client for the dialogue LLM.
oai_llm = AsyncOpenAI(api_key=OPENAI_API_KEY)
