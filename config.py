"""
config.py — All environment variables and runtime constants for Aditi.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()


def _req(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


# ── Plivo credentials ─────────────────────────────────────────────────────────
PLIVO_AUTH_ID     = _req("PLIVO_AUTH_ID")
PLIVO_AUTH_TOKEN  = _req("PLIVO_AUTH_TOKEN")
PLIVO_PHONE_NUMBER = _req("PLIVO_PHONE_NUMBER")

# ── Required credentials ──────────────────────────────────────────────────────
SARVAM_API_KEY = _req("SARVAM_API_KEY")
NGROK_URL      = _req("NGROK_URL").rstrip("/")

# ── Sarvam endpoints (STT REST, TTS streaming) ────────────────────────────────
SARVAM_STT_REST_URL   = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_STREAM_URL = "https://api.sarvam.ai/text-to-speech/stream"
SARVAM_TTS_REST_URL   = "https://api.sarvam.ai/text-to-speech"   # fallback only

# Hindi-native transcription model. saarika v3 does not exist; v2.5 is the
# latest in the saarika line. (saaras:v3 is newer but translates to English by
# default — we stay on saarika for Devanagari output that matches the prompt.)
SARVAM_STT_MODEL    = os.getenv("SARVAM_STT_MODEL",    "saarika:v2.5")
SARVAM_STT_LANGUAGE = os.getenv("SARVAM_STT_LANGUAGE", "hi-IN")
# bulbul:v3 formal female speakers: ishita (default here), simran, ritu, priya,
# pooja, neha, kavya, shreya, roopa. (v2-only speakers like anushka won't work.)
SARVAM_VOICE        = os.getenv("SARVAM_VOICE",        "ishita")

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL      = os.getenv("LLM_MODEL",      "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Orchestrator (voice dialogue brain)
ORCHESTRATOR_TEMPERATURE    = float(os.getenv("ORCHESTRATOR_TEMPERATURE",    "0.1"))
ORCHESTRATOR_MAX_HISTORY    = int(os.getenv("ORCHESTRATOR_MAX_HISTORY",      "28"))
ORCHESTRATOR_API_RETRIES    = int(os.getenv("ORCHESTRATOR_API_RETRIES",      "3"))
# 400 tokens: enough for the longest payment_confirm template + JSON wrapper
ORCHESTRATOR_MAX_TOKENS     = int(os.getenv("ORCHESTRATOR_MAX_TOKENS",       "400"))

# ── Server ────────────────────────────────────────────────────────────────────
PORT              = int(os.getenv("PORT",           "5050"))
TRANSCRIPTS_DIR   = os.getenv("TRANSCRIPTS_DIR",   "transcripts")
RECORDINGS_DIR    = os.getenv("RECORDINGS_DIR",    "recordings")
LOGS_DIR          = os.getenv("LOGS_DIR",          "logs")
LOG_FILE          = os.getenv("LOG_FILE",          "logs/aditi.log")
LOG_ERROR_FILE    = os.getenv("LOG_ERROR_FILE",    "logs/aditi_error.log")
MAKE_CALL_API_KEY = os.getenv("MAKE_CALL_API_KEY",  "")

# Post-call JSON (all CRM fields) — pushed to n8n /webhook/push_data
CALL_SUMMARY_WEBHOOK_URL = os.getenv(
    "CALL_SUMMARY_WEBHOOK_URL",
    "https://web-n8n.easyhomefinance.in/webhook/push_data",
).strip()

# Plivo will POST recording metadata here when recording is ready (leave blank to disable)
RECORDING_CALLBACK_URL = os.getenv("RECORDING_CALLBACK_URL", "").strip()

# Webhook to receive combined audio + transcript after call ends
AUDIO_TRANSCRIPT_WEBHOOK_URL = os.getenv(
    "AUDIO_TRANSCRIPT_WEBHOOK_URL",
    "https://web-n8n.easyhomefinance.in/webhook/audio_and_transcripts",
).strip()

# ── Call behaviour tunables ───────────────────────────────────────────────────
HANGUP_GRACE_SEC         = float(os.getenv("HANGUP_GRACE_SEC",         "1.5"))
SILENCE_TIMEOUT_SEC      = float(os.getenv("SILENCE_TIMEOUT_SEC",      "6.5"))
TTS_PACE                 = float(os.getenv("TTS_PACE",                 "1.1"))
BARGE_IN_GUARD_SEC       = float(os.getenv("BARGE_IN_GUARD_SEC",       "1.5"))

# ── STT (REST + local VAD) tunables ───────────────────────────────────────────
# Silence after speech before the utterance is POSTed to Sarvam REST.
STT_SILENCE_HANGOVER_MS = int(os.getenv("STT_SILENCE_HANGOVER_MS", "700"))
# Drop bursts shorter than this (cough/pop/noise).
STT_MIN_UTTERANCE_MS    = int(os.getenv("STT_MIN_UTTERANCE_MS",    "200"))
# Force-flush long monologues. Sarvam REST caps at 30 s.
STT_MAX_UTTERANCE_SEC   = float(os.getenv("STT_MAX_UTTERANCE_SEC",  "25"))
# Consecutive VAD-speech frames required before firing the barge-in signal.
STT_BARGE_IN_MIN_FRAMES = int(os.getenv("STT_BARGE_IN_MIN_FRAMES", "12"))

# TTS concurrency
TTS_CONCURRENCY      = int(os.getenv("TTS_CONCURRENCY",      "12"))
TTS_MAX_RETRIES      = int(os.getenv("TTS_MAX_RETRIES",      "4"))
TTS_READ_TIMEOUT_SEC = float(os.getenv("TTS_READ_TIMEOUT_SEC","20"))

# ── VAD (WebRTC noise gate) ───────────────────────────────────────────────────
VAD_MODE        = int(os.getenv("VAD_MODE",        "2"))
# 100ms = fast END_SPEECH; raise to 200 if trailing syllables get clipped
VAD_HANGOVER_MS = int(os.getenv("VAD_HANGOVER_MS", "100"))
VAD_ENABLED     = os.getenv("VAD_ENABLED", "true").lower() not in ("0", "false", "no")

# ── Spectral denoiser ─────────────────────────────────────────────────────────
DENOISE_ENABLED     = os.getenv("DENOISE_ENABLED",     "true").lower() not in ("0", "false", "no")
DENOISE_STRENGTH    = float(os.getenv("DENOISE_STRENGTH",    "0.92"))
DENOISE_PROFILE_SEC = float(os.getenv("DENOISE_PROFILE_SEC", "2.0"))
DENOISE_STATIONARY  = os.getenv("DENOISE_STATIONARY", "false").lower() not in ("0", "false", "no")
