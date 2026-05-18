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

# ── Sarvam endpoints ──────────────────────────────────────────────────────────
SARVAM_STT_WS_BASE  = os.getenv("SARVAM_STT_WS_BASE",  "wss://api.sarvam.ai/speech-to-text/ws")
SARVAM_STT_MODEL    = os.getenv("SARVAM_STT_MODEL",     "saaras:v3")
SARVAM_STT_LANGUAGE = os.getenv("SARVAM_STT_LANGUAGE",  "hi-IN")
SARVAM_LLM_BASE_URL = os.getenv("SARVAM_LLM_BASE_URL",  "https://api.sarvam.ai/v1")
SARVAM_VOICE        = os.getenv("SARVAM_VOICE",         "simran")

SARVAM_TTS_STREAM_URL = "https://api.sarvam.ai/text-to-speech/stream"
SARVAM_TTS_REST_URL   = "https://api.sarvam.ai/text-to-speech"

SARVAM_STT_WS_URL = (
    f"{SARVAM_STT_WS_BASE}"
    f"?language-code={SARVAM_STT_LANGUAGE}"
    f"&model={SARVAM_STT_MODEL}"
    f"&mode=transcribe"
    f"&sample_rate=8000"
    f"&input_audio_codec=pcm_s16le"
    f"&vad_signals=true"
    f"&flush_signal=true"
)

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL      = os.getenv("LLM_MODEL",      "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Orchestrator (voice dialogue brain)
ORCHESTRATOR_TEMPERATURE    = float(os.getenv("ORCHESTRATOR_TEMPERATURE",    "0.1"))
ORCHESTRATOR_MAX_HISTORY    = int(os.getenv("ORCHESTRATOR_MAX_HISTORY",      "28"))
ORCHESTRATOR_API_RETRIES    = int(os.getenv("ORCHESTRATOR_API_RETRIES",      "3"))
# Lower = faster answers + less latency (typical Hindi turn fits well under 400)
ORCHESTRATOR_MAX_TOKENS     = int(os.getenv("ORCHESTRATOR_MAX_TOKENS",       "300"))

# ── Server ────────────────────────────────────────────────────────────────────
PORT              = int(os.getenv("PORT",           "5050"))
TRANSCRIPTS_DIR   = os.getenv("TRANSCRIPTS_DIR",   "transcripts")
RECORDINGS_DIR    = os.getenv("RECORDINGS_DIR",    "recordings")
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
SILENCE_TIMEOUT_SEC      = float(os.getenv("SILENCE_TIMEOUT_SEC",      "20.0"))
TTS_PACE                 = float(os.getenv("TTS_PACE",                 "1.1"))
BARGE_IN_GUARD_SEC       = float(os.getenv("BARGE_IN_GUARD_SEC",       "1.5"))
# 0.2s: END_SPEECH from Sarvam fires before this timer for normal sentences
POST_UTTERANCE_PAUSE_SEC = float(os.getenv("POST_UTTERANCE_PAUSE_SEC", "0.05"))

# ── VAD (WebRTC noise gate) ───────────────────────────────────────────────────
VAD_MODE        = int(os.getenv("VAD_MODE",        "2"))
# 100ms = fast END_SPEECH; raise to 200 if trailing syllables get clipped
VAD_HANGOVER_MS = int(os.getenv("VAD_HANGOVER_MS", "100"))
VAD_ENABLED     = os.getenv("VAD_ENABLED", "true").lower() not in ("0", "false", "no")

# ── Spectral denoiser ─────────────────────────────────────────────────────────
DENOISE_ENABLED     = os.getenv("DENOISE_ENABLED",     "true").lower() not in ("0", "false", "no")
DENOISE_STRENGTH    = float(os.getenv("DENOISE_STRENGTH",    "0.88"))
DENOISE_PROFILE_SEC = float(os.getenv("DENOISE_PROFILE_SEC", "2.0"))
DENOISE_STATIONARY  = os.getenv("DENOISE_STATIONARY", "false").lower() not in ("0", "false", "no")
