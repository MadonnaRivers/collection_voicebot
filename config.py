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
# bulbul:v3 formal female speakers: simran (default here), ishita, ritu, priya,
# pooja, neha, kavya, shreya, roopa. (v2-only speakers like anushka won't work.)
SARVAM_VOICE        = os.getenv("SARVAM_VOICE",        "simran")

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL      = os.getenv("LLM_MODEL",      "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Orchestrator (voice dialogue brain)
ORCHESTRATOR_TEMPERATURE    = float(os.getenv("ORCHESTRATOR_TEMPERATURE",    "0.1"))
# Fewer history messages → smaller prompt → faster LLM time-to-first-token
# on later turns. 18 still keeps ~9 exchanges of context.
ORCHESTRATOR_MAX_HISTORY    = int(os.getenv("ORCHESTRATOR_MAX_HISTORY",      "18"))
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

# Plivo will POST recording metadata here when recording is ready.
# Defaults to {NGROK_URL}/recording-callback so a single NGROK_URL change in
# .env is enough — no more stale-URL surprises after ngrok tunnel rotates.
# Explicitly set RECORDING_CALLBACK_URL=  (blank) to disable recording metadata.
RECORDING_CALLBACK_URL = os.getenv(
    "RECORDING_CALLBACK_URL", f"{NGROK_URL}/recording-callback",
).strip()

# ── Trigger-envelope webhook ──────────────────────────────────────────────────
# Fires ONLY for payment-positive outcomes:
#   payment_today_confirmed → trigger_code = "PAYMENT_VOICEBOT"
#   ptp_confirmed           → trigger_code = "PAYMENT_VOICEBOT"
#   partial_confirmed       → trigger_code = "PARTIAL_PAYMENT_VOICEBOT"
# All other outcomes (cannot_pay / already_paid / deceased / disputed_loan /
# no_response / voicemail / no_pickup / busy / rejected) are SKIPPED here.
# CALL_SUMMARY_WEBHOOK_URL still receives EVERY call (untouched).
# Leave TRIGGER_WEBHOOK_URL blank to disable the trigger push entirely.
TRIGGER_WEBHOOK_URL = os.getenv("TRIGGER_WEBHOOK_URL", "").strip()

# ── Voicemail / Answering Machine Detection ──────────────────────────────────
# When AMD_ENABLED, /make-call adds Plivo's async-AMD params to the call POST.
# Plivo runs detection in parallel with the call connecting (no extra latency
# for humans) and POSTs the verdict to AMD_CALLBACK_URL.
AMD_ENABLED           = os.getenv("AMD_ENABLED", "true").lower() not in ("0", "false", "no")
AMD_CALLBACK_URL      = os.getenv("AMD_CALLBACK_URL", f"{NGROK_URL}/amd-callback").strip()
# How long Plivo analyses incoming audio (ms). 2000 ms is a fast/accurate
# default; raise to 4000 if too many false 'human' verdicts on voicemails.
AMD_DETECTION_TIME_MS = int(os.getenv("AMD_DETECTION_TIME_MS", "2000"))
# How long the media-stream handler waits for the AMD verdict before
# committing to the normal opening (human flow). If a 'machine_*' verdict
# arrives within this window we switch to voicemail mode.
AMD_WAIT_MS           = int(os.getenv("AMD_WAIT_MS", "1500"))
# When True, we leave a short pre-recorded message on voicemail before
# hanging up. When False, we hang up silently with hangup_reason="voicemail".
AMD_LEAVE_MESSAGE     = os.getenv("AMD_LEAVE_MESSAGE", "true").lower() not in ("0", "false", "no")
# Delay (ms) between AMD verdict and speaking the voicemail message. Gives
# the voicemail's own greeting + beep time to finish so our message lands
# AFTER the beep (i.e. inside the recording window). Typical voicemail
# greetings are 5–8 s. Raise if first half of the message gets cut off in
# the recording; lower if customers report unrecorded silence at the start.
AMD_BEEP_WAIT_MS      = int(os.getenv("AMD_BEEP_WAIT_MS", "6000"))

# ── Hangup callback (captures EVERY call's end — answered or not) ────────────
# Plivo POSTs here when the call ends. For connected calls we already have a
# transcript file written by call_handler — we just log and move on. For
# unanswered/busy/rejected calls (no transcript, no WS ever connected) we
# synthesise a minimal transcript so /transcripts shows the attempt and the
# CRM webhook gets the no-pickup signal for retry workflows.
HANGUP_CALLBACK_URL   = os.getenv("HANGUP_CALLBACK_URL", f"{NGROK_URL}/call-hangup").strip()

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
STT_SILENCE_HANGOVER_MS = int(os.getenv("STT_SILENCE_HANGOVER_MS", "500"))
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
