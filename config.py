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

# Sarvam STT — saaras:v3 is the current SOTA: 19% WER on IndicVoices (vs
# 22% on v2.5), native streaming with <150 ms time-to-first-token, 23
# languages, 8 kHz telephony-optimized, supports codemix mode for
# Hinglish. saarika:v2.5 is the legacy model and is scheduled for
# deprecation; saaras:v2.5 has been removed by Sarvam (returns 4000
# "invalid model" on connect — see 16:34 prod incident).
SARVAM_STT_MODEL    = os.getenv("SARVAM_STT_MODEL",    "saaras:v3")
SARVAM_STT_LANGUAGE = os.getenv("SARVAM_STT_LANGUAGE", "hi-IN")
# bulbul:v3 female voices: ishita (Sarvam's recommended default — best across
# Hindi/Telugu/Kannada accents), priya, simran, ritu, neha, pooja, kavya,
# shreya, roopa, suhani, kavitha, rupali, tanya, shruti.
SARVAM_VOICE        = os.getenv("SARVAM_VOICE",        "ishita")

# ── LLM ───────────────────────────────────────────────────────────────────────
# gpt-4.1 (full) — significantly better instruction-following, JSON adherence,
# and Hindi reasoning than the -mini variants. ~7-8× more expensive but worth
# it for production reliability on noisy real calls (where STT garble + LLM
# weakness compound). Set LLM_MODEL=gpt-4.1-mini in .env to fall back.
LLM_MODEL      = os.getenv("LLM_MODEL",      "gpt-4.1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Orchestrator (voice dialogue brain)
ORCHESTRATOR_TEMPERATURE    = float(os.getenv("ORCHESTRATOR_TEMPERATURE",    "0.1"))
# Fewer history messages → smaller prompt → faster LLM time-to-first-token.
# 10 keeps ~5 exchanges (enough for typical EMI collection flow) and trims
# ~150-250ms off LLM TTFT vs 18-message history.
ORCHESTRATOR_MAX_HISTORY    = int(os.getenv("ORCHESTRATOR_MAX_HISTORY",      "10"))
ORCHESTRATOR_API_RETRIES    = int(os.getenv("ORCHESTRATOR_API_RETRIES",      "3"))
# 300 tokens: trimmed from 400 — every closing template fits comfortably.
ORCHESTRATOR_MAX_TOKENS     = int(os.getenv("ORCHESTRATOR_MAX_TOKENS",       "300"))

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
# Fires for mapped outcomes:
#   payment_today_confirmed → trigger_code = "PAYMENT_VOICEBOT"
#   ptp_confirmed           → trigger_code = "PAYMENT_VOICEBOT"
#   partial_confirmed       → trigger_code = "PARTIAL_PAYMENT_VOICEBOT"
#   busy / no_answer / no_pickup / network_failure / no_response
#                          → trigger_code = "BUSY"
# Unmapped outcomes are skipped.
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

# ── Max call duration (wall-clock hard cap) ───────────────────────────────────
# Layered so a long call ends gracefully with a proper outcome, never abruptly:
#   SOFT: bot injects a wrap-up turn — confirms any commitment the customer
#         already made, else a "we'll call back — please pay soon" message —
#         then closes cleanly (hangup_reason=time_limit for the fallback case).
#   HARD: watchdog force-hangs up if the soft wrap-up itself stalls (LLM/TTS hang).
#   PLIVO_CALL_TIME_LIMIT_SEC: carrier-side backstop passed to Plivo's Call API;
#         Plivo kills the call at this duration regardless of app state. Keep it
#         above HARD. Set 0 to disable the carrier backstop.
# Counted from call-connect (media stream up). Set SOFT/HARD to 0 to disable.
MAX_CALL_SOFT_SEC         = float(os.getenv("MAX_CALL_SOFT_SEC",        "270"))  # 4:30
MAX_CALL_HARD_SEC         = float(os.getenv("MAX_CALL_HARD_SEC",        "300"))  # 5:00
PLIVO_CALL_TIME_LIMIT_SEC = int(os.getenv("PLIVO_CALL_TIME_LIMIT_SEC",  "330"))  # 5:30

# ── Call behaviour tunables ───────────────────────────────────────────────────
HANGUP_GRACE_SEC         = float(os.getenv("HANGUP_GRACE_SEC",         "1.5"))
SILENCE_TIMEOUT_SEC      = float(os.getenv("SILENCE_TIMEOUT_SEC",      "6.5"))
TTS_PACE                 = float(os.getenv("TTS_PACE",                 "1.1"))
BARGE_IN_GUARD_SEC       = float(os.getenv("BARGE_IN_GUARD_SEC",       "1.5"))

# ── STT (REST + local VAD) tunables ───────────────────────────────────────────
# Silence after speech before the utterance is POSTed to Sarvam REST.
# 400ms is the sweet spot: low enough for snappy turn latency, high enough
# that natural thinking-pauses don't fragment a sentence into 2 utterances.
# Real-call log showed 220ms was splitting "मैंने ... पेमेंट" into two,
# which then confused the LLM.
STT_SILENCE_HANGOVER_MS = int(os.getenv("STT_SILENCE_HANGOVER_MS", "400"))
# Drop bursts shorter than this (cough/pop/noise).
STT_MIN_UTTERANCE_MS    = int(os.getenv("STT_MIN_UTTERANCE_MS",    "150"))
# Force-flush long monologues. Sarvam REST caps at 30 s.
STT_MAX_UTTERANCE_SEC   = float(os.getenv("STT_MAX_UTTERANCE_SEC",  "25"))
# Consecutive VAD-speech frames required before firing the barge-in signal.
STT_BARGE_IN_MIN_FRAMES = int(os.getenv("STT_BARGE_IN_MIN_FRAMES", "12"))

# TTS concurrency — global cap on simultaneous Sarvam TTS requests across all
# active calls. Sarvam in practice chokes well below the published per-plan
# limits: 32 concurrent dials in a single second made greetings take 8-18 s
# (logs/aditi.log, 12:30 batch). 16 keeps comfortable headroom on the Pro
# plan and lets the retry+backoff in tts.py recover on the occasional 429.
# Raise carefully and only after observing real Sarvam throughput at scale.
TTS_CONCURRENCY      = int(os.getenv("TTS_CONCURRENCY",      "16"))
TTS_MAX_RETRIES      = int(os.getenv("TTS_MAX_RETRIES",      "4"))
TTS_READ_TIMEOUT_SEC = float(os.getenv("TTS_READ_TIMEOUT_SEC","20"))

# ── VAD (WebRTC noise gate) ───────────────────────────────────────────────────
VAD_MODE        = int(os.getenv("VAD_MODE",        "2"))
# 80ms = aggressive END_SPEECH for sub-1.2s turn latency. Raise to 120 if
# trailing syllables get clipped (e.g. customer's "हाँ" gets cut to "ह").
VAD_HANGOVER_MS = int(os.getenv("VAD_HANGOVER_MS", "80"))
VAD_ENABLED     = os.getenv("VAD_ENABLED", "true").lower() not in ("0", "false", "no")

# ── Spectral denoiser ─────────────────────────────────────────────────────────
# Enabled — improves STT accuracy on noisy mobile calls. Adds ~50-100ms but
# the audio-quality / STT-accuracy win is more important than the latency cost.
DENOISE_ENABLED     = os.getenv("DENOISE_ENABLED",     "true").lower() not in ("0", "false", "no")
DENOISE_STRENGTH    = float(os.getenv("DENOISE_STRENGTH",    "0.92"))
DENOISE_PROFILE_SEC = float(os.getenv("DENOISE_PROFILE_SEC", "2.0"))
DENOISE_STATIONARY  = os.getenv("DENOISE_STATIONARY", "false").lower() not in ("0", "false", "no")
