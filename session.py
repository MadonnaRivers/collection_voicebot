"""
session.py — CallSession dataclass and shared pending-context registry.
"""
from __future__ import annotations
import dataclasses

# Stores per-call context indexed by Plivo CallUUID.
# Populated by /make-call route; consumed by the media-stream WebSocket handler.
pending_ctx: dict[str, dict[str, str]] = {}

# Optional callback map (kept for compatibility with recording-callback route).
recording_pending: dict[str, str] = {}

# Async AMD (Answering Machine Detection) results keyed by Plivo CallUUID.
# Populated by /amd-callback when Plivo posts back its verdict. Consumed by
# call_handler.media_stream to decide voicemail vs. human-call flow.
# Values: "human" | "machine_start" | "machine_end_beep" | "machine_end_silence"
#         | "machine_end_other" | "fax" | "unknown" | other strings.
amd_results: dict[str, str] = {}

@dataclasses.dataclass
class CallSession:
    ctx:             dict[str, str]  # customer data + dynamic per-turn values

    stream_sid:      str   = ""
    call_sid:        str   = ""
    state:           str   = "llm"   # last LLM call_phase for logs
    done:            bool  = False
    speaking:        bool  = False
    tts_started_at:  float = 0.0    # monotonic time when current TTS play started
    marks_out:       int   = 0
    last_queued:     str   = ""
    last_interim:    str   = ""
    transcript_path: str   = ""

    # LLM-only conversation: OpenAI-style messages (user/assistant), no system stored here
    llm_history:     list[dict[str, str]] = dataclasses.field(default_factory=list)
    greeting_sent:   bool = False
    # False until first bot utterance finishes — no barge-in during opening
    opening_complete: bool = False
    # True while playing terminal goodbye — no barge-in during closing
    closing_in_progress: bool = False

    barge_in_active:   bool = False # True while collecting post-barge-in utterance

    # Monotonic time when STT last detected customer speech-start. While >0,
    # the silence-prompt timer in the LLM loop pauses so we don't fire
    # "हैलो आप वहाँ हैं?" over a customer who's actively mid-sentence.
    # Cleared once the utterance is transcribed (or stale-cleared after grace).
    last_speech_started_at: float = 0.0

    # Hangup guard — set synchronously before first await to prevent double-hangup
    _hangup_started: bool = False

    # (audio capture buffers removed — Plivo server-side recording is used instead)
