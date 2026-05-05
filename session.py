"""
session.py — CallSession dataclass and shared pending-context registry.
"""
from __future__ import annotations
import dataclasses
from typing import List

# Stores per-call context indexed by Plivo CallUUID.
# Populated by /make-call route; consumed by the media-stream WebSocket handler.
pending_ctx: dict[str, dict[str, str]] = {}

# Optional callback map (kept for compatibility with recording-callback route).
recording_pending: dict[str, str] = {}

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

    # Hangup guard — set synchronously before first await to prevent double-hangup
    _hangup_started: bool = False

    # Local audio capture buffers (PCM16, 8000 Hz, mono)
    _customer_audio: List[bytes] = dataclasses.field(default_factory=list)
    _bot_audio: List[bytes] = dataclasses.field(default_factory=list)
    # Byte offset in customer audio when bot starts speaking (for stereo sync)
    _bot_audio_offset_bytes: int = 0
