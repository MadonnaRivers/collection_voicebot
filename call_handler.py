"""
call_handler.py — WebSocket media-stream handler + LLM conversation loop.

Receives raw µ-law audio from Plivo, pipes it through:
  denoiser → VAD → Sarvam STT (WebSocket)
All dialogue logic and structured data capture run through llm_orchestrator (OpenAI).
"""
from __future__ import annotations

import asyncio
import audioop
import concurrent.futures
import io
import json
import logging
import time
import wave
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import webrtcvad
import websockets
import websockets.exceptions as ws_exc
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

import carrier as _carrier
from config import (
    SARVAM_API_KEY, SARVAM_STT_WS_URL,
    VAD_ENABLED, VAD_MODE, VAD_HANGOVER_MS,
    DENOISE_ENABLED, DENOISE_STRENGTH, DENOISE_PROFILE_SEC, DENOISE_STATIONARY,
    HANGUP_GRACE_SEC, SILENCE_TIMEOUT_SEC, BARGE_IN_GUARD_SEC,
    POST_UTTERANCE_PAUSE_SEC, TRANSCRIPTS_DIR,
    CALL_SUMMARY_WEBHOOK_URL, AUDIO_TRANSCRIPT_WEBHOOK_URL,
)
from clients import http as _http_client
from classifier import finalize_call_variables
from call_webhook import build_call_summary_push_body, push_call_summary_webhook
from denoiser import StreamDenoiser
from llm_orchestrator import run_conversation_turn, conversation_to_storage_text
from scripts import build_opening_greeting
from session import CallSession, pending_ctx
from tts import tts_rest, tts_stream_pipelined, _strip_wav_header

log = logging.getLogger("aditi")

_VAD_HANGOVER_FRAMES = max(1, VAD_HANGOVER_MS // 20)


def _save_call_audio(sess: "CallSession") -> dict[str, str]:
    """Build a single mixed call WAV buffer for upload."""
    cust_pcm = b"".join(sess._customer_audio)
    bot_pcm = b"\x00" * max(0, sess._bot_audio_offset_bytes) + b"".join(sess._bot_audio)
    if not cust_pcm and not bot_pcm:
        return {}

    max_len = max(len(cust_pcm), len(bot_pcm))
    if max_len % 2:
        max_len += 1
    cust_pcm = cust_pcm.ljust(max_len, b"\x00")
    bot_pcm = bot_pcm.ljust(max_len, b"\x00")

    # Mix both sides into one mono stream for phone-call style playback.
    cust_mix = audioop.mul(cust_pcm, 2, 0.5)
    bot_mix = audioop.mul(bot_pcm, 2, 0.5)
    mixed_pcm = audioop.add(cust_mix, bot_mix, 2)

    combined_wav_buf = io.BytesIO()
    with wave.open(combined_wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(mixed_pcm)
    combined_wav = combined_wav_buf.getvalue()

    transcript_stem = Path(sess.transcript_path).stem if sess.transcript_path else ""
    call_sid = (sess.call_sid or "unknown").replace("/", "_")
    loan_id = (
        sess.ctx.get("loan_id")
        or sess.ctx.get("loan_account_id")
        or sess.ctx.get("loan_account_no")
        or ""
    ).strip()
    ident = transcript_stem or (f"{loan_id}_{call_sid}" if loan_id else call_sid)

    return {
        "id": ident,
        "loan_id": loan_id,
        "combined_wav_bytes": combined_wav,
    }


async def _upload_audio_to_n8n(sess: "CallSession") -> bool:
    if not AUDIO_TRANSCRIPT_WEBHOOK_URL:
        return False
    payload = _save_call_audio(sess)
    if not payload:
        return False
    ts_str = datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z"
    file_name = f"{payload['id']}_combined.wav"
    files = {
        "file": (file_name, payload["combined_wav_bytes"], "audio/wav"),
    }
    data = {
        "call_sid": sess.call_sid,
        "loan_id": payload.get("loan_id", ""),
        "id": payload["id"],
        "ts": ts_str,
    }
    r = await _http_client.post(AUDIO_TRANSCRIPT_WEBHOOK_URL, data=data, files=files, timeout=90.0)
    log.info("audio_and_transcripts webhook upload → HTTP %d call=%s", r.status_code, sess.call_sid)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket handler
# ─────────────────────────────────────────────────────────────────────────────
async def media_stream(ws: WebSocket) -> None:
    await ws.accept()
    log.info("Plivo media stream connected")

    from scripts import build_default_ctx
    sess  = CallSession(ctx=build_default_ctx())
    abort = [False]
    _fallback_task: list[asyncio.Task | None] = [None]
    utt_q: asyncio.Queue[str] = asyncio.Queue()
    drained = asyncio.Event()
    drained.set()

    _denoiser = (
        StreamDenoiser(
            prop_decrease=DENOISE_STRENGTH,
            profile_sec=DENOISE_PROFILE_SEC,
            stationary=DENOISE_STATIONARY,
        )
        if DENOISE_ENABLED else None
    )
    _denoise_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _loop = asyncio.get_event_loop()

    _vad_inst           = webrtcvad.Vad(VAD_MODE) if VAD_ENABLED else None
    _vad_hangover_left  = [0]

    # ── Connect to Sarvam STT with retries ──────────────────────────────────
    stt_ws = None
    for _attempt in range(4):
        try:
            stt_ws = await asyncio.wait_for(
                websockets.connect(
                    SARVAM_STT_WS_URL,
                    extra_headers={"Api-Subscription-Key": SARVAM_API_KEY},
                    ping_interval=20,
                    ping_timeout=10,
                ),
                timeout=8.0,
            )
            log.info("Sarvam STT connected (attempt %d)", _attempt + 1)
            break
        except Exception as exc:
            log.warning("STT connect attempt %d/4 failed: %s", _attempt + 1, exc)
            if _attempt < 3:
                await asyncio.sleep(0.8)

    if stt_ws is None:
        log.error("Cannot connect to Sarvam STT after 4 attempts — dropping call")
        await ws.close()
        return

    try:
        # ── Transcript logger ────────────────────────────────────────────────
        def record(event: str, **fields: Any) -> None:
            if not sess.transcript_path:
                ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                sid = sess.call_sid.replace("/", "_") or "unknown"
                Path(TRANSCRIPTS_DIR).mkdir(parents=True, exist_ok=True)
                sess.transcript_path = f"{TRANSCRIPTS_DIR}/{ts}_{sid}.jsonl"
                if sess.call_sid:
                    from session import recording_pending
                    recording_pending[sess.call_sid] = sess.transcript_path
            row = {
                "ts":    datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z",
                "event": event, "state": sess.state, "sid": sess.call_sid, **fields,
            }
            try:
                with open(sess.transcript_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError:
                pass

        # ── Push µ-law frames to carrier ─────────────────────────────────────
        _push_frame_count   = [0]
        _utt_push_start     = [0.0]   # monotonic when current utterance push started
        _utt_bytes_pushed   = [0]     # bytes sent in current utterance
        _audio_drain_time   = [0.0]   # estimated monotonic when Plivo finishes playing

        async def push(chunk: bytes) -> None:
            if sess.done or abort[0] or not sess.stream_sid:
                return
            try:
                for offset in range(0, len(chunk), 160):
                    if sess.done or abort[0]:
                        return
                    frame = chunk[offset:offset + 160]
                    if not frame:
                        return
                    await ws.send_json(
                        _carrier.media_msg(frame, sess.stream_sid)
                    )
                    _push_frame_count[0] += 1
                    _utt_bytes_pushed[0] += len(frame)
                    # Update estimated Plivo drain time: push_start + total_audio_duration
                    _audio_drain_time[0] = _utt_push_start[0] + _utt_bytes_pushed[0] / 8000.0
                    if _push_frame_count[0] == 1:
                        sess._bot_audio_offset_bytes = sum(len(b) for b in sess._customer_audio)
                    try:
                        sess._bot_audio.append(audioop.ulaw2lin(frame, 2))
                    except Exception:
                        pass
                    await asyncio.sleep(0)
            except Exception as exc:
                log.error("push error after %d frames: %s", _push_frame_count[0], exc)

        async def send_mark() -> None:
            if sess.done or not sess.stream_sid:
                return
            msg = _carrier.mark_msg(sess.stream_sid)
            if msg is None:
                # Carrier has no mark support — treat audio as immediately drained
                drained.set()
                return
            drained.clear()
            sess.marks_out += 1
            try:
                await ws.send_json(msg)
            except Exception:
                pass

        async def send_clear() -> None:
            if sess.done or not sess.stream_sid:
                return
            msg = _carrier.clear_msg(sess.stream_sid)
            sess.marks_out = 0
            drained.set()
            if msg is None:
                return  # Carrier has no clear support — just reset local state
            try:
                await ws.send_json(msg)
            except Exception:
                pass

        def _cancel_fallback() -> None:
            t = _fallback_task[0]
            if t and not t.done():
                t.cancel()
            _fallback_task[0] = None

        async def _queue_after_silence(text: str) -> None:
            """POST_UTTERANCE_PAUSE_SEC silence timer — primary utterance trigger."""
            try:
                await asyncio.sleep(POST_UTTERANCE_PAUSE_SEC)
            except asyncio.CancelledError:
                return
            sess.barge_in_active = False
            if text and not sess.done and text != sess.last_queued:
                log.info("[USER] %s", text)
                sess.last_queued  = text
                sess.last_interim = ""
                record("user", text=text)
                await utt_q.put(text)

        # ── TTS ───────────────────────────────────────────────────────────────
        async def play_tts(text: str) -> None:
            text = text.strip()
            if not text or sess.done or not sess.stream_sid:
                return
            t0 = time.perf_counter()
            abort[0] = False
            sess.barge_in_active = False
            sess.tts_started_at   = time.monotonic()
            _utt_push_start[0]   = time.monotonic()  # track when this utterance push begins
            _utt_bytes_pushed[0] = 0                  # reset per-utterance byte counter
            sess.speaking = True
            try:
                # Use pipelined TTS for lower latency on long scripts
                ok = await tts_stream_pipelined(text, push, abort)
                if not ok and not sess.done:
                    log.warning("TTS stream failed — REST fallback")
                    audio = await tts_rest(text)
                    await push(_strip_wav_header(audio))
                if not sess.done:
                    await send_mark()
            except Exception as exc:
                log.error("play_tts error: %s", exc)
            finally:
                sess.speaking = False
            log.info("TTS %.0f ms | %.60s", (time.perf_counter() - t0) * 1000, text)

        class _SafeMap(dict):
            def __missing__(self, key: str) -> str:
                return f"{{{key}}}"

        async def speak(text: str) -> None:
            text = text.format_map(_SafeMap(sess.ctx)).strip()
            if not text:
                return
            log.info("[ADITI] %s", text)
            record("bot", text=text)
            await play_tts(text)
            sess.opening_complete = True  # allow barge-in after first bot audio finishes

        # ── Hangup ───────────────────────────────────────────────────────────
        async def hangup(reason: str = "unknown") -> None:
            if sess.done or sess._hangup_started:
                return
            sess._hangup_started = True  # set before first await — prevents double-hangup
            # Do NOT set sess.done yet — recv_ws must keep running to
            # receive the carrier's mark event so drained can fire.
            abort[0] = True   # stop any future TTS from sending audio
            log.info("Hangup: %s", reason)
            record("hangup", reason=reason)
            if sess.marks_out > 0:
                try:
                    await asyncio.wait_for(drained.wait(), timeout=12.0)
                except asyncio.TimeoutError:
                    log.warning("Hangup: audio drain timeout")
            else:
                # Plivo has no mark/drain events — wait until estimated audio playback
                # finishes before terminating (otherwise call drops while audio still plays)
                remaining = _audio_drain_time[0] - time.monotonic()
                if remaining > 0.1:
                    log.info("Hangup: waiting %.1f s for Plivo audio drain", remaining)
                    await asyncio.sleep(remaining)
            # Now safe to stop recv_ws
            sess.done = True
            call_vars: dict[str, Any] = {}
            try:
                call_vars = await finalize_call_variables(
                    reason,
                    sess.ctx,
                    conversation_to_storage_text(sess.llm_history),
                ) or {}
                if call_vars:
                    record("call_summary", **call_vars)
            except Exception as exc:
                log.warning("call vars error: %s", exc)
            if CALL_SUMMARY_WEBHOOK_URL and sess.call_sid:
                wh_body = build_call_summary_push_body(
                    sess.call_sid,
                    reason,
                    call_vars,
                    ctx=sess.ctx,
                    state=sess.state,
                )
                await push_call_summary_webhook(CALL_SUMMARY_WEBHOOK_URL, wh_body)
            try:
                uploaded = await _upload_audio_to_n8n(sess)
                if uploaded:
                    record("recording_uploaded", webhook=AUDIO_TRANSCRIPT_WEBHOOK_URL)
            except Exception as exc:
                log.warning("Audio upload to n8n failed: %s", exc)
            await asyncio.sleep(HANGUP_GRACE_SEC)
            if sess.call_sid:
                try:
                    await _carrier.hangup(sess.call_sid)
                except Exception as exc:
                    log.error("Carrier hangup error: %s", exc)
            for _sock in (stt_ws, ws):
                try:
                    await _sock.close()
                except Exception:
                    pass

        # ── Carrier WebSocket receiver ────────────────────────────────────────
        async def recv_ws() -> None:
            import base64 as _b64
            try:
                async for raw in ws.iter_text():
                    if sess.done:
                        break
                    data     = json.loads(raw)
                    evt_type, payload = _carrier.parse_ws_frame(data)

                    if evt_type == "call_start":
                        sess.stream_sid = payload["stream_sid"]
                        sess.call_sid   = payload["call_sid"]
                        if sess.call_sid in pending_ctx:
                            sess.ctx = pending_ctx.pop(sess.call_sid)
                        log.info("Stream=%s Call=%s", sess.stream_sid, sess.call_sid)
                        record("call_start")

                    elif evt_type == "audio_frame":
                        if sess.done:
                            continue
                        mulaw = payload["mulaw"]
                        try:
                            pcm16 = audioop.ulaw2lin(mulaw, 2)
                            sess._customer_audio.append(pcm16)
                            if _denoiser is not None:
                                pcm16 = await _loop.run_in_executor(
                                    _denoise_pool, _denoiser.feed_sync, pcm16
                                )
                            if _vad_inst is not None and len(pcm16) == 320:
                                try:
                                    is_speech = _vad_inst.is_speech(pcm16, 8000)
                                except Exception:
                                    is_speech = True
                                if is_speech:
                                    _vad_hangover_left[0] = _VAD_HANGOVER_FRAMES
                                elif _vad_hangover_left[0] > 0:
                                    _vad_hangover_left[0] -= 1
                                else:
                                    pcm16 = b"\x00" * 320
                            await stt_ws.send(json.dumps({
                                "audio": {
                                    "data":        _b64.b64encode(pcm16).decode(),
                                    "sample_rate": 8000,
                                    "encoding":    "audio/wav",
                                }
                            }))
                        except (ws_exc.ConnectionClosedOK, ws_exc.ConnectionClosed):
                            break
                        except Exception as exc:
                            log.error("STT send error: %s", exc)
                            break

                    elif evt_type == "mark_ack":
                        if sess.marks_out > 0:
                            sess.marks_out -= 1
                        if sess.marks_out <= 0:
                            sess.marks_out = 0
                            drained.set()

                    # evt_type == "ignore" → nothing to do

            except WebSocketDisconnect:
                log.info("Carrier WS disconnected")
            except RuntimeError as exc:
                if "WebSocket is not connected" not in str(exc):
                    raise
            finally:
                try:
                    await stt_ws.close()
                except Exception:
                    pass

        # ── Sarvam STT receiver ──────────────────────────────────────────────
        def _barge_in_allowed() -> bool:
            """Only mid-call: after opening finishes and before/at closing — not during opening or closing audio."""
            return sess.opening_complete and not sess.closing_in_progress

        async def recv_sarvam_stt() -> None:
            nonlocal stt_ws
            try:
                async for msg in stt_ws:
                    if sess.done:
                        break
                    if isinstance(msg, bytes):
                        continue
                    try:
                        frame = json.loads(msg)
                    except Exception:
                        continue

                    msg_type = str(frame.get("type", "")).lower()
                    inner    = frame.get("data") if isinstance(frame.get("data"), dict) else {}

                    if msg_type == "events":
                        signal = str(inner.get("signal_type", "")).upper()
                        if signal == "START_SPEECH" and sess.speaking:
                            _guard_elapsed = time.monotonic() - sess.tts_started_at
                            if not _barge_in_allowed():
                                log.debug(
                                    "Barge-in suppressed (opening=%s closing=%s)",
                                    not sess.opening_complete,
                                    sess.closing_in_progress,
                                )
                            elif _guard_elapsed < BARGE_IN_GUARD_SEC:
                                log.debug("Barge-in suppressed (guard %.0f ms)", _guard_elapsed * 1000)
                            else:
                                log.info("Barge-in detected — aborting TTS")
                                abort[0] = True
                                sess.barge_in_active = True
                                await send_clear()
                        elif signal == "END_SPEECH":
                            _cancel_fallback()
                            sess.barge_in_active = False
                            pending = sess.last_interim.strip()
                            if pending and not sess.done and pending != sess.last_queued:
                                log.info("[USER END_SPEECH] %s", pending)
                                sess.last_queued  = pending
                                sess.last_interim = ""
                                record("user", text=pending)
                                await utt_q.put(pending)
                        continue

                    if msg_type == "error":
                        log.error("STT error: %s", frame)
                        continue

                    transcript = (
                        inner.get("transcript")
                        or frame.get("transcript")
                        or frame.get("text")
                        or ""
                    ).strip()

                    is_final = bool(
                        msg_type == "data"
                        or frame.get("is_final")
                        or frame.get("speech_final")
                        or frame.get("final")
                    )

                    if not transcript:
                        continue

                    if is_final:
                        if sess.done or transcript == sess.last_queued:
                            continue
                        if sess.speaking:
                            _guard_elapsed = time.monotonic() - sess.tts_started_at
                            if not _barge_in_allowed():
                                log.debug(
                                    "Barge-in suppressed (opening/closing): %s",
                                    transcript,
                                )
                                continue
                            elif _guard_elapsed < BARGE_IN_GUARD_SEC:
                                log.debug("Barge-in suppressed (guard %.0f ms): %s", _guard_elapsed * 1000, transcript)
                                continue
                            else:
                                abort[0] = True
                                await send_clear()
                                sess.barge_in_active = True
                                log.info("Barge-in detected via is_final")
                        if transcript != sess.last_interim:
                            sess.last_interim = transcript
                        _cancel_fallback()
                        _fallback_task[0] = asyncio.create_task(
                            _queue_after_silence(transcript)
                        )
                        if sess.barge_in_active:
                            log.info("Barge-in fragment (timer armed): %s", transcript)
                        else:
                            log.debug("[USER~final] %s", transcript)
                    else:
                        if transcript != sess.last_interim:
                            log.debug("[USER~] %s", transcript)
                            sess.last_interim = transcript

            except Exception as exc:
                if not sess.done:
                    log.error("STT recv error: %s", exc)
            finally:
                if not sess.done:
                    # ── Try to reconnect STT once before ending the call ──────
                    log.warning("STT WebSocket dropped — attempting reconnect …")
                    reconnected = False
                    for _r in range(3):
                        try:
                            new_ws = await asyncio.wait_for(
                                websockets.connect(
                                    SARVAM_STT_WS_URL,
                                    extra_headers={"Api-Subscription-Key": SARVAM_API_KEY},
                                    ping_interval=20,
                                    ping_timeout=10,
                                ),
                                timeout=6.0,
                            )
                            stt_ws = new_ws
                            reconnected = True
                            log.info("STT reconnected (attempt %d)", _r + 1)
                            # Re-enter the receive loop with the new socket
                            async for msg in stt_ws:
                                if sess.done:
                                    break
                                if isinstance(msg, bytes):
                                    continue
                                # (minimal passthrough — just keep the call alive)
                            break
                        except Exception as exc2:
                            log.warning("STT reconnect attempt %d failed: %s", _r + 1, exc2)
                            await asyncio.sleep(0.5)
                    if not reconnected and not sess.done:
                        log.error("STT reconnect failed — ending call")
                        asyncio.create_task(hangup("stt_failure"))

        # ── LLM conversation (all logic except STT/TTS) ───────────────────
        async def llm_conversation_loop() -> None:
            for _ in range(400):
                if sess.done:
                    return
                if sess.call_sid and sess.stream_sid:
                    break
                await asyncio.sleep(0.05)
            else:
                log.error("call_start never arrived — cannot run LLM loop")
                return

            async def _apply_turn(
                turn: dict[str, Any],
                user_msg_for_history: str | None,
            ) -> bool:
                patch = turn.get("context_patch")
                if isinstance(patch, dict):
                    sess.ctx.update({str(k): str(v) for k, v in patch.items()})
                sess.state = str(turn.get("call_phase") or "llm")
                record(
                    "llm_turn",
                    call_phase=turn.get("call_phase"),
                    end_call=bool(turn.get("end_call")),
                    hangup_reason=turn.get("hangup_reason"),
                    context_patch=turn.get("context_patch"),
                )
                say = (turn.get("say") or "").strip()
                if user_msg_for_history is not None:
                    sess.llm_history.append({"role": "user", "content": user_msg_for_history})
                if say:
                    if turn.get("end_call"):
                        sess.closing_in_progress = True  # no barge-in during final goodbye
                    await speak(say)
                    sess.llm_history.append({"role": "assistant", "content": say})
                elif turn.get("end_call"):
                    log.warning("LLM requested end_call with empty say")
                if turn.get("end_call"):
                    hr = str(turn.get("hangup_reason") or "llm_terminal")
                    asyncio.create_task(hangup(hr))
                    return True
                return False

            if not sess.greeting_sent:
                # ── Instant opening: no LLM round-trip, fires in milliseconds ──
                opening_text = build_opening_greeting(sess.ctx)
                turn: dict[str, Any] = {
                    "say":           opening_text,
                    "context_patch": {},
                    "end_call":      False,
                    "hangup_reason": "",
                    "call_phase":    "opening",
                }
                sess.greeting_sent = True
                if await _apply_turn(turn, None):
                    return

            while not sess.done:
                try:
                    utterance = await asyncio.wait_for(utt_q.get(), timeout=SILENCE_TIMEOUT_SEC)
                except asyncio.TimeoutError:
                    if sess.done:
                        break
                    log.info("LLM: %.0f s silence", SILENCE_TIMEOUT_SEC)
                    turn = await run_conversation_turn(
                        sess.ctx,
                        sess.llm_history,
                        "[घटना: मौन — ग्राहक ने उत्तर नहीं दिया। संक्षेप में फिर से पूछें या वसूली नीति के अनुसार "
                        "विनम्र समापन। बोलने योग्य पूरा उत्तर केवल हिंदी देवनागरी में दें।]",
                    )
                    if await _apply_turn(turn, "[मौन — कोई उत्तर नहीं]"):
                        break
                    continue

                if sess.done:
                    break

                while not utt_q.empty():
                    try:
                        utterance = utt_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                utterance = utterance.strip() or "[silence]"
                record("user_turn", text=utterance)
                turn = await run_conversation_turn(sess.ctx, sess.llm_history, utterance)
                if await _apply_turn(turn, utterance):
                    break

            log.info("LLM loop ended (phase=%s)", sess.state)

        await asyncio.gather(recv_ws(), recv_sarvam_stt(), llm_conversation_loop())

    finally:
        _denoise_pool.shutdown(wait=False)
        try:
            await stt_ws.close()
        except Exception:
            pass
        log.info("Media stream handler closed")
