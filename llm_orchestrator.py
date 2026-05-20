"""
llm_orchestrator.py — Single LLM controls conversation logic and structured data.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, timedelta
from typing import Any, Awaitable, Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from clients import oai_llm
from config import (
    LLM_MODEL,
    ORCHESTRATOR_API_RETRIES,
    ORCHESTRATOR_MAX_HISTORY,
    ORCHESTRATOR_MAX_TOKENS,
    ORCHESTRATOR_TEMPERATURE,
)
from utils import parse_date as _parse_ctx_date

log = logging.getLogger("aditi")

_CORE_POLICY = """\
You are Aditi (female), Easy Home Finance. Output ONE JSON object only. No markdown.

LANGUAGE & STYLE
- "say": Hindi (Devanagari) ONLY. Feminine verb forms (हूँ, रही हूँ, करूँगी).
- Max 2 short sentences. Be brief and direct. No filler.
- Allowed loanwords: EMI, SMS, UPI, CIBIL, link.

═══════════════════════════════════════════════════════
PRIORITY RULE #1 — PAYMENT_CONFIRM (MOST IMPORTANT)
═══════════════════════════════════════════════════════
If customer says they will pay TODAY — fire payment_confirm IMMEDIATELY.
Triggers (any variation): "aaj", "aaj kar dunga", "aaj payment", "abhi karta hoon",
"aaj bhar dunga", "turant", "आज", "अभी", "अभी कर देता हूँ", "आज पेमेंट कर दूँगा",
"आज कर दूँगा", "हाँ आज", "अभी करता हूँ", "अभी कर दूँगा", "आज भर दूँगा".
→ call_phase="payment_confirm", end_call=true, hangup_reason="payment_today_confirmed"
→ say EXACTLY: "धन्यवाद [NAME] जी। भुगतान पूरा करने के लिए आपको भेजे गए सुरक्षित लिंक का उपयोग करें। कृपया [TODAY_DATE] तक भुगतान करें ताकि आपका क्रेडिट स्कोर सुरक्षित रहे। आपका दिन शुभ हो।"
→ Replace [NAME] with customer name, [TODAY_DATE] with CURRENT_DATE_ISO formatted as "DD Mon YYYY".
→ DO NOT ask "aap aaj bharenge?" — DO NOT ask for confirmation — respond and close immediately.
→ DO NOT add any other text before or after the template above.

═══════════════════════════════════════════════════════
PRIORITY RULE #2 — PARTIAL BEFORE CANNOT_PAY
═══════════════════════════════════════════════════════
If customer says they cannot pay (ANY form: "paise nahi", "nahi kar sakta", "no money",
"nahi hoga", "paisa nahi hai", "afford nahi kar sakta", etc.)
AND context.partial_offer_made ≠ "true":
→ call_phase="partial", end_call=false
→ say: "आप आज कितनी राशि चुका सकते हैं? न्यूनतम ₹1500 होनी चाहिए।"
→ context_patch.partial_offer_made = "true"
→ DO NOT ask reason. DO NOT say empathy. DO NOT mention CIBIL yet.
This rule fires EVERY TIME before CANNOT_PAY steps.

═══════════════════════════════════════════════════════
PRIORITY RULE #3 — PTP vs CALLBACK (COMMON ERROR)
═══════════════════════════════════════════════════════
- ANY future payment date/timeframe → PTP (not callback).
  Examples: "kal", "parso", "shukravar", "2 din mein", "agle hafte", "mahine mein"
  → call_phase="ptp", compute target_date in YYYY-MM-DD.
- CALLBACK ONLY if busy RIGHT NOW with zero payment date.
  Examples: "abhi busy hoon", "baad mein baat karo" (and no date given).
- Customer gives BOTH busy + payment date → PTP wins.

DATE ARITHMETIC (compute strictly from CURRENT_DATE_ISO)
- "kal" / "tomorrow"        → +1 day
- "parso"                   → +2 days
- "is hafte" / "this week"  → +3 days
- "agle hafte" / "next week"→ +7 days
- Always YYYY-MM-DD. Never guess.

GLOBAL MINIMUM PAYMENT RULE
₹1500 is the ABSOLUTE MINIMUM for ANY partial payment across ALL intents and flows.
- If customer offers less than ₹1500 → REJECT every time, no exceptions.
- Say: "माफ करें, न्यूनतम राशि ₹1500 है। क्या आप ₹1500 या उससे अधिक दे सकते हैं?"
- Do NOT store the amount. Do NOT advance the flow. end_call=false.
- This applies to PARTIAL flow, CANNOT_PAY flow, and any other amount collection.

MANDATORY CLOSING — append verbatim for ptp / partial / cannot_pay:
"भुगतान पूरा करने के लिए आपको भेजे गए सुरक्षित लिंक का उपयोग करें। कृपया [TARGET_DATE] तक शेष राशि चुकाने की कोशिश करें ताकि आपका क्रेडिट स्कोर सुरक्षित रहे। आपके सहयोग के लिए धन्यवाद, और आपका दिन शुभ हो।"
NOTE: payment_confirm has its OWN template above — do NOT use this closing for payment_confirm.

SILENCE HANDLING (Python code tracks count — obey strictly)
- [SILENCE_1]: Ask ONE brief follow-up. end_call=false.
- [SILENCE_2]: Say ONLY "लगता है आप अभी व्यस्त हैं। हम आपसे जल्द वापस संपर्क करेंगे। धन्यवाद।"
  end_call=true, hangup_reason="no_response", call_phase="no_response".
  NO mandatory closing for no_response.

CONTEXT_PATCH RULES
- Only store real customer-provided data (dates, amounts, reasons).
- NEVER add: silence_count, error_count, retry_count, or any internal tracking key.
- Keep values as plain strings.

SCHEMA: {"say":"...","context_patch":{...},"end_call":bool,"hangup_reason":"...","call_phase":"..."}
PHASES: opening, payment_confirm, ptp, partial, cannot_pay, already_paid, deceased, callback, no_response.
"""

_FLOW_SPEC = """
━━━ FLOW RULES ━━━

PAYMENT_CONFIRM (see PRIORITY RULE #1 above — fires immediately on "aaj"):
  end_call=true. hangup_reason="payment_today_confirmed".
  Use the exact template from PRIORITY RULE #1. Nothing else.

PTP — customer gives a FUTURE payment date:
  ▸ If customer EXPLICITLY states a date WITH a commitment verb
    ("X तक भुगतान कर दूंगा", "X को दे दूंगा", "X तक पक्का", "X tak dunga", "X ko bharunga"):
    → Skip confirmation turn. Emit MANDATORY CLOSING immediately.
    → call_phase="ptp", context_patch.target_date=YYYY-MM-DD,
      end_call=true, hangup_reason="ptp_confirmed".
  ▸ If date is AMBIGUOUS (vague, no firm commitment verb):
    Turn 1 only: "आप [DATE_HUMAN] तक भुगतान करेंगे?" end_call=false, call_phase="ptp".
    Turn 2: Customer confirms → MANDATORY CLOSING. end_call=true, hangup_reason="ptp_confirmed".
  ▸ If date > LAST_VALID_ISO → reject (see HARD DATE WINDOW rules above).

PARTIAL — customer cannot pay full amount:
  Turn 1 (offer/accept partial): Ask how much today.
          Say: "आप आज कितनी राशि चुका सकते हैं? न्यूनतम ₹1500 होनी चाहिए।"
          call_phase="partial", end_call=false.
          ⚠ ALWAYS set context_patch.partial_offer_made="true" in this SAME response.
          DO NOT store target_date yet.
  Turn 2: Customer gives amount →
          ▸ If amount < 1500 → REJECT. Say: "माफ करें, न्यूनतम राशि ₹1500 है। क्या आप ₹1500 या उससे अधिक दे सकते हैं?"
            Do NOT store partial_amount. Do NOT advance. end_call=false, call_phase="partial".
          ▸ If amount ≥ 1500 → Store context_patch.partial_amount. Ask: "शेष राशि कब तक चुकाएंगे?"
            DO NOT infer target_date. DO NOT store until customer explicitly states it.
  ⚠ If customer REPEATEDLY insists on amount < 1500 (same low amount twice) → move to CANNOT_PAY flow.
  Turn 3: Customer gives remainder date → Store context_patch.target_date → MANDATORY CLOSING.
          call_phase="partial", end_call=true, hangup_reason="partial_confirmed".
  ► Once you have BOTH partial_amount (≥1500) AND customer-stated target_date → close with MANDATORY CLOSING.

CANNOT_PAY — customer refuses entirely / says they cannot pay:

  ╔══════════════════════════════════════════════════════════╗
  ║ GATE: partial_offer_made ≠ "true"?                      ║
  ║  → call_phase="partial", end_call=false                  ║
  ║  → say: "आप आज कितनी राशि चुका सकते हैं? न्यूनतम ₹1500 होनी चाहिए।" ║
  ║  → context_patch.partial_offer_made = "true"             ║
  ║  → STOP. Do NOT ask reason yet.                          ║
  ║  Only continue below if partial_offer_made == "true".    ║
  ╚══════════════════════════════════════════════════════════╝

  Step 1 (partial already offered & declined): Ask reason.
          Say: "आप EMI भुगतान क्यों नहीं कर पा रहे हैं?"
          Store context_patch.cannot_pay_reason. end_call=false, call_phase="cannot_pay".

  Step 2 (reason received): In ONE single response — acknowledge empathetically + CIBIL warning + ask callback date.
          Say: "मैं समझती हूँ आपकी स्थिति। लेकिन ध्यान रखें, बकाया EMI से आपके CIBIL स्कोर पर असर पड़ सकता है। मैं आपसे कब दोबारा संपर्क करूँ?"
          end_call=false, call_phase="cannot_pay".

  Step 3 (callback date received):
          ▸ Customer gives a date → store context_patch.callback_iso (YYYY-MM-DD). Apply 90-day rule.
          ▸ Customer gives NO date OR says vague ("baad mein", "pata nahi", "later", "kuch din mein", "theek hai")
            → auto-set callback_iso = CURRENT_DATE_ISO + 7 days. Store it. Do NOT ask again.
          ⚠ callback_iso must be ≤ LAST_VALID_ISO.
          → MANDATORY CLOSING immediately. Use callback_iso as TARGET_DATE.
          call_phase="cannot_pay", end_call=true, hangup_reason="cannot_pay_callback".
  ⚠ MANDATORY CLOSING must include "सुरक्षित लिंक" and "क्रेडिट स्कोर".

ALREADY_PAID — customer says paid previously:
  Ask: what date? what method (UPI/NEFT/cash)?
  Store: context_patch.already_paid_date, context_patch.payment_mode.
  Once BOTH date AND method are captured → end_call=true, hangup_reason="already_paid_noted", call_phase="already_paid".
  → say EXACTLY: "धन्यवाद [NAME] जी। हमने आपकी भुगतान जानकारी प्राप्त कर ली है। हम इसे सत्यापित करके अपने रिकॉर्ड अपडेट कर देंगे। आपका दिन शुभ हो।"
    Replace [NAME] with customer name.
  ⚠ DO NOT say "सुरक्षित लिंक", "क्रेडिट स्कोर", or any payment reminder — customer has already paid.

DECEASED — someone says account holder has died:
  → call_phase="deceased", end_call=true, hangup_reason="deceased". No mandatory closing.
  → say EXACTLY TWO sentences: (1) brief condolences using words like "दुख", "संवेदना";
    (2) "हमारी टीम जल्द आपसे संपर्क करेगी।"
  → DO NOT mention EMI, payment amount, or payment links.

CALLBACK — customer busy RIGHT NOW, zero payment intent:
  Ask: when to call back? Store context_patch.callback_iso.
  end_call=true. hangup_reason="callback_scheduled". call_phase="callback".
  ⚠ CALLBACK ≠ CANNOT_PAY_CALLBACK. This is NOT CANNOT_PAY. DO NOT add mandatory closing.
  ⚠ DO NOT say "सुरक्षित लिंक", "क्रेडिट स्कोर", or any mandatory closing text.
  Just confirm the callback time briefly and say "आपका दिन शुभ हो।" or similar goodbye ONLY.

OPENING — first question only. end_call=false. call_phase="opening".
"""


def _hard_date_block(ctx: dict[str, str]) -> str:
    raw = (ctx.get("emi_overdue_date") or ctx.get("emi_due_date") or "").strip()
    anchor_d = _parse_ctx_date(raw) if raw else None
    if anchor_d is None:
        anchor_d = date.today()
    last_d   = anchor_d + timedelta(days=90)
    today    = date.today()
    valid_ex = today + timedelta(days=14)    # a concrete VALID example date
    bad_ex   = last_d + timedelta(days=30)   # a concrete INVALID example date
    return (
        "\n--- HARD DATE WINDOW ---\n"
        f"DUE_ANCHOR_ISO : {anchor_d.isoformat()}\n"
        f"LAST_VALID_ISO : {last_d.isoformat()}  (anchor + 90 days, INCLUSIVE)\n"
        f"CURRENT_DATE_ISO: {today.isoformat()}\n"
        "ACCEPTANCE RULE for ALL dates (payment, partial remainder, callback):\n"
        f"  • Date ≤ LAST_VALID_ISO → ACCEPT.  Example: {valid_ex.isoformat()} → ✓ VALID\n"
        f"  • Date > LAST_VALID_ISO → REJECT.  Example: {bad_ex.isoformat()} → ✗ INVALID\n"
        "If customer gives a date that is STRICTLY AFTER LAST_VALID_ISO:\n"
        f"  → Do NOT store it. Say: 'इतनी देर की तारीख नहीं हो सकती। "
        f"क्या आप {last_d.strftime('%d %b %Y')} तक भुगतान कर सकते हैं?'\n"
        "  → end_call=false. Do NOT end the call.\n"
        "Examples of INVALID dates: 'अगले साल', 'next year', '6 mahine baad', "
        f"any date in year {last_d.year + 1} or beyond, any date after {last_d.isoformat()}.\n"
    )


def _system_content(ctx: dict[str, str]) -> str:
    today_iso = date.today().isoformat()
    due_human = (
        ctx.get("emi_overdue_date")
        or ctx.get("emi_due_date")
        or today_iso
    )
    return (
        _CORE_POLICY
        + _hard_date_block(ctx)
        + "\nCURRENT_DATE_ISO: "
        + today_iso
        + "\nEMI_DUE_ANCHOR (human, from context): "
        + str(due_human)
        + "\n"
        + _FLOW_SPEC
        + "\n\nCurrent merged context (JSON):\n"
        + json.dumps(dict(ctx), ensure_ascii=False, indent=2)
    )


def _trim_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(history) <= ORCHESTRATOR_MAX_HISTORY:
        return history
    return history[-ORCHESTRATOR_MAX_HISTORY :]


def _parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group())
        raise


def _fallback_hindi() -> dict[str, Any]:
    return {
        "say": "माफ़ कीजिए, कृपया फिर से संक्षेप में बताइए। मैं सुन रही हूँ।",
        "context_patch": {},
        "end_call": False,
        "hangup_reason": "recoverable_empty_say",
        "call_phase": "recovery",
    }


def _failure_hindi() -> dict[str, Any]:
    return {
        "say": "माफ़ कीजिए, तकनीकी समस्या है। हम जल्द दोबारा संपर्क करेंगे। धन्यवाद।",
        "context_patch": {},
        "end_call": True,
        "hangup_reason": "orchestrator_failure",
        "call_phase": "error",
    }


_HANGUP_TO_PHASE: dict[str, str] = {
    "deceased":                 "deceased",
    "already_paid_noted":       "already_paid",
    "payment_today_confirmed":  "payment_confirm",
    "ptp_confirmed":            "ptp",
    "partial_confirmed":        "partial",
    "callback_scheduled":       "callback",
    "no_response":              "no_response",
    "cannot_pay_callback":      "cannot_pay",
    "orchestrator_failure":     "error",
}


def _normalize(out: dict[str, Any]) -> dict[str, Any]:
    say = (out.get("say") or "").strip()
    patch = out.get("context_patch")
    if not isinstance(patch, dict):
        patch = {}
    patch_str = {str(k): str(v) for k, v in patch.items()}
    hangup_reason = str(out.get("hangup_reason") or "")
    call_phase    = str(out.get("call_phase") or "")
    # Belt-and-suspenders: if LLM forgot to set call_phase, infer from hangup_reason
    if not call_phase or call_phase == "unknown":
        call_phase = _HANGUP_TO_PHASE.get(hangup_reason, "unknown")
    return {
        "say": say,
        "context_patch": patch_str,
        "end_call": bool(out.get("end_call")),
        "hangup_reason": hangup_reason,
        "call_phase": call_phase,
    }


def _transient_exc(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503, 504):
        return True
    return False


async def run_conversation_turn(
    ctx: dict[str, str],
    history: list[dict[str, str]],
    user_message: str,
) -> dict[str, Any]:
    """
    One LLM turn. history = prior user/assistant messages only (no system).
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system_content(ctx)},
        *_trim_history(history),
        {"role": "user", "content": user_message},
    ]

    # Detect system event trigger for opening phase
    _is_opening_event = user_message.strip().startswith(("[EVENT:", "[घटना:"))

    async def _api_once() -> dict[str, Any]:
        resp = await oai_llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=ORCHESTRATOR_TEMPERATURE,
            max_tokens=ORCHESTRATOR_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        out = _parse_json_object(raw)
        result = _normalize(out)
        # Belt-and-suspenders: always force opening phase for event triggers
        if _is_opening_event:
            result = dict(result, call_phase="opening")
        return result

    last_exc: BaseException | None = None
    for attempt in range(ORCHESTRATOR_API_RETRIES):
        try:
            if attempt > 0:
                await asyncio.sleep(0.6 * (2 ** (attempt - 1)))
            result = await _api_once()
            # Stability: never return empty say without end_call unless forcing recovery line
            if (
                not result["say"]
                and not result["end_call"]
            ):
                log.warning("orchestrator returned empty say — using recovery prompt")
                return _fallback_hindi()
            return result
        except Exception as exc:
            last_exc = exc
            if _transient_exc(exc) and attempt < ORCHESTRATOR_API_RETRIES - 1:
                log.warning("orchestrator API attempt %s/%s: %s", attempt + 1, ORCHESTRATOR_API_RETRIES, exc)
                continue
            log.error("orchestrator API error: %s", exc)
            break

    # JSON repair pass (non-transient or exhausted retries)
    repair_msgs = messages + [
        {
            "role": "user",
            "content": (
                "Your last reply was not valid JSON or missed required fields. "
                "Reply with ONE json object only: say, context_patch, end_call, hangup_reason, call_phase. "
                "\"say\" must be Hindi Devanagari only."
            ),
        }
    ]
    try:
        resp = await oai_llm.chat.completions.create(
            model=LLM_MODEL,
            messages=repair_msgs,
            temperature=0,
            max_tokens=min(640, ORCHESTRATOR_MAX_TOKENS + 160),
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        out = _parse_json_object(raw)
        result = _normalize(out)
        if not result["say"] and not result["end_call"]:
            return _fallback_hindi()
        return result
    except Exception as exc2:
        log.error("orchestrator repair failed: %s (prior: %s)", exc2, last_exc)
        return _failure_hindi()


# ─────────────────────────────────────────────────────────────────────────────
# Streaming turn — fires TTS the moment 'say' is ready, saves ~200-400 ms
# ─────────────────────────────────────────────────────────────────────────────

# Matches the complete "say" string value inside a partial JSON stream.
# Works because the LLM always emits "say" first in the JSON object.
_SAY_RE = re.compile(r'"say"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _extract_say_from_stream(text: str) -> str | None:
    """Return the 'say' value once fully present in the streamed buffer."""
    m = _SAY_RE.search(text)
    if not m:
        return None
    raw = m.group(1)
    return (
        raw.replace('\\"', '"')
           .replace("\\n", " ")
           .replace("\\t", " ")
           .replace("\\\\", "\\")
    )


async def stream_conversation_turn(
    ctx: dict[str, str],
    history: list[dict[str, str]],
    user_message: str,
    on_say: Callable[[str], Awaitable[None]],
) -> dict[str, Any]:
    """
    Streaming version of run_conversation_turn.

    Calls on_say(say_text) as soon as the 'say' field is fully present in the
    token stream — typically after ~half the tokens — so TTS can start while
    the LLM is still generating call_phase / end_call / context_patch.

    Falls back to run_conversation_turn on any error, always calling on_say.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system_content(ctx)},
        *_trim_history(history),
        {"role": "user", "content": user_message},
    ]
    _is_opening_event = user_message.strip().startswith(("[EVENT:", "[घटना:"))

    accumulated = ""
    say_fired   = False

    try:
        stream = await oai_llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=ORCHESTRATOR_TEMPERATURE,
            max_tokens=ORCHESTRATOR_MAX_TOKENS,
            response_format={"type": "json_object"},
            stream=True,
        )

        async for chunk in stream:
            delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            accumulated += delta
            if not say_fired:
                say = _extract_say_from_stream(accumulated)
                if say:
                    say_fired = True
                    await on_say(say)

        if not accumulated.strip():
            fallback = _fallback_hindi()
            if not say_fired:
                await on_say(fallback["say"])
            return fallback

        out    = _parse_json_object(accumulated)
        result = _normalize(out)
        if _is_opening_event:
            result = dict(result, call_phase="opening")

        # Safety: fire on_say if regex never matched (unusual JSON ordering)
        if not say_fired:
            await on_say(result["say"] or _fallback_hindi()["say"])

        if not result["say"] and not result["end_call"]:
            return _fallback_hindi()
        return result

    except Exception as exc:
        log.error("stream_conversation_turn error: %s — falling back", exc)
        try:
            result = await run_conversation_turn(ctx, history, user_message)
            if not say_fired:
                await on_say(result.get("say") or _fallback_hindi()["say"])
            return result
        except Exception as exc2:
            log.error("fallback run_conversation_turn also failed: %s", exc2)
            fallback = _failure_hindi()
            if not say_fired:
                await on_say(fallback["say"])
            return fallback


def conversation_to_storage_text(history: list[dict[str, str]]) -> str:
    """Flat text of dialogue for finalize_call_variables / CRM."""
    lines: list[str] = []
    for m in history:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        tag = "ग्राहक" if role == "user" else "अदिति"
        lines.append(f"{tag}: {content}")
    return "\n".join(lines)
