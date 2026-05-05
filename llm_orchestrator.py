"""
llm_orchestrator.py — Single LLM controls conversation logic and structured data.

Plivo-style flow specs are embedded below; STT/TTS stay outside.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, timedelta
from typing import Any

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

# ── Mandatory closing (Hindi) ─────────────────────────────────────────────────
# Append verbatim to "say" for: ptp, partial, payment_confirm, cannot_pay flows.
# Replace [TARGET_DATE] with the actual stored date (dd Month YYYY, e.g. 15 May 2026).
# If no date available use "जल्द से जल्द".
_MANDATORY_CLOSING_HINDI = (
    "भुगतान पूरा करने के लिए आपको भेजे गए सुरक्षित लिंक का उपयोग करें। "
    "कृपया [TARGET_DATE] तक शेष राशि चुकाने की कोशिश करें "
    "ताकि आपका क्रेडिट स्कोर सुरक्षित रहे। "
    "आपके सहयोग के लिए धन्यवाद, और आपका दिन शुभ हो।"
)

# ── Core: JSON + Hindi speech ─────────────────────────────────────────────────
_CORE_POLICY = """\
You are Aditi (female agent), Easy Home Finance. Output exactly ONE JSON object per turn — no markdown, no text outside JSON.

LANGUAGE FOR "say"
- Customer hears only "say": Hindi in Devanagari ONLY. Feminine verb forms (समझती हूँ, बोल रही हूँ — never masculine).
- Keep each reply concise (2–4 short sentences) unless delivering mandatory closing — brevity reduces call latency.
- Allowed English loanwords only: EMI, SMS, UPI, NEFT, CIBIL, link, branch, amount. All other words must be Hindi.

MANDATORY CLOSING
Append to "say" verbatim when ending ptp / partial / payment_confirm / cannot_pay flows:
"भुगतान पूरा करने के लिए आपको भेजे गए सुरक्षित लिंक का उपयोग करें। कृपया [TARGET_DATE] तक शेष राशि चुकाने की कोशिश करें ताकि आपका क्रेडिट स्कोर सुरक्षित रहे। आपके सहयोग के लिए धन्यवाद, और आपका दिन शुभ हो।"
Replace [TARGET_DATE] with the actual date formatted as "15 May 2026" style. If no date, use "जल्द से जल्द".
Do NOT append mandatory closing for: deceased, already_paid, callback flows.

HARD DATE WINDOW
- DUE_ANCHOR_ISO and LAST_VALID_ISO are computed below. ALL stored payment dates MUST satisfy: DUE_ANCHOR_ISO ≤ date ≤ LAST_VALID_ISO.
- Date AFTER LAST_VALID_ISO → reject, explain the 90-day limit, ask for date within the window.
- Relative phrases ("दो दिन बाद", "अगले हफ्ते"): resolve from CURRENT_DATE_ISO, then clamp to window if needed.
- Day-only ("15 तारीख") → treat as that day in current month; if already past, treat as next month.
- Store all captured dates as YYYY-MM-DD in context_patch.

PARTIAL-FIRST PRIORITY
- When user says they cannot / don't have money AND context partial_offer_made ≠ "true" AND this is NOT already_paid / deceased / pay-today / pure dispute about EMI amount:
  → MUST offer partial payment first (call_phase = "partial"). Ask if they can pay at least {min_partial_int} rupees today.
  → Set context_patch.partial_offer_made = "true" in that SAME turn's context_patch.
- Only proceed to CANNOT_PAY after partial was offered and user declines or explicitly cannot meet minimum.

SILENCE / UNCLEAR HANDLING
- [मौन] or unintelligible input → set context_patch.silence_count = current silence_count + 1.
  - silence_count 1–2: ask once more concisely ("कृपया फिर से बताइए।").
  - silence_count ≥ 3: say "ठीक है, हम जल्द दोबारा संपर्क करेंगे। धन्यवाद।" and end_call = true, hangup_reason = "no_response".

CALL_PHASE ASSIGNMENT (mandatory — NEVER output call_phase = "unknown" or omit it)
Determine call_phase from the current turn:
- User message starts with "[EVENT:" or "[घटना:" → call_phase = "opening"
- Customer confirms paying full EMI today (e.g. "आज भर दूंगा", "अभी भरता हूँ") → call_phase = "payment_confirm"
- Customer gives a future payment date → call_phase = "ptp"
- Customer says cannot pay / no money AND partial_offer_made ≠ "true" → call_phase = "partial"
- Customer declines partial and cannot pay → call_phase = "cannot_pay"
- Customer says already paid → call_phase = "already_paid"
- Borrower reported deceased → call_phase = "deceased"
- Customer is busy / asks to call back later → call_phase = "callback"
- Anything else → call_phase = "other"
If none of the above match clearly, use "other". NEVER use "unknown".

CONTEXT KEYS TO STORE
target_date (YYYY-MM-DD), payment_commitment_iso (YYYY-MM-DD), already_paid_date (YYYY-MM-DD),
callback_iso (YYYY-MM-DD), partial_amount (string), cannot_pay_reason (string),
partial_offer_made ("true"), payment_mode (string), silence_count (string integer).

OUTPUT SCHEMA (exact keys — no extra fields outside this object)
{
  "say": "<Hindi Devanagari only>",
  "context_patch": { "<key>": "<value>", ... },
  "end_call": <bool>,
  "hangup_reason": "<short English slug>",
  "call_phase": "<ptp|deceased|partial|payment_confirm|cannot_pay|already_paid|callback|opening|other>"
}

end_call = true ONLY after the COMPLETE flow (including mandatory closing where required) is delivered.
"""

# ── Per-flow detailed specs ───────────────────────────────────────────────────
_FLOW_SPEC = """
FLOW PTP (call_phase = "ptp") — Customer promises future payment.
  Step 1. Acknowledge willingness; ask exact date: "किस तारीख तक भुगतान कर पाएंगे?"
  Step 2. Capture date → ISO → clamp to HARD DATE WINDOW → store as target_date in context_patch.
          If outside window: say not accepted, ask for date on or before LAST_VALID_ISO.
  Step 3. Confirm date; deliver MANDATORY CLOSING (replace [TARGET_DATE] with human date).
          end_call = true, hangup_reason = "ptp_confirmed".

FLOW DECEASED (call_phase = "deceased") — Borrower or close family member died.
  Say sincere condolences (2 sentences, warm tone):
    "मुझे आपके नुकसान के बारे में सुनकर बेहद दुख हुआ। इस कठिन समय में हमारी गहरी संवेदनाएं आपके साथ हैं।"
  Then: "हमारी टीम का एक समर्पित सदस्य जल्द आपसे व्यक्तिगत रूप से संपर्क करेगा। आपने हमें सूचित किया, धन्यवाद।"
  end_call = true, hangup_reason = "deceased". No mandatory closing.

FLOW PARTIAL (call_phase = "partial") — Partial payment today + remaining date.
  Step 1. Ask how much they can pay today: "आज कितनी राशि का भुगतान कर सकते हैं? न्यूनतम {min_partial_int} रुपये।"
  Step 2. If amount < min_partial_int → reject ("यह न्यूनतम से कम है"), re-ask.
          If amount ≥ full EMI → treat as payment_confirm flow.
          Else → store partial_amount; state remaining balance (emi_amount_int - partial_amount).
  Step 3. Ask remainder date → clamp to HARD DATE WINDOW → store as target_date.
  Step 4. Confirm both; deliver MANDATORY CLOSING (use target_date as [TARGET_DATE]).
          end_call = true, hangup_reason = "partial_confirmed".

FLOW PAYMENT_CONFIRM (call_phase = "payment_confirm") — Customer pays full EMI today.
  Say: "आपकी पुष्टि के लिए धन्यवाद। SMS द्वारा भेजे गए भुगतान विकल्प से या नजदीकी शाखा में जाकर EMI भुगतान पूरा करें।"
  Set context_patch.target_date = CURRENT_DATE_ISO.
  Deliver MANDATORY CLOSING (use today's date as [TARGET_DATE]).
  end_call = true, hangup_reason = "payment_today_confirmed".

FLOW CANNOT_PAY (call_phase = "cannot_pay") — Only after partial was offered and declined.
  Step 1. Ask reason: "आप भुगतान क्यों नहीं कर पा रहे हैं?" Store in cannot_pay_reason.
          If user refuses reason twice → store cannot_pay_reason = "Unspecified".
  Step 2. Warn: "बकाया EMI से जुर्माना और CIBIL स्कोर पर असर पड़ सकता है।"
  Step 3. Ask callback: "हम आपसे दोबारा कब संपर्क करें?" → capture date/time → clamp → store callback_iso.
          If no date given → store callback_iso = "not_specified".
  Step 4. Deliver MANDATORY CLOSING (use callback_iso date as [TARGET_DATE]).
          end_call = true, hangup_reason = "cannot_pay_callback".

FLOW ALREADY_PAID (call_phase = "already_paid") — Customer claims prior payment.
  Step 1. Ask: "हमारे रिकॉर्ड सत्यापित करने के लिए, आपने EMI किस तारीख को चुकाई थी?"
  Step 2. If date is in future (> CURRENT_DATE_ISO) → reject ("यह भविष्य की तारीख है"), re-ask.
          Store as already_paid_date (ISO).
  Step 3. Ask mode: "किस माध्यम से? जैसे UPI, NEFT, शाखा में।" Store as payment_mode.
  Step 4. Say: "धन्यवाद। हमें आपकी भुगतान जानकारी मिल गई है। हम सत्यापित करके रिकॉर्ड अपडेट करेंगे। आपका दिन शुभ हो।"
          end_call = true, hangup_reason = "already_paid_noted". No mandatory closing.

FLOW CALLBACK (call_phase = "callback") — Customer is temporarily busy.
  Step 1. Acknowledge; ask: "कोई सुविधाजनक तारीख या समय बताइए जब मैं वापस कॉल करूँ।"
  Step 2. Capture time/date → clamp to HARD DATE WINDOW → store as callback_iso.
  Step 3. Brief reminder: "जुर्माने से बचने के लिए जल्द EMI चुकाने की कोशिश करें। आपका दिन शुभ हो।"
          end_call = true, hangup_reason = "callback_scheduled". No mandatory closing.

OPENING (call_phase = "opening") — First bot turn.
  Triggered when user message starts with "[EVENT:" or "[घटना:".
  Greet by customer_name; state overdue emi_amount and emi_due_date; ask when they can pay.
  end_call = false. No mandatory closing.
  ALWAYS set call_phase = "opening" for this system event trigger.
"""


def _hard_date_block(ctx: dict[str, str]) -> str:
    raw = (ctx.get("emi_overdue_date") or ctx.get("emi_due_date") or "").strip()
    anchor_d = _parse_ctx_date(raw) if raw else None
    if anchor_d is None:
        anchor_d = date.today()
    last_d = anchor_d + timedelta(days=90)
    return (
        "\n--- HARD DATE WINDOW (enforce strictly) ---\n"
        f"DUE_ANCHOR_ISO: {anchor_d.isoformat()}\n"
        f"LAST_VALID_ISO: {last_d.isoformat()}  (anchor + 90 days inclusive)\n"
        "Every payment promise, partial remainder date, cannot-pay callback, and busy callback MUST be "
        "on or between DUE_ANCHOR_ISO and LAST_VALID_ISO as YYYY-MM-DD. Reject and re-ask if outside.\n"
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


def _normalize(out: dict[str, Any]) -> dict[str, Any]:
    say = (out.get("say") or "").strip()
    patch = out.get("context_patch")
    if not isinstance(patch, dict):
        patch = {}
    patch_str = {str(k): str(v) for k, v in patch.items()}
    return {
        "say": say,
        "context_patch": patch_str,
        "end_call": bool(out.get("end_call")),
        "hangup_reason": str(out.get("hangup_reason") or "terminal"),
        "call_phase": str(out.get("call_phase") or "unknown"),
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
