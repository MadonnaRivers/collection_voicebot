"""
llm_orchestrator.py — Single LLM controls conversation logic and structured data.
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

_CORE_POLICY = """\
You are Aditi (female), Easy Home Finance. Output ONE JSON object only. No markdown.

CONCISE HINDI
- "say": Hindi (Devanagari) only. Feminine verb forms (हूँ, रही हूँ). 
- Max 2-3 short sentences. Brevity is key.
- Loanwords: EMI, SMS, UPI, CIBIL, link, branch, amount.

MANDATORY CLOSING
Append verbatim when ending ptp/partial/payment_confirm/cannot_pay:
"भुगतान पूरा करने के लिए आपको भेजे गए सुरक्षित लिंक का उपयोग करें। कृपया [TARGET_DATE] तक शेष राशि चुकाने की कोशिश करें ताकि आपका क्रेडिट स्कोर सुरक्षित रहे। आपके सहयोग के लिए धन्यवाद, और आपका दिन शुभ हो।"
Replace [TARGET_DATE] with date like "15 May 2026".

PARTIAL-FIRST
If cannot pay AND partial_offer_made ≠ "true" AND not already_paid/deceased:
- MUST offer partial (call_phase="partial"). Ask for at least {min_partial_int} today.
- Set context_patch.partial_offer_made = "true".

SILENCE
- [मौन] 1-2: ask again concisely.
- ≥ 3: end call.

PHASES: opening, payment_confirm, ptp, partial, cannot_pay, already_paid, deceased, callback, other.

SCHEMA: {"say": "...", "context_patch": {...}, "end_call": bool, "hangup_reason": "...", "call_phase": "..."}
"""

_FLOW_SPEC = """
PTP: Capture target_date (YYYY-MM-DD). Confirm + Closing. hangup_reason="ptp_confirmed".
DECEASED: Sincere condolences (2 sentences). No closing. hangup_reason="deceased".
PARTIAL: Ask amount (today) + date (remainder). Confirm + Closing. hangup_reason="partial_confirmed".
PAYMENT_CONFIRM: Thank for today's payment. Confirm + Closing. hangup_reason="payment_today_confirmed".
CANNOT_PAY: Ask reason + callback_iso. Warn CIBIL. Closing. hangup_reason="cannot_pay_callback".
ALREADY_PAID: Ask already_paid_date + payment_mode. No closing. hangup_reason="already_paid_noted".
CALLBACK: Ask callback_iso. No closing. hangup_reason="callback_scheduled".
OPENING: State overdue amount/date. Ask when paying. end_call=false.
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
