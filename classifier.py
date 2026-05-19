"""
classifier.py — Post-call storage normalisation (LLM only).

Conversation flow and per-turn data live in llm_orchestrator.py.
"""
from __future__ import annotations
import json
import logging
from datetime import date as _date

from clients import oai_llm
from config import LLM_MODEL

log = logging.getLogger("aditi")


def _backfill_from_ctx(result: dict, hangup_reason: str, ctx: dict[str, str]) -> None:
    """Ensure key output fields are filled from ctx when the LLM missed them."""
    # target_date — universal follow-up date across all flows
    # Priority: explicit target_date > payment_commitment_iso > callback_iso
    for key in ("target_date", "payment_commitment_iso", "callback_iso"):
        if ctx.get(key) and not result.get("target_date"):
            result["target_date"] = ctx[key]
            break
    # partial_amount
    if ctx.get("partial_amount") and not result.get("partial_amount"):
        result["partial_amount"] = ctx["partial_amount"]
    # already_paid fields
    if ctx.get("already_paid_date") and not result.get("already_paid_date"):
        result["already_paid_date"] = ctx["already_paid_date"]
    if ctx.get("payment_mode") and not result.get("already_paid_mode"):
        result["already_paid_mode"] = ctx["payment_mode"]
    # callback_time from callback_iso (busy/callback flow only)
    if ctx.get("callback_iso") and not result.get("callback_time"):
        result["callback_time"] = ctx["callback_iso"]
    # cannot_pay_reason
    if ctx.get("cannot_pay_reason") and not result.get("cannot_pay_reason"):
        result["cannot_pay_reason"] = ctx["cannot_pay_reason"]


async def finalize_call_variables(
    hangup_reason: str,
    ctx: dict[str, str],
    transcript_text: str = "",
) -> dict:
    """
    Derive CRM-ready fields purely from context + full transcript using LLM.
    Returns a dict with any subset of: summary, partial_amount, remaining_balance,
    partial_remainder_due_date, partial_offer_made, pay_later_date, cannot_pay_reason,
    target_date, call_back_time, already_paid_date, payment_mode, structured_notes, etc.
    """
    today_str = _date.today().isoformat()
    customer = ctx.get("customer_name", "customer")
    phone = ctx.get("phone_number", "")
    loan_id = ctx.get("loan_id", "")
    emi = ctx.get("emi_overdue_amt") or ctx.get("emi_amount", "")
    emi_date = ctx.get("emi_overdue_date") or ctx.get("emi_due_date", "")

    try:
        prompt = (
            f"An EMI collection voice call just ended. Today: {today_str}\n"
            f"Customer: {customer}, Phone: {phone}, Loan ID: {loan_id}\n"
            f"Overdue EMI: ₹{emi} (original due {emi_date})\n"
            f"Hangup reason (system): {hangup_reason}\n\n"
            f"Final merged context JSON (key facts the agent stored):\n"
            f"{json.dumps(dict(ctx), ensure_ascii=False, indent=2)}\n\n"
            f"Full dialogue transcript (Hindi/Hinglish):\n{transcript_text or '(no transcript)'}\n\n"
            "Output ONE JSON object with ONLY the applicable fields below. Use exact key names.\n"
            "- summary: one-line English outcome (REQUIRED)\n"
            "- target_date: YYYY-MM-DD — the single universal follow-up date:\n"
            "    ptp → date customer promised to pay in full (context: target_date)\n"
            "    partial → date remainder balance is due (context: target_date)\n"
            "    cannot_pay → callback/follow-up date (context: callback_iso)\n"
            "    deceased → team follow-up date (context: callback_iso or target_date)\n"
            "    Use context target_date or callback_iso if set.\n"
            "- partial_amount: rupees string (e.g. \"3000\") — ONLY for partial payment flow. OMIT for payment_confirm/ptp/cannot_pay/already_paid/callback.\n"
            "- cannot_pay_reason: 5-15 word English — why they cannot pay\n"
            "- already_paid_date: YYYY-MM-DD — date they claim to have already paid\n"
            "- already_paid_mode: UPI / NEFT / cash / branch / etc.\n"
            "- callback_time: ISO date or short phrase — when to call back (busy/callback flow; "
            "use context callback_iso if set)\n"
            "Use ISO dates. Omit unknown fields. Output ONLY valid JSON."
        )
        resp = await oai_llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        raw_out = (resp.choices[0].message.content or "").strip()
        result = json.loads(raw_out)
        if isinstance(result, dict):
            _backfill_from_ctx(result, hangup_reason, ctx)
            log.info("CALL_VARS %s", result)
            return result
    except Exception as exc:
        log.warning("finalize_call_variables error: %s", exc)

    return {
        "summary": f"Call ended ({hangup_reason}). Context keys: {', '.join(ctx.keys())}.",
    }
