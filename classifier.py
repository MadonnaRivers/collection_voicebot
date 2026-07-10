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
from call_webhook import _INTERNAL_CTX_KEYS

log = logging.getLogger("aditi")


def _ctx_for_prompt(ctx: dict[str, str]) -> dict[str, str]:
    """Strip internal/transient bookkeeping keys before serializing ctx into
    the finalize_call_variables LLM prompt. Keeps the prompt focused on
    real call data — out_of_window_attempts, silence_count, _inserted_at
    etc. only add noise the LLM has to wade through."""
    return {k: v for k, v in ctx.items() if k not in _INTERNAL_CTX_KEYS}


def _backfill_from_ctx(result: dict, hangup_reason: str, ctx: dict[str, str]) -> None:
    """Fill ground-truth ctx values the transcript LLM missed.

    Unlike the old version this does NOT strip fields based on hangup_reason —
    that gating was the main source of null CRM data. Extraction is now driven
    by what the customer actually said (see the prompt's ground-truth rules);
    here we only *add* agent-validated ctx values when the LLM left a slot
    empty. Output keys are unchanged from before, so the webhook payload shape
    is byte-for-byte identical to the previous version.
    """
    # Prefer ctx (agent already validated these mid-call) when LLM left blank.
    if ctx.get("target_date") and not result.get("target_date"):
        result["target_date"] = ctx["target_date"]
    if ctx.get("partial_amount") and not result.get("partial_amount"):
        result["partial_amount"] = ctx["partial_amount"]
    if ctx.get("already_paid_date") and not result.get("already_paid_date"):
        result["already_paid_date"] = ctx["already_paid_date"]
    if ctx.get("payment_mode") and not result.get("already_paid_mode"):
        result["already_paid_mode"] = ctx["payment_mode"]
    if ctx.get("cannot_pay_reason") and not result.get("cannot_pay_reason"):
        result["cannot_pay_reason"] = ctx["cannot_pay_reason"]

    # callback intent has been removed from the flow — never emit callback_time.
    result.pop("callback_time", None)


async def finalize_call_variables(
    hangup_reason: str,
    ctx: dict[str, str],
    transcript_text: str = "",
) -> dict:
    """
    Derive CRM-ready fields purely from context + full transcript using LLM.
    Returns a dict with any subset of:
      summary, target_date, partial_amount, cannot_pay_reason,
      already_paid_date, already_paid_mode.
    """
    # Voicemail / machine-answered calls — no human conversation to classify.
    # Skip the LLM round-trip and return a deterministic summary.
    if hangup_reason in ("voicemail", "voicemail_left"):
        if hangup_reason == "voicemail_left":
            return {"summary": "Voicemail reached — pre-recorded reminder message left."}
        return {"summary": "Voicemail reached — call ended without leaving a message."}

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
            f"{json.dumps(_ctx_for_prompt(ctx), ensure_ascii=False, indent=2)}\n\n"
            f"Full dialogue transcript (Hindi/Hinglish):\n{transcript_text or '(no transcript)'}\n\n"
            "Read the FULL transcript and extract every CRM field the customer actually\n"
            "revealed — regardless of how the call ended. The system hangup_reason is only\n"
            "a hint; base every field on what the CUSTOMER said. Output ONE JSON object with\n"
            "exact key names below. Output ONLY valid JSON.\n"
            "\n"
            "GROUND-TRUTH RULES — read carefully:\n"
            "  • Base every value ONLY on what the CUSTOMER said (in ANY language — they may\n"
            "    speak Hindi, Hinglish, or a regional language; understand all of them).\n"
            "  • NEVER copy values the BOT said (offers, prompts, defaults, examples).\n"
            "  • Never invent dates, amounts, modes, or reasons. If a field wasn't stated,\n"
            "    OMIT it (except payment_intent and summary, which are always required).\n"
            "  • Prefer context values when they exist (the agent already validated them).\n"
            "\n"
            "FIELDS (exact key names — same set as always; do not add new keys):\n"
            "- summary (REQUIRED): one-line English outcome describing what happened.\n"
            "    If the customer asked us to auto-debit / deduct the EMI from their bank\n"
            "    account or mandate, say so plainly here (this is how auto-debit is recorded).\n"
            "- target_date: YYYY-MM-DD — any future date the customer said they'd pay by\n"
            "    (full or remainder). Compute relative dates ('kal','parso','next week') off\n"
            "    today. Include whenever the customer named/implied a pay date.\n"
            "- partial_amount: rupee number as string — the amount the CUSTOMER said they can\n"
            "    pay now (the bot never offers amounts, so it must be customer-stated).\n"
            "- cannot_pay_reason: 5-15 word English summary of why the customer cannot pay,\n"
            "    whenever they gave any reason. Use 'uncooperative' if evasive/no real reason.\n"
            "- already_paid_date: YYYY-MM-DD — date the customer claims they already paid.\n"
            "- already_paid_mode: UPI / NEFT / IMPS / RTGS / cash / cheque / card / netbanking\n"
            "    — whenever the customer named how they paid.\n"
            "\n"
            "Dates must be ISO YYYY-MM-DD. Omit fields the customer never stated (except\n"
            "summary, which is required). Output ONLY the JSON object."
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

    # LLM path failed — still return a non-null intent + backfilled ctx values.
    fallback = {
        "summary": f"Call ended ({hangup_reason}).",
    }
    _backfill_from_ctx(fallback, hangup_reason, ctx)
    return fallback
