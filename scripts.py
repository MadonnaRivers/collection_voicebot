"""
scripts.py — Scripted bot lines, FSM state constants, customer defaults,
             and context builder.
"""
from __future__ import annotations
import os
from datetime import date as _date, timedelta
from utils import fmt_date

# ─────────────────────────────────────────────────────────────────────────────
# Default customer data  (.env fallbacks for testing; override per-call via API)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CUSTOMER: dict[str, str] = {
    # ── Core caller inputs (pass these from your CRM / n8n) ──────────────────
    "customer_name":    os.getenv("DEFAULT_CUSTOMER_NAME",    "Rahul"),
    "phone_number":     os.getenv("DEFAULT_PHONE_NUMBER",     ""),
    "loan_id":          os.getenv("DEFAULT_LOAN_ID",          "EH12345"),
    "emi_overdue_amt":  os.getenv("DEFAULT_EMI_OVERDUE_AMT",  "8,500"),   # formatted e.g. "8,500"
    "emi_overdue_date": os.getenv("DEFAULT_EMI_OVERDUE_DATE", ""),
    "min_partial":      os.getenv("DEFAULT_MIN_PARTIAL",      "1,500"),   # formatted e.g. "1,500"
    "payment_deadline": os.getenv("DEFAULT_PAYMENT_DEADLINE", ""),
}


def _to_int_str(formatted: str) -> str:
    """'8,500' → '8500' — strip commas so LLM can do arithmetic."""
    return formatted.replace(",", "").strip()


def build_default_ctx() -> dict[str, str]:
    """Return a fresh per-call context dict with all defaults filled in."""
    ctx = dict(DEFAULT_CUSTOMER)
    if not ctx.get("payment_deadline"):
        ctx["payment_deadline"] = fmt_date(_date.today() + timedelta(days=7))

    # Derive integer variants internally — callers never need to pass these
    if ctx.get("emi_overdue_amt"):
        ctx["emi_amount_int"] = _to_int_str(ctx["emi_overdue_amt"])
    if ctx.get("min_partial"):
        ctx["min_partial_int"] = _to_int_str(ctx["min_partial"])

    # Backward-compat aliases so both old and new field names work in LLM prompts
    ctx.setdefault("emi_amount",    ctx.get("emi_overdue_amt", ""))
    ctx.setdefault("emi_due_date",  ctx.get("emi_overdue_date", ""))
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Instant opening greeting — no LLM needed, fires in milliseconds
# ─────────────────────────────────────────────────────────────────────────────
def build_opening_greeting(ctx: dict[str, str]) -> str:
    """
    Build the opening greeting directly from context — zero LLM latency.

    Kept short (~14 words / ~5s of audio) so customers hear the bot speak
    within ~1.5s of pickup. The longer "मैं अदिति बोल रही हूँ Easy Home
    Finance से" intro is removed from the opening — bot's brand identity
    is established by the company name + EMI amount, and customers who
    ask "कौन सी company?" get the full company name via the FAQ handler.
    """
    name   = ctx.get("customer_name") or ctx.get("name") or ""
    amount = ctx.get("emi_amount") or ctx.get("emi_overdue_amt") or ""

    greeting = f"नमस्ते {name} जी, " if name else "नमस्ते जी, "

    if amount:
        return (
            f"{greeting}Easy Home Finance से अदिति। "
            f"आपकी EMI {amount} रुपये pending है — कब तक pay कर पाएंगे?"
        )
    return (
        f"{greeting}Easy Home Finance से अदिति। "
        "आपकी EMI pending है — कब तक pay कर पाएंगे?"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Voicemail message — spoken when Plivo AMD detects an answering machine.
# One-way: no questions, no expectation of reply. Includes name, amount,
# CIBIL hook and brand so the customer gets useful context on playback.
# ─────────────────────────────────────────────────────────────────────────────
def build_voicemail_message(ctx: dict[str, str]) -> str:
    name   = ctx.get("customer_name") or ctx.get("name") or ""
    amount = ctx.get("emi_amount") or ctx.get("emi_overdue_amt") or ""

    greeting = f"नमस्ते {name} जी, " if name else "नमस्ते जी, "
    if amount:
        return (
            f"{greeting}Easy Home Finance से अदिति बोल रही हूँ। "
            f"आपकी {amount} रुपये की EMI pending है। "
            "कृपया जल्द से जल्द payment कर दीजिए ताकि penalty charges से बचें "
            "और आपका CIBIL score safe रहे। धन्यवाद।"
        )
    return (
        f"{greeting}Easy Home Finance से अदिति बोल रही हूँ। "
        "आपकी EMI pending है। कृपया जल्द से जल्द payment कर दीजिए "
        "ताकि penalty charges से बचें और आपका CIBIL score safe रहे। धन्यवाद।"
    )


# Legacy FSM script dict, TERMINAL / BARGE_IN_LOCKED / AUTO_ADVANCE sets,
# and the _MANDATORY_CLOSING template lived here when the flow was driven
# by a hand-rolled state machine. They were removed when call_handler
# switched to llm_orchestrator.stream_conversation_turn — closings now
# come from the LLM prompt (_FLOW_SPEC in llm_orchestrator.py) and are
# locked deterministically by the _enforce_* safety nets there.
# Nothing in the live code path imports them; do not re-introduce.
