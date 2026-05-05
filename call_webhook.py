"""
call_webhook.py — Push normalized call-summary payloads to n8n / CRM webhooks.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from clients import http as http_client

log = logging.getLogger("aditi")

# Canonical output fields pushed to n8n after every call.
# classifier.finalize_call_variables() must use these exact key names.
CALL_SUMMARY_OUTPUT_KEYS: tuple[str, ...] = (
    "state",             # final call phase: ptp | partial | cannot_pay | payment_confirm |
                         #   already_paid | callback | deceased | no_response | error
    "summary",           # one-line English outcome (always present)
    "target_date",       # YYYY-MM-DD  — universal follow-up date:
                         #   ptp → promised payment date
                         #   partial → remainder balance due date
                         #   cannot_pay → callback date
                         #   deceased → team follow-up date
    "partial_amount",    # rupees (string) — partial payment committed today
    "cannot_pay_reason", # short English — why they cannot pay
    "doing_payment",     # true/false — customer confirmed paying full EMI today
    "already_paid_date", # YYYY-MM-DD  — date they claim to have already paid
    "already_paid_mode", # UPI / NEFT / cash / branch etc.
    "callback_time",     # ISO or human phrase — when to call back (busy/callback flow)
)


def build_call_summary_push_body(
    call_sid: str,
    hangup_reason: str,
    call_vars: dict[str, Any] | None,
    ctx: dict[str, str] | None = None,
    state: str = "",
) -> dict[str, Any]:
    cv = dict(call_vars or {})
    cx = dict(ctx or {})
    body: dict[str, Any] = {
        "call_sid":      call_sid,
        "hangup_reason": hangup_reason,
        "ended_at":      (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        ),
        # state — prefer explicit arg, fall back to what classifier may have set
        "state": state or cv.pop("state", ""),
        # doing_payment is deterministic from hangup_reason — no LLM needed
        "doing_payment": hangup_reason == "payment_today_confirmed",
    }
    for k in CALL_SUMMARY_OUTPUT_KEYS:
        if k not in body:
            body[k] = cv.get(k)

    # Always include base customer/loan fields expected by downstream n8n flow.
    # Prefer classifier output, then fall back to per-call ctx.
    body["phone_number"] = cv.get("phone_number") or cx.get("phone_number", "")
    body["customer_name"] = cv.get("customer_name") or cx.get("customer_name", "")
    body["loan_id"] = cv.get("loan_id") or cx.get("loan_id", "")
    body["emi_overdue_amt"] = (
        cv.get("emi_overdue_amt")
        or cx.get("emi_overdue_amt")
        or cx.get("emi_amount", "")
    )
    body["emi_overdue_date"] = (
        cv.get("emi_overdue_date")
        or cx.get("emi_overdue_date")
        or cx.get("emi_due_date", "")
    )
    body["min_partial"] = cv.get("min_partial") or cx.get("min_partial", "")
    body["payment_deadline"] = cv.get("payment_deadline") or cx.get("payment_deadline", "")

    # Pass through any extra fields the LLM classifier added
    for k, v in cv.items():
        if k not in body:
            body[k] = v
    return body


async def push_call_summary_webhook(url: str, body: dict[str, Any]) -> None:
    if not url:
        return
    try:
        r = await http_client.post(url, json=body)
        r.raise_for_status()
        log.info("Call summary webhook OK (%s)", r.status_code)
    except Exception as exc:
        log.warning("Call summary webhook failed: %s", exc)
