"""
call_webhook.py — Push normalized call-summary payloads to n8n / CRM webhooks.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from clients import http as http_client

log = logging.getLogger("aditi")

# Human-readable state labels sent to n8n.
# Internal call_phase values are kept as-is everywhere else (transcripts, logs).
_STATE_LABEL: dict[str, str] = {
    "ptp":             "PTP",
    "payment_confirm": "Agrees to Pay",
    "partial":         "Partial Payment",
    "cannot_pay":      "Cannot Pay Reason",
    "already_paid":    "Already Paid",
    "deceased":        "Deceased Report",
    "no_response":     "No Response",
}


def humanize_state(state: str) -> str:
    """Return the human-readable label for a call_phase value."""
    return _STATE_LABEL.get(state.lower().strip(), state)


# Canonical classifier-output fields. Listed here for documentation only —
# build_call_summary_push_body() flattens ALL classifier output into the body.
CALL_SUMMARY_OUTPUT_KEYS: tuple[str, ...] = (
    "state",             # final call phase
    "summary",           # one-line English outcome (always present)
    "target_date",       # YYYY-MM-DD — universal follow-up date
    "partial_amount",    # rupees (string) — partial committed today
    "cannot_pay_reason", # short English — why they cannot pay
    "doing_payment",     # true/false — customer confirmed paying full EMI today
    "already_paid_date", # YYYY-MM-DD — date they claim to have already paid
    "already_paid_mode", # UPI / NEFT / cash / branch / etc.
)


# Input-context keys that should always appear in the webhook body, even if
# the underlying call did not touch them. Anything else present in ctx will
# also be passed through verbatim (e.g. custom CRM fields).
CALL_SUMMARY_INPUT_KEYS: tuple[str, ...] = (
    "customer_name",
    "phone_number",
    "loan_id",
    "emi_overdue_amt",
    "emi_overdue_date",
    "emi_amount",         # alias of emi_overdue_amt
    "emi_due_date",       # alias of emi_overdue_date
    "emi_amount_int",     # derived integer form (no commas)
    "min_partial",
    "min_partial_int",
    "payment_deadline",
)


# Internal/transient bookkeeping keys that should NOT leak into the webhook
# payload. Anything in this set is stripped on the way out.
_INTERNAL_CTX_KEYS: frozenset[str] = frozenset({
    "out_of_window_attempts",
    "silence_count",
    "error_count",
    "retry_count",
    "turn_count",
    "partial_offer_made",   # legacy V1 flag, no longer used
})


def build_call_summary_push_body(
    call_sid: str,
    hangup_reason: str,
    call_vars: dict[str, Any] | None,
    ctx: dict[str, str] | None = None,
    state: str = "",
) -> dict[str, Any]:
    """
    Build the JSON payload pushed to the n8n /push_data webhook.

    Contains EVERY parameter known about the call:
      • runtime fields  (call_sid, hangup_reason, state, ended_at, doing_payment)
      • input ctx       (customer_name, phone_number, loan_id, EMI details, …
                         and any custom CRM key the caller passed to /make-call)
      • classifier out  (summary, target_date, partial_amount, cannot_pay_reason,
                         already_paid_date, already_paid_mode, … plus any extras
                         the classifier LLM added)

    Resolution rules when the same key appears in both ctx and call_vars:
      classifier value WINS  (it knows the actual outcome of the call).
      Empty / missing classifier values fall back to ctx.

    Internal bookkeeping keys (out_of_window_attempts, silence_count, etc.)
    are stripped — they never reach n8n.
    """
    cv = dict(call_vars or {})
    cx = dict(ctx or {})

    # ── Runtime fields ──────────────────────────────────────────────────────
    body: dict[str, Any] = {
        "call_sid":      call_sid,
        "hangup_reason": hangup_reason,
        "ended_at":      (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        ),
        "state":         humanize_state(state or cv.pop("state", "") or ""),
        # Deterministic — doesn't need the LLM
        "doing_payment": hangup_reason == "payment_today_confirmed",
    }

    # ── 1. Input ctx (everything the caller passed in for this call) ────────
    # First the well-known keys (so they're always present, even if blank),
    # then any other custom fields from ctx.
    for k in CALL_SUMMARY_INPUT_KEYS:
        if k in _INTERNAL_CTX_KEYS:
            continue
        body[k] = cx.get(k, "")
    for k, v in cx.items():
        if k in _INTERNAL_CTX_KEYS:
            continue
        if k not in body:
            body[k] = v

    # ── 2. Classifier output (overrides ctx where it provides a real value) ─
    for k, v in cv.items():
        if k in _INTERNAL_CTX_KEYS:
            continue
        if v in (None, "", [], {}):
            # Don't overwrite a populated ctx value with an empty classifier value
            body.setdefault(k, v)
        else:
            body[k] = v

    # ── 3. Guarantee the canonical output keys exist (null if not produced) ─
    for k in CALL_SUMMARY_OUTPUT_KEYS:
        body.setdefault(k, None)

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
