"""
test_recent_fixes.py — Focused tests for the recent prompt+plumbing fixes.

Covers:
  R01  parse_date numeric formats (DD/MM/YYYY, ISO, etc.)
  R02  _hard_date_block uses the real EMI due date as the 90-day anchor
  R03  _maybe_enforce_90day_window — 1st rejection (no cap mentioned)
  R04  _maybe_enforce_90day_window — 2nd rejection (cap revealed)
  R05  _maybe_enforce_90day_window — valid date passes through untouched
  R06  payment_today_confirmed safety net still demotes future-date utterances
  R07  PTP concrete date ("कल") → straight close, no confirmation question (live LLM)
  R08  PTP out-of-window ("2 महीने बाद" with old due date) → 2-step rejection (live LLM)
  R09  cannot_pay flow — bot never offers partial (live LLM)
  R10  partial flow — only triggers when customer proposes amount (live LLM)

Run:
  set PYTHONIOENCODING=utf-8
  venv/Scripts/python.exe -X utf8 test_recent_fixes.py
"""
from __future__ import annotations

import asyncio
import io
import logging
import sys
from datetime import date, timedelta

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s - %(message)s")

from utils import parse_date
from llm_orchestrator import (
    _ctx_anchor_date,
    _hard_date_block,
    _maybe_enforce_90day_window,
    _maybe_fix_payment_confirm_misclassification,
    run_conversation_turn,
)
from scripts import build_default_ctx

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

_results: list[tuple[str, str, str]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"    {icon}  {label}" + (f"   [{detail}]" if detail else ""))
    _results.append((icon, label, detail))
    return ok


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE tests (no network)
# ─────────────────────────────────────────────────────────────────────────────
def t_parse_date() -> None:
    banner("R01  parse_date — numeric formats")
    cases = {
        "06/04/2026": date(2026, 4, 6),
        "6-4-2026":   date(2026, 4, 6),
        "2026-04-06": date(2026, 4, 6),
        "06/04/26":   date(2026, 4, 6),
        "31/12/2025": date(2025, 12, 31),
    }
    for inp, expected in cases.items():
        got = parse_date(inp)
        check(f"parse_date({inp!r}) == {expected}", got == expected, f"got={got}")


def t_anchor_block() -> None:
    banner("R02  _hard_date_block — anchor uses real due date")
    ctx = {"emi_due_date": "06/04/2026"}
    anchor = _ctx_anchor_date(ctx)
    check("anchor == 2026-04-06", anchor == date(2026, 4, 6), f"got={anchor}")
    expected_last = anchor + timedelta(days=90)
    block = _hard_date_block(ctx)
    check("LAST_VALID_ISO 2026-07-05 present in date block",
          "2026-07-05" in block,
          f"last_valid={expected_last.isoformat()}")
    check("anchor 2026-04-06 present in date block",
          "2026-04-06" in block)
    check("date block mentions out_of_window_attempts counter",
          "out_of_window_attempts" in block)
    check("date block contains FIRST-rejection wording (generic, no cap)",
          "कोई और date बताइए" in block)


def t_window_safety_net() -> None:
    banner("R03–R05  _maybe_enforce_90day_window — 2-step rejection")
    ctx = {"customer_name": "Kartik", "emi_due_date": "06/04/2026"}

    # R03 — first strike: generic rejection, no cap date mentioned
    r1 = _maybe_enforce_90day_window(
        {
            "say": "(LLM tried to accept)",
            "context_patch": {"target_date": "2026-07-30"},
            "end_call": True,
            "hangup_reason": "ptp_confirmed",
            "call_phase": "ptp",
        },
        ctx,
    )
    check("R03  1st strike — generic rejection wording",
          "यह date valid नहीं है" in r1["say"]
          and "कोई और date" in r1["say"])
    check("R03  1st strike — cap date NOT revealed",
          "05 Jul 2026" not in r1["say"] and "2026-07-05" not in r1["say"])
    check("R03  1st strike — bad target_date stripped",
          "target_date" not in r1["context_patch"])
    check("R03  1st strike — counter incremented to 1",
          r1["context_patch"].get("out_of_window_attempts") == "1")
    check("R03  1st strike — end_call=False",
          r1["end_call"] is False)
    check("R03  1st strike — phase preserved",
          r1["call_phase"] == "ptp")

    # R04 — second strike (ctx now carries the counter): cap revealed
    ctx2 = dict(ctx, **r1["context_patch"])
    r2 = _maybe_enforce_90day_window(
        {
            "say": "(LLM tried again)",
            "context_patch": {"target_date": "2026-08-15"},
            "end_call": True,
            "hangup_reason": "ptp_confirmed",
            "call_phase": "ptp",
        },
        ctx2,
    )
    check("R04  2nd strike — cap date IS revealed",
          "05 Jul 2026" in r2["say"])
    check("R04  2nd strike — counter incremented to 2",
          r2["context_patch"].get("out_of_window_attempts") == "2")
    check("R04  2nd strike — still end_call=False",
          r2["end_call"] is False)

    # R05 — valid date passes through unchanged
    ok_in = {
        "say": "Thank you Kartik जी। 20 Jun 2026 तक payment कर दीजिए...",
        "context_patch": {"target_date": "2026-06-20"},
        "end_call": True,
        "hangup_reason": "ptp_confirmed",
        "call_phase": "ptp",
    }
    ok_out = _maybe_enforce_90day_window(ok_in, ctx)
    check("R05  valid date — passes through unchanged",
          ok_out == ok_in)


def t_payment_confirm_safety_net() -> None:
    banner("R06  payment_today_confirmed safety net (existing fixer)")
    # Customer said "कल" but LLM mistakenly fired payment_today_confirmed
    bad = {
        "say": "Thank you Kartik जी। आज तक payment कर दीजिए...",
        "context_patch": {},
        "end_call": True,
        "hangup_reason": "payment_today_confirmed",
        "call_phase": "payment_confirm",
    }
    out = _maybe_fix_payment_confirm_misclassification(bad, "कल कर दूँगा।")
    check("R06  demoted to PTP",
          out["hangup_reason"] == "ptp_confirmed"
          and out["call_phase"] == "ptp")
    check("R06  target_date stored",
          "target_date" in out.get("context_patch", {}))


# ─────────────────────────────────────────────────────────────────────────────
# LIVE LLM tests (require OPENAI_API_KEY in .env)
# ─────────────────────────────────────────────────────────────────────────────
async def t_concrete_ptp_no_confirmation() -> None:
    banner("R07  LIVE — PTP with concrete date ('कल') → no confirmation")
    ctx = build_default_ctx()
    ctx["customer_name"] = "Kartik"
    ctx["emi_overdue_date"] = (date.today() - timedelta(days=5)).strftime("%d/%m/%Y")
    ctx["emi_due_date"]     = ctx["emi_overdue_date"]
    history: list[dict[str, str]] = [
        {"role": "assistant", "content":
            f"नमस्ते {ctx['customer_name']} जी, मैं अदिति बोल रही हूँ Easy Home Finance से। "
            f"आपकी home loan EMI {ctx['emi_overdue_amt']} रुपये pending है। "
            "बताइए, कब तक payment कर पाएंगे?"
        },
    ]
    user_msg = "कल कर दूँगा।"
    r = await run_conversation_turn(ctx, history, user_msg)
    print(f"      LLM say   : {r['say']}")
    print(f"      phase     : {r['call_phase']}    end_call: {r['end_call']}")
    print(f"      patch     : {r['context_patch']}")
    check("R07  call_phase == ptp", r["call_phase"] == "ptp")
    check("R07  end_call == True (closed without confirmation step)",
          r["end_call"] is True)
    check("R07  hangup_reason == ptp_confirmed",
          r["hangup_reason"] == "ptp_confirmed")
    check("R07  target_date stored as YYYY-MM-DD",
          "target_date" in r["context_patch"]
          and len(r["context_patch"].get("target_date", "")) == 10)
    check("R07  did NOT ask for confirmation",
          "क्या आप" not in r["say"] and "तक payment कर देंगे?" not in r["say"])


async def t_out_of_window_two_step() -> None:
    banner("R08  LIVE — out-of-window ('2 महीने बाद' with old due date)")
    ctx = build_default_ctx()
    ctx["customer_name"] = "Kartik"
    # Due date in past such that today + 60 days exceeds anchor + 90 days
    ctx["emi_overdue_date"] = "06/04/2026"
    ctx["emi_due_date"]     = "06/04/2026"

    # Note: today's real date matters for whether 2 महीने बाद is past LAST_VALID.
    # As of the test environment date (2026-05-23), today+60 ≈ 22 Jul 2026,
    # while LAST_VALID = 06 Apr + 90 = 05 Jul 2026 → out of window.
    history = [{"role": "assistant", "content":
        "नमस्ते Kartik जी, मैं अदिति बोल रही हूँ Easy Home Finance से। "
        "आपकी home loan EMI 8,500 रुपये pending है, जिसकी due date 06/04/2026 थी। "
        "बताइए, आप कब तक payment कर पाएंगे?"
    }]

    r1 = await run_conversation_turn(ctx, history, "मैं 2 महीने बाद कर दूंगा।")
    print(f"      [TURN 1] say   : {r1['say']}")
    print(f"      [TURN 1] patch : {r1['context_patch']}    end: {r1['end_call']}")

    # The safety net should kick in either because LLM accepted past-window,
    # OR the LLM itself rejected per the prompt. Either way we expect:
    # - no target_date stored
    # - end_call == False
    # - cap date NOT shown on first strike
    check("R08  TURN 1 — end_call=False (rejected)", r1["end_call"] is False)
    check("R08  TURN 1 — target_date NOT stored",
          not r1["context_patch"].get("target_date"))
    check("R08  TURN 1 — cap date '05 Jul 2026' NOT mentioned",
          "05 Jul 2026" not in r1["say"]
          and "05 जुलाई" not in r1["say"]
          and "2026-07-05" not in r1["say"])

    # Apply turn-1 context patch and feed turn-2
    ctx_after = dict(ctx)
    ctx_after.update(r1.get("context_patch", {}) or {})
    history2 = history + [
        {"role": "user",      "content": "मैं 2 महीने बाद कर दूंगा।"},
        {"role": "assistant", "content": r1["say"]},
    ]
    r2 = await run_conversation_turn(ctx_after, history2, "ठीक है, 2 महीने बाद ही।")
    print(f"      [TURN 2] say   : {r2['say']}")
    print(f"      [TURN 2] patch : {r2['context_patch']}    end: {r2['end_call']}")
    check("R08  TURN 2 — end_call=False (still rejected)", r2["end_call"] is False)
    check("R08  TURN 2 — target_date still NOT stored",
          not r2["context_patch"].get("target_date"))
    check("R08  TURN 2 — cap date '05 Jul 2026' IS mentioned",
          "05 Jul 2026" in r2["say"] or "05 जुलाई 2026" in r2["say"])


async def t_cannot_pay_no_partial_offer() -> None:
    banner("R09  LIVE — cannot_pay never mentions partial")
    ctx = build_default_ctx()
    ctx["customer_name"] = "Kartik"
    history = [{"role": "assistant", "content":
        "नमस्ते Kartik जी, मैं अदिति बोल रही हूँ Easy Home Finance से। "
        "आपकी home loan EMI 8,500 रुपये pending है। बताइए, कब तक payment कर पाएंगे?"
    }]
    r = await run_conversation_turn(ctx, history, "मेरे पास पैसे नहीं हैं, pay नहीं कर सकता।")
    print(f"      say   : {r['say']}")
    print(f"      phase : {r['call_phase']}    end: {r['end_call']}")
    check("R09  phase == cannot_pay", r["call_phase"] == "cannot_pay")
    check("R09  end_call == False (asks reason first)", r["end_call"] is False)
    say_lower = r["say"].lower()
    check("R09  bot did NOT offer partial",
          "partial" not in say_lower and "₹1500" not in r["say"]
          and "1500" not in r["say"]
          and "कुछ amount" not in r["say"])


async def t_partial_only_when_customer_proposes() -> None:
    banner("R10  LIVE — partial triggers only when customer proposes amount")
    ctx = build_default_ctx()
    ctx["customer_name"] = "Kartik"
    history = [{"role": "assistant", "content":
        "नमस्ते Kartik जी, मैं अदिति बोल रही हूँ Easy Home Finance से। "
        "आपकी home loan EMI 8,500 रुपये pending है। बताइए, कब तक payment कर पाएंगे?"
    }]
    r = await run_conversation_turn(ctx, history, "अभी मैं 3000 रुपये दे सकता हूँ।")
    print(f"      say   : {r['say']}")
    print(f"      phase : {r['call_phase']}    end: {r['end_call']}")
    print(f"      patch : {r['context_patch']}")
    check("R10  phase == partial", r["call_phase"] == "partial")
    check("R10  partial_amount stored == '3000'",
          r["context_patch"].get("partial_amount") == "3000")
    check("R10  bot asks remainder date",
          "बाकी" in r["say"] or "remaining" in r["say"].lower()
          or "remainder" in r["say"].lower() or "कब तक" in r["say"])


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
async def main() -> int:
    # offline first
    t_parse_date()
    t_anchor_block()
    t_window_safety_net()
    t_payment_confirm_safety_net()

    # live LLM
    try:
        await t_concrete_ptp_no_confirmation()
        await t_out_of_window_two_step()
        await t_cannot_pay_no_partial_offer()
        await t_partial_only_when_customer_proposes()
    except Exception as exc:
        print(f"\n{RED}LLM tests aborted: {exc}{RESET}")

    banner("SUMMARY")
    passed = sum(1 for r in _results if "PASS" in r[0])
    failed = sum(1 for r in _results if "FAIL" in r[0])
    total  = len(_results)
    print(f"  {passed}/{total} checks passed   ({failed} failed)")
    if failed:
        print("\n  Failures:")
        for icon, label, detail in _results:
            if "FAIL" in icon:
                print(f"    {RED}{label}{RESET}   [{detail}]")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
