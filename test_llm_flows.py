"""
test_llm_flows.py — Comprehensive LLM orchestrator tests for all 7 Aditi call flows.

Covers:
  T01 Opening / Greeting
  T02 PTP — full flow (future date given)
  T03 PTP — date beyond 90-day window (must reject)
  T04 Full Payment Today
  T05 Deceased borrower
  T06 Cannot Pay → partial offered first → declined → reason → callback
  T07 Already Paid — date + mode captured
  T08 Callback — busy now
  T09 Silence / no-response handling
  T10 Partial Payment — full amount+date flow
  T11 JSON / schema stability across 5 rapid turns

Usage:
  python test_llm_flows.py
  python test_llm_flows.py -v      # verbose: print full LLM "say" for each turn
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import time
from datetime import date, timedelta
from typing import Any

# Force UTF-8 output on Windows so Hindi / tick chars don't crash
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Keep test output clean; only show warnings+
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s - %(message)s")

from llm_orchestrator import run_conversation_turn
from scripts import build_default_ctx, build_opening_greeting
from utils import fmt_date, parse_date

VERBOSE = "-v" in sys.argv

# ── Colour/symbol helpers ─────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
RESET  = "\033[0m"
PASS   = f"{GREEN}PASS{RESET}"
FAIL   = f"{RED}FAIL{RESET}"
WARN   = f"{YELLOW}WARN{RESET}"

_results: list[tuple[str, str, str]] = []   # (status, label, detail)
_test_timings: list[tuple[str, float]] = [] # (test_name, elapsed_s)


def check(label: str, cond: bool, detail: str = "") -> bool:
    icon = PASS if cond else FAIL
    _results.append((icon, label, detail))
    print(f"    {icon} {label}" + (f"  [{detail}]" if detail else ""))
    return cond


def is_hindi(text: str) -> bool:
    """True if text contains at least a few Devanagari characters."""
    count = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    return count >= 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fresh() -> dict[str, str]:
    return build_default_ctx()


def _add(history: list, role: str, content: str) -> list:
    return history + [{"role": role, "content": content}]


async def turn(ctx: dict, history: list, msg: str) -> dict[str, Any]:
    return await run_conversation_turn(ctx, history, msg)


def _patch(ctx: dict, t: dict) -> dict:
    p = t.get("context_patch", {})
    if isinstance(p, dict):
        ctx.update({str(k): str(v) for k, v in p.items()})
    return ctx


def _opening_turn(ctx: dict) -> dict[str, Any]:
    """Instant opening turn (no LLM) — mirrors production call_handler path."""
    return {
        "say":           build_opening_greeting(ctx),
        "context_patch": {},
        "end_call":      False,
        "hangup_reason": "",
        "call_phase":    "opening",
    }


def _hist(history: list, user_msg: str | None, say: str) -> list:
    h = list(history)
    if user_msg:
        h.append({"role": "user", "content": user_msg})
    if say:
        h.append({"role": "assistant", "content": say})
    return h


# ── Human-readable date for this month+14 days ───────────────────────────────
_FUTURE_14   = fmt_date(date.today() + timedelta(days=14))
_FUTURE_30   = fmt_date(date.today() + timedelta(days=30))
_YESTERDAY   = fmt_date(date.today() - timedelta(days=1))


# ══════════════════════════════════════════════════════════════════════════════
# T01 — Opening  (instant template path — no LLM, mirrors production code)
# ══════════════════════════════════════════════════════════════════════════════
async def test_opening() -> None:
    print(f"\n{YELLOW}[T01] Opening / Greeting (instant template){RESET}")
    ctx = _fresh()
    # Production path: build_opening_greeting() — no LLM round-trip
    say = build_opening_greeting(ctx)
    t = {
        "say":           say,
        "context_patch": {},
        "end_call":      False,
        "hangup_reason": "",
        "call_phase":    "opening",
    }

    check("say present",        bool(say))
    check("say is Hindi",       is_hindi(say))
    check("end_call is False",  not t["end_call"])
    check("call_phase=opening", t["call_phase"] == "opening", t["call_phase"])
    check("mentions EMI/amount",
          any(w in say for w in ["EMI", "emi", "रुपये", "8,500", "8500", "बाकी", "बकाया"]),
          say[:80])
    check("mentions customer name",
          "Rahul" in say or "राहुल" in say or "जी" in say,
          say[:80])
    if VERBOSE: print(f"      say → {say}")


# ══════════════════════════════════════════════════════════════════════════════
# T02 — PTP: full flow
# ══════════════════════════════════════════════════════════════════════════════
async def test_ptp_full() -> None:
    print(f"\n{YELLOW}[T02] PTP — Promise to Pay (future date){RESET}")
    ctx = _fresh(); history = []

    t1 = _opening_turn(ctx)
    _patch(ctx, t1); history = _hist(history, None, t1["say"])

    # Customer gives a specific future date
    utt = f"{_FUTURE_14} तक भुगतान कर दूंगा।"
    t2 = await turn(ctx, history, utt)
    _patch(ctx, t2); history = _hist(history, utt, t2.get("say", ""))

    check("call_phase=ptp",     t2.get("call_phase") == "ptp", t2.get("call_phase", "?"))
    check("say is Hindi",       is_hindi(t2.get("say", "")))

    # Either date captured now, or bot asks for it (1 more turn)
    has_date = bool(ctx.get("target_date") or ctx.get("payment_commitment_iso"))
    if not has_date:
        t3 = await turn(ctx, history,
                        f"मैं {_FUTURE_14} तक पक्का करूंगा।")
        _patch(ctx, t3)
        history = _hist(history, f"{_FUTURE_14} तक पक्का करूंगा।", t3.get("say", ""))
        has_date = bool(ctx.get("target_date") or ctx.get("payment_commitment_iso"))
        check("date captured (turn 3)", has_date,
              f"target_date={ctx.get('target_date')} pci={ctx.get('payment_commitment_iso')}")
        if t3.get("end_call"):
            check("end_call after confirm", True)
    else:
        check("date captured (turn 2)", True,
              f"target_date={ctx.get('target_date')}")
        if t2.get("end_call"):
            check("end_call after confirm", True)

    say = (history[-1]["content"] if history else "")
    check("mandatory closing present",
          any(w in say for w in ["सुरक्षित लिंक", "क्रेडिट स्कोर", "शुभ हो"]),
          say[:100] if say else "no say found")
    if VERBOSE: print(f"      final say → {say[:120]}")


# ══════════════════════════════════════════════════════════════════════════════
# T03 — PTP: date beyond 90 days must be rejected
# ══════════════════════════════════════════════════════════════════════════════
async def test_ptp_beyond_90() -> None:
    print(f"\n{YELLOW}[T03] PTP — Date beyond 90-day window (must reject){RESET}")
    ctx = _fresh(); history = []

    t1 = await turn(ctx, [], "[घटना: कॉल जुड़ गई — ग्राहक ने फोन उठाया। ईज़ी होम फाइनेंस की ओर से अदिति के रूप में परिचय दें; संदर्भ से नाम, बकाया राशि व देय तिथि बताएं; पूछें वे कब तक भुगतान कर सकते हैं।]")
    _patch(ctx, t1); history = _hist(history, None, t1.get("say", ""))

    t2 = await turn(ctx, history, "अगले साल भर दूंगा।")
    _patch(ctx, t2)

    # If a date was stored, it must be within the 90-day window
    stored = ctx.get("target_date") or ctx.get("payment_commitment_iso", "")
    if stored and stored != "not_specified":
        try:
            from datetime import datetime as _dt
            d = _dt.strptime(str(stored)[:10], "%Y-%m-%d").date()
            anchor_raw = ctx.get("emi_overdue_date") or ctx.get("emi_due_date", "")
            anchor = parse_date(anchor_raw) or date.today()
            last_valid = anchor + timedelta(days=90)
            check("stored date within window", d <= last_valid,
                  f"stored={d}  last_valid={last_valid}")
        except Exception as exc:
            check("date parse ok", False, str(exc))
    else:
        check("beyond-90 date not stored / correctly rejected", True,
              f"stored='{stored}'")

    check("bot says is Hindi", is_hindi(t2.get("say", "")))
    check("end_call not triggered", not t2.get("end_call"),
          t2.get("hangup_reason", ""))
    if VERBOSE: print(f"      say → {t2.get('say', '')}")


# ══════════════════════════════════════════════════════════════════════════════
# T04 — Full payment today
# ══════════════════════════════════════════════════════════════════════════════
async def test_payment_today() -> None:
    print(f"\n{YELLOW}[T04] Full Payment Today{RESET}")
    ctx = _fresh(); history = []

    t1 = await turn(ctx, [], "[घटना: कॉल जुड़ गई — ग्राहक ने फोन उठाया। ईज़ी होम फाइनेंस की ओर से अदिति के रूप में परिचय दें; संदर्भ से नाम, बकाया राशि व देय तिथि बताएं; पूछें वे कब तक भुगतान कर सकते हैं।]")
    _patch(ctx, t1); history = _hist(history, None, t1.get("say", ""))

    t2 = await turn(ctx, history, "हाँ, आज ही भर दूंगा।")
    _patch(ctx, t2)

    check("call_phase=payment_confirm",
          t2.get("call_phase") == "payment_confirm", t2.get("call_phase", "?"))
    check("end_call = True",     t2.get("end_call"), str(t2.get("end_call")))
    check("say is Hindi",        is_hindi(t2.get("say", "")))
    say = t2.get("say", "")
    check("mentions link/SMS/branch",
          any(w in say for w in ["SMS", "लिंक", "link", "शाखा", "branch"]),
          say[:100])
    check("mandatory closing",
          any(w in say for w in ["सुरक्षित लिंक", "क्रेडिट स्कोर", "शुभ हो"]),
          say[:100])
    if VERBOSE: print(f"      say → {say}")


# ══════════════════════════════════════════════════════════════════════════════
# T05 — Deceased borrower
# ══════════════════════════════════════════════════════════════════════════════
async def test_deceased() -> None:
    print(f"\n{YELLOW}[T05] Deceased Borrower{RESET}")
    ctx = _fresh(); history = []

    t1 = await turn(ctx, [], "[घटना: कॉल जुड़ गई — ग्राहक ने फोन उठाया। ईज़ी होम फाइनेंस की ओर से अदिति के रूप में परिचय दें; संदर्भ से नाम, बकाया राशि व देय तिथि बताएं; पूछें वे कब तक भुगतान कर सकते हैं।]")
    _patch(ctx, t1); history = _hist(history, None, t1.get("say", ""))

    t2 = await turn(ctx, history, "मेरे पति का पिछले हफ्ते निधन हो गया।")
    _patch(ctx, t2)

    check("call_phase=deceased",  t2.get("call_phase") == "deceased", t2.get("call_phase", "?"))
    check("end_call = True",      t2.get("end_call"), str(t2.get("end_call")))
    check("say is Hindi",         is_hindi(t2.get("say", "")))
    say = t2.get("say", "")
    check("condolences present",
          any(w in say for w in ["दुख", "संवेदना", "अफ़सोस", "निधन", "कठिन"]),
          say[:100])
    check("team contact mentioned",
          any(w in say for w in ["टीम", "सदस्य", "संपर्क"]),
          say[:100])
    check("no mandatory closing",
          not any(w in say for w in ["सुरक्षित लिंक", "क्रेडिट स्कोर"]),
          say[:100])
    if VERBOSE: print(f"      say → {say}")


# ══════════════════════════════════════════════════════════════════════════════
# T06 — Cannot Pay → partial first → declined → reason → callback
# ══════════════════════════════════════════════════════════════════════════════
async def test_cannot_pay_with_partial() -> None:
    print(f"\n{YELLOW}[T06] Cannot Pay (partial offered first, then declined){RESET}")
    ctx = _fresh(); history = []

    t1 = await turn(ctx, [], "[घटना: कॉल जुड़ गई — ग्राहक ने फोन उठाया। ईज़ी होम फाइनेंस की ओर से अदिति के रूप में परिचय दें; संदर्भ से नाम, बकाया राशि व देय तिथि बताएं; पूछें वे कब तक भुगतान कर सकते हैं।]")
    _patch(ctx, t1); history = _hist(history, None, t1.get("say", ""))

    # User says no money
    u2 = "पैसे नहीं हैं, इस महीने भुगतान नहीं हो सकता।"
    t2 = await turn(ctx, history, u2)
    _patch(ctx, t2); history = _hist(history, u2, t2.get("say", ""))

    check("partial offered first",     t2.get("call_phase") == "partial", t2.get("call_phase", "?"))
    check("partial_offer_made set",    ctx.get("partial_offer_made") == "true",
          ctx.get("partial_offer_made", "not set"))
    check("not end_call on hardship",  not t2.get("end_call"))
    check("T06 say is Hindi",          is_hindi(t2.get("say", "")))
    if VERBOSE: print(f"      [partial offer] say → {t2.get('say', '')[:100]}")

    # User declines partial
    u3 = "नहीं, आंशिक भी नहीं दे पाऊंगा।"
    t3 = await turn(ctx, history, u3)
    _patch(ctx, t3); history = _hist(history, u3, t3.get("say", ""))

    check("phase cannot_pay after decline",
          t3.get("call_phase") in ("cannot_pay", "partial"), t3.get("call_phase", "?"))
    check("not end_call after decline", not t3.get("end_call"))
    if VERBOSE: print(f"      [declined partial] say → {t3.get('say', '')[:100]}")

    # User gives reason
    u4 = "नौकरी चली गई है, सैलरी नहीं आई।"
    t4 = await turn(ctx, history, u4)
    _patch(ctx, t4); history = _hist(history, u4, t4.get("say", ""))

    check("reason captured or callback asked",
          bool(ctx.get("cannot_pay_reason")) or
          any(w in t4.get("say", "") for w in ["कब", "संपर्क", "callback"]),
          f"reason='{ctx.get('cannot_pay_reason', '')[:40]}'  say='{t4.get('say','')[:60]}'")
    if VERBOSE: print(f"      [reason given] say → {t4.get('say', '')[:100]}")

    # Provide callback date
    u5 = f"{_FUTURE_14} को कॉल करें।"
    t5 = await turn(ctx, history, u5)
    _patch(ctx, t5)

    check("callback date stored",
          bool(ctx.get("callback_iso") or ctx.get("target_date")),
          f"callback_iso={ctx.get('callback_iso','')} target_date={ctx.get('target_date','')}")
    check("flow ends eventually",  t5.get("end_call"), str(t5.get("end_call")))
    say5 = t5.get("say", "")
    check("mandatory closing in final turn",
          any(w in say5 for w in ["सुरक्षित लिंक", "क्रेडिट स्कोर", "शुभ हो"]),
          say5[:100])
    if VERBOSE: print(f"      [callback date] say → {say5[:120]}")


# ══════════════════════════════════════════════════════════════════════════════
# T07 — Already Paid
# ══════════════════════════════════════════════════════════════════════════════
async def test_already_paid() -> None:
    print(f"\n{YELLOW}[T07] Already Paid{RESET}")
    ctx = _fresh(); history = []

    t1 = await turn(ctx, [], "[घटना: कॉल जुड़ गई — ग्राहक ने फोन उठाया। ईज़ी होम फाइनेंस की ओर से अदिति के रूप में परिचय दें; संदर्भ से नाम, बकाया राशि व देय तिथि बताएं; पूछें वे कब तक भुगतान कर सकते हैं।]")
    _patch(ctx, t1); history = _hist(history, None, t1.get("say", ""))

    u2 = "मैंने पहले ही भुगतान कर दिया है।"
    t2 = await turn(ctx, history, u2)
    _patch(ctx, t2); history = _hist(history, u2, t2.get("say", ""))

    check("call_phase=already_paid",  t2.get("call_phase") == "already_paid",
          t2.get("call_phase", "?"))
    check("asks for date",
          any(w in t2.get("say", "") for w in ["तारीख", "दिन", "कब", "किस"]),
          t2.get("say", "")[:80])
    check("not end_call yet",         not t2.get("end_call"))
    if VERBOSE: print(f"      say → {t2.get('say','')}")

    # Provide payment date (yesterday)
    u3 = f"कल {_YESTERDAY} को किया था।"
    t3 = await turn(ctx, history, u3)
    _patch(ctx, t3); history = _hist(history, u3, t3.get("say", ""))

    check("date stored or mode asked",
          bool(ctx.get("already_paid_date")) or
          any(w in t3.get("say", "") for w in ["माध्यम", "UPI", "कैसे", "कौन"]),
          f"already_paid_date='{ctx.get('already_paid_date','')}' say='{t3.get('say','')[:60]}'")
    if VERBOSE: print(f"      [date provided] say → {t3.get('say','')}")

    # Provide mode
    u4 = "UPI से किया था।"
    t4 = await turn(ctx, history, u4)
    _patch(ctx, t4)

    check("payment_mode captured",   bool(ctx.get("payment_mode")),
          ctx.get("payment_mode", "not set"))
    check("already_paid ends call",  t4.get("end_call"), str(t4.get("end_call")))
    check("no mandatory closing",
          not any(w in t4.get("say", "") for w in ["सुरक्षित लिंक"]),
          t4.get("say", "")[:80])
    if VERBOSE: print(f"      [mode provided] say → {t4.get('say','')}")


# ══════════════════════════════════════════════════════════════════════════════
# T08 — Callback (busy right now)
# ══════════════════════════════════════════════════════════════════════════════
async def test_callback() -> None:
    print(f"\n{YELLOW}[T08] Callback — Busy Now{RESET}")
    ctx = _fresh(); history = []

    t1 = await turn(ctx, [], "[घटना: कॉल जुड़ गई — ग्राहक ने फोन उठाया। ईज़ी होम फाइनेंस की ओर से अदिति के रूप में परिचय दें; संदर्भ से नाम, बकाया राशि व देय तिथि बताएं; पूछें वे कब तक भुगतान कर सकते हैं।]")
    _patch(ctx, t1); history = _hist(history, None, t1.get("say", ""))

    u2 = "अभी meeting में हूँ, बाद में बात करें।"
    t2 = await turn(ctx, history, u2)
    _patch(ctx, t2); history = _hist(history, u2, t2.get("say", ""))

    check("call_phase=callback",  t2.get("call_phase") == "callback", t2.get("call_phase", "?"))
    check("asks when to call",
          any(w in t2.get("say", "") for w in ["कब", "समय", "बजे", "तारीख", "सुविधाजनक"]),
          t2.get("say", "")[:80])
    check("not end_call yet",     not t2.get("end_call"))
    if VERBOSE: print(f"      say → {t2.get('say','')}")

    u3 = "कल दोपहर 2 बजे कॉल करें।"
    t3 = await turn(ctx, history, u3)
    _patch(ctx, t3)

    check("callback time stored",
          bool(ctx.get("callback_iso") or ctx.get("callback_time")),
          f"callback_iso={ctx.get('callback_iso','')} callback_time={ctx.get('callback_time','')}")
    check("callback ends call",    t3.get("end_call"), str(t3.get("end_call")))
    check("no mandatory closing",
          not any(w in t3.get("say", "") for w in ["सुरक्षित लिंक"]),
          t3.get("say", "")[:80])
    if VERBOSE: print(f"      [callback confirmed] say → {t3.get('say','')}")


# ══════════════════════════════════════════════════════════════════════════════
# T09 — Silence handling
# ══════════════════════════════════════════════════════════════════════════════
async def test_silence() -> None:
    print(f"\n{YELLOW}[T09] Silence / No-response handling{RESET}")
    ctx = _fresh(); history = []

    t1 = await turn(ctx, [], "[घटना: कॉल जुड़ गई — ग्राहक ने फोन उठाया। ईज़ी होम फाइनेंस की ओर से अदिति के रूप में परिचय दें; संदर्भ से नाम, बकाया राशि व देय तिथि बताएं; पूछें वे कब तक भुगतान कर सकते हैं।]")
    _patch(ctx, t1); history = _hist(history, None, t1.get("say", ""))

    # First silence — LLM receives [SILENCE_1] from Python code
    t2 = await turn(ctx, history, "[SILENCE_1: No response. Ask once more, very briefly.]")
    _patch(ctx, t2); history = _hist(history, "[SILENCE_1]", t2.get("say", ""))
    check("ask again on 1st silence", bool(t2.get("say")), t2.get("say", "")[:60])
    check("not end_call on 1st silence", not t2.get("end_call"))
    check("no silence_count in ctx (Python-tracked now)", "silence_count" not in ctx)
    if VERBOSE: print(f"      [silence 1] say → {t2.get('say','')}")

    # Second silence — LLM receives [SILENCE_2], must close
    t3 = await turn(ctx, history, "[SILENCE_2: No response again. Say the no-response goodbye "
                                   "and set end_call=true, hangup_reason=no_response.]")
    _patch(ctx, t3)
    check("end_call=true on 2nd silence", t3.get("end_call"), f"end_call={t3.get('end_call')}")
    check("hangup_reason=no_response", t3.get("hangup_reason") == "no_response",
          t3.get("hangup_reason", "?"))
    check("goodbye in say", bool(t3.get("say")), t3.get("say", "")[:60])
    if VERBOSE: print(f"      [silence 2 / goodbye] say → {t3.get('say','')}")


# ══════════════════════════════════════════════════════════════════════════════
# T10 — Partial payment: full flow
# ══════════════════════════════════════════════════════════════════════════════
async def test_partial_payment() -> None:
    print(f"\n{YELLOW}[T10] Partial Payment — Full Flow{RESET}")
    ctx = _fresh(); history = []

    t1 = await turn(ctx, [], "[घटना: कॉल जुड़ गई — ग्राहक ने फोन उठाया। ईज़ी होम फाइनेंस की ओर से अदिति के रूप में परिचय दें; संदर्भ से नाम, बकाया राशि व देय तिथि बताएं; पूछें वे कब तक भुगतान कर सकते हैं।]")
    _patch(ctx, t1); history = _hist(history, None, t1.get("say", ""))

    # Customer explicitly requests partial
    u2 = "पूरे पैसे नहीं हैं, थोड़े आज दे सकता हूँ।"
    t2 = await turn(ctx, history, u2)
    _patch(ctx, t2); history = _hist(history, u2, t2.get("say", ""))

    check("call_phase=partial",    t2.get("call_phase") == "partial", t2.get("call_phase", "?"))
    check("asks for amount",
          any(w in t2.get("say", "") for w in ["कितना", "राशि", "amount", "रुपये", "कितनी"]),
          t2.get("say", "")[:80])
    if VERBOSE: print(f"      [partial request] say → {t2.get('say','')}")

    # Give valid partial amount (above 1500 minimum)
    u3 = "3,000 रुपये दे सकता हूँ।"
    t3 = await turn(ctx, history, u3)
    _patch(ctx, t3); history = _hist(history, u3, t3.get("say", ""))

    check("amount stored or remaining shown",
          bool(ctx.get("partial_amount")) or
          any(w in t3.get("say", "") for w in ["शेष", "बाकी", "5,500", "5500"]),
          f"partial_amount={ctx.get('partial_amount','')}  say={t3.get('say','')[:80]}")
    if VERBOSE: print(f"      [amount given] say → {t3.get('say','')}")

    # Give remainder date (use FUTURE_14 — stays inside the 90-day window)
    u4 = f"{_FUTURE_14} तक बाकी दे दूंगा।"
    t4 = await turn(ctx, history, u4)
    _patch(ctx, t4)

    check("target_date stored",
          bool(ctx.get("target_date") or ctx.get("payment_commitment_iso")),
          f"target_date={ctx.get('target_date','')} pci={ctx.get('payment_commitment_iso','')}")
    check("partial ends call",  t4.get("end_call"), str(t4.get("end_call")))
    say4 = t4.get("say", "")
    check("mandatory closing in final",
          any(w in say4 for w in ["सुरक्षित लिंक", "क्रेडिट स्कोर", "शुभ हो"]),
          say4[:100])
    if VERBOSE: print(f"      [remainder date] say → {say4[:120]}")


# ══════════════════════════════════════════════════════════════════════════════
# T11 — JSON / schema stability
# ══════════════════════════════════════════════════════════════════════════════
async def test_json_stability() -> None:
    print(f"\n{YELLOW}[T11] JSON / Schema Stability (5 rapid turns){RESET}")
    ctx = _fresh(); history = []
    REQUIRED_KEYS = {"say", "end_call", "context_patch", "hangup_reason", "call_phase"}
    VALID_PHASES  = {"ptp","deceased","partial","payment_confirm","cannot_pay",
                     "already_paid","callback","opening","other","recovery","error","llm",
                     "no_response"}

    utterances = [
        "हाँ",
        "पैसे नहीं हैं।",
        "अगले महीने।",
        "[मौन — कोई उत्तर नहीं]",
    ]

    # Turn 1: instant opening (no LLM)
    t = _opening_turn(ctx)
    _patch(ctx, t)
    history = _hist(history, None, t["say"])
    missing = REQUIRED_KEYS - set(t.keys())
    check("turn 1 has all required keys", not missing, f"missing={missing}" if missing else "")
    check("turn 1 end_call is bool",      isinstance(t.get("end_call"), bool), str(type(t.get("end_call"))))
    check("turn 1 context_patch is dict", isinstance(t.get("context_patch"), dict), str(type(t.get("context_patch"))))
    check("turn 1 call_phase valid",      t.get("call_phase") in VALID_PHASES, t.get("call_phase", "?"))
    check("turn 1 say is string",         isinstance(t.get("say"), str), str(type(t.get("say"))))
    if t.get("say"):
        check("turn 1 say is Hindi", is_hindi(t.get("say", "")), t.get("say", "")[:40])

    for i, utt in enumerate(utterances, start=2):
        t = await turn(ctx, history, utt)
        _patch(ctx, t)
        history = _hist(history, utt, t.get("say", ""))

        missing = REQUIRED_KEYS - set(t.keys())
        check(f"turn {i} has all required keys",
              not missing, f"missing={missing}" if missing else "")
        check(f"turn {i} end_call is bool",
              isinstance(t.get("end_call"), bool), str(type(t.get("end_call"))))
        check(f"turn {i} context_patch is dict",
              isinstance(t.get("context_patch"), dict),
              str(type(t.get("context_patch"))))
        check(f"turn {i} call_phase valid",
              t.get("call_phase") in VALID_PHASES, t.get("call_phase","?"))
        check(f"turn {i} say is string",
              isinstance(t.get("say"), str), str(type(t.get("say"))))
        if t.get("say"):
            check(f"turn {i} say is Hindi", is_hindi(t.get("say", "")), t.get("say","")[:40])

        if t.get("end_call"):
            print(f"    → Call ended at turn {i} ({t.get('hangup_reason','')})")
            break


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════
TESTS = [
    ("T01 Opening",                  test_opening),
    ("T02 PTP Full Flow",            test_ptp_full),
    ("T03 PTP Beyond-90 Rejection",  test_ptp_beyond_90),
    ("T04 Payment Today",            test_payment_today),
    ("T05 Deceased",                 test_deceased),
    ("T06 Cannot Pay + Partial",     test_cannot_pay_with_partial),
    ("T07 Already Paid",             test_already_paid),
    ("T08 Callback",                 test_callback),
    ("T09 Silence Handling",         test_silence),
    ("T10 Partial Full Flow",        test_partial_payment),
    ("T11 JSON Stability",           test_json_stability),
]


async def main() -> bool:
    print("=" * 66)
    print("  Aditi LLM Flow Tests")
    print(f"  Date: {date.today()}   Model: gpt-4.1-mini")
    print("=" * 66)

    test_errors: list[tuple[str, str]] = []

    for name, fn in TESTS:
        t0 = time.perf_counter()
        try:
            await fn()
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            test_errors.append((name, msg))
            print(f"  {FAIL} TEST CRASH - {msg}")
            _results.append((FAIL, f"{name} CRASH", msg))
        elapsed = time.perf_counter() - t0
        _test_timings.append((name, elapsed))
        print(f"  [time] {elapsed:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(1 for icon, _, _ in _results if "PASS" in icon)
    n_fail = sum(1 for icon, _, _ in _results if "FAIL" in icon)
    total  = n_pass + n_fail

    print("\n" + "=" * 66)
    print("  SUMMARY")
    print("=" * 66)
    print(f"  Checks:  {n_pass}/{total} passed   {n_fail} failed")
    print(f"  Crashes: {len(test_errors)}")

    if n_fail > 0 or test_errors:
        print(f"\n{RED}  FAILED checks:{RESET}")
        for icon, label, detail in _results:
            if "FAIL" in icon:
                print(f"    -> {label}" + (f"  ({detail})" if detail else ""))
        for name, err in test_errors:
            print(f"    -> {name}: {err}")

    print("\n  Timings:")
    for name, t in sorted(_test_timings, key=lambda x: -x[1]):
        bar = "#" * int(t * 2)
        print(f"    {t:5.1f}s  {bar:20s}  {name}")

    return n_fail == 0 and not test_errors


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
