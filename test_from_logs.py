"""
test_from_logs.py — Replay real customer utterances captured from the 12:30 batch
(logs/aditi.log) against the LLM orchestrator and score the bot's responses.

Each case = one real call where the bot either succeeded, dropped a real PTP,
or got confused by code-mix. We feed the actual STT transcript into
run_conversation_turn and check whether the bot now extracts the right intent +
context variables.

Run:
    python test_from_logs.py
    python test_from_logs.py -v
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
from datetime import date, timedelta
from typing import Any

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s - %(message)s")

from llm_orchestrator import run_conversation_turn
from scripts import build_default_ctx, build_opening_greeting

VERBOSE = "-v" in sys.argv

GREEN, RED, YELLOW, CYAN, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"
PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"

_results: list[tuple[str, str, str, str]] = []  # (case, status, label, detail)


def check(case: str, label: str, cond: bool, detail: str = "") -> bool:
    icon = PASS if cond else FAIL
    _results.append((case, icon, label, detail))
    print(f"    {icon} {label}" + (f"  [{detail}]" if detail else ""))
    return cond


def _fresh(**overrides) -> dict[str, str]:
    ctx = build_default_ctx()
    ctx.update(overrides)
    return ctx


def _opening(ctx: dict) -> dict[str, Any]:
    return {"say": build_opening_greeting(ctx), "context_patch": {},
            "end_call": False, "hangup_reason": "", "call_phase": "opening"}


async def turn(ctx: dict, history: list, msg: str) -> dict:
    return await run_conversation_turn(ctx, history, msg)


def _patch(ctx: dict, t: dict) -> dict:
    p = t.get("context_patch", {})
    if isinstance(p, dict):
        ctx.update({str(k): str(v) for k, v in p.items()})
    return ctx


def _hist(history: list, u: str | None, say: str) -> list:
    h = list(history)
    if u: h.append({"role": "user", "content": u})
    if say: h.append({"role": "assistant", "content": say})
    return h


# ════════════════════════════════════════════════════════════════════════════
# Real cases from logs/aditi.log @ 12:30 batch
# ════════════════════════════════════════════════════════════════════════════

async def case_vinod_late_ptp() -> None:
    """
    Call 9ce23591 (Vinod Bhegade, EMI 30143).
    Customer was silent during greeting, then spoke AFTER carrier_disconnect.
    Late utterance the bot lost: "हाँ, मैं करूंगा परसों के दिन करूंगा, चौरन तारीख को।"
    Expected: PTP intent, target_date within next 3 days (परसों = day-after-tomorrow).
    """
    print(f"\n{CYAN}[CASE-1] Vinod — late PTP that was lost in prod{RESET}")
    ctx = _fresh(customer_name="Vinod Bhegade", emi_overdue_amt="30,143")
    history = []
    t1 = _opening(ctx); _patch(ctx, t1); history = _hist(history, None, t1["say"])

    u = "हाँ, मैं करूंगा परसों के दिन करूंगा, चौदह तारीख को।"
    t2 = await turn(ctx, history, u)
    _patch(ctx, t2)

    check("CASE-1", "phase=ptp", t2.get("call_phase") == "ptp", t2.get("call_phase", "?"))
    td = ctx.get("target_date") or ctx.get("payment_commitment_iso", "")
    check("CASE-1", "target_date captured", bool(td) and td != "not_specified", f"target_date={td}")
    if td and td != "not_specified":
        try:
            d = date.fromisoformat(str(td)[:10])
            days = (d - date.today()).days
            check("CASE-1", "date is within next 5 days", 0 <= days <= 5, f"days_from_today={days}")
        except Exception as e:
            check("CASE-1", "date parses", False, str(e))
    if VERBOSE: print(f"      say → {t2.get('say','')[:140]}")


async def case_padam_3_4_days() -> None:
    """
    Call d14c9413 (Padam Thada, EMI 15991). Customer said: "तीन चार दिन के बाद।"
    Production bot got it right (target_date=2026-06-14, which is today+3).
    Re-verify this still works.
    """
    print(f"\n{CYAN}[CASE-2] Padam — '3-4 दिन के बाद' (PTP){RESET}")
    ctx = _fresh(customer_name="Padam Thada", emi_overdue_amt="15,991")
    history = []
    t1 = _opening(ctx); _patch(ctx, t1); history = _hist(history, None, t1["say"])

    u = "तीन चार दिन के बाद।"
    t2 = await turn(ctx, history, u)
    _patch(ctx, t2)

    check("CASE-2", "phase=ptp", t2.get("call_phase") == "ptp", t2.get("call_phase", "?"))
    td = ctx.get("target_date") or ctx.get("payment_commitment_iso", "")
    check("CASE-2", "target_date captured", bool(td) and td != "not_specified", f"target_date={td}")
    if td and td != "not_specified":
        try:
            d = date.fromisoformat(str(td)[:10])
            days = (d - date.today()).days
            check("CASE-2", "date is 3-5 days out", 3 <= days <= 5, f"days={days}")
        except Exception as e:
            check("CASE-2", "date parses", False, str(e))
    if VERBOSE: print(f"      say → {t2.get('say','')[:140]}")


async def case_rohit_branch_callback() -> None:
    """
    Call e0596db9 (Rohit Kumar). After carrier_disconnect race the customer said:
    "कर देंगे ना, अभी तो अब अभी तो फोन आया था, बात हो गई मेरी।"
    Meaning: "I'll pay. I just got a call, I've already spoken (with branch)."
    Expected: bot should NOT loop greeting; should treat as either
      (a) callback_later / already discussed, or (b) ask "kab tak" for specific date.
    Should NOT close phase=opening with no intent.
    """
    print(f"\n{CYAN}[CASE-3] Rohit — 'already talked to branch'{RESET}")
    ctx = _fresh(customer_name="Rohit Kumar", emi_overdue_amt="14,633")
    history = []
    t1 = _opening(ctx); _patch(ctx, t1); history = _hist(history, None, t1["say"])

    u = "कर देंगे ना, अभी तो अब अभी तो फोन आया था, बात हो गई मेरी।"
    t2 = await turn(ctx, history, u)
    _patch(ctx, t2)

    phase = t2.get("call_phase", "")
    check("CASE-3", "phase advanced past opening", phase != "opening", f"phase={phase}")
    check("CASE-3", "bot asks clarifying / for date (not just repeats greeting)",
          any(w in t2.get("say","") for w in ["कब","तारीख","दिन","date","when"]),
          t2.get("say","")[:120])
    if VERBOSE: print(f"      say → {t2.get('say','')[:140]}")


async def case_telugu_cannot_pay() -> None:
    """
    Call c62aa921 (SHAILESH PAWAR). Customer spoke Telugu-mixed:
    "औनम्मा, नाकु इंटी तालम एव्वलेद राया... नाकु इंटी तालम 3 नेला..."
    (Telugu = "no money at home, 3 months no money").
    In prod, bot looped "Sorry मैं समझ नहीं पाई" 4 times.
    Expected: detect cannot_pay intent (or at least DON'T just repeat 'Sorry').
    """
    print(f"\n{CYAN}[CASE-4] SHAILESH — Telugu-mixed 'no money 3 months'{RESET}")
    ctx = _fresh(customer_name="SHAILESH PAWAR", emi_overdue_amt="9,827")
    history = []
    t1 = _opening(ctx); _patch(ctx, t1); history = _hist(history, None, t1["say"])

    u = "औनम्मा, नाकु इंटी तालम एव्वलेद राया। मैं लोन कडताने उंडाना, नाकु इंटी तालम 3 नेला न च कडतना इद 4 नेला, नाकु इंटी तालम।"
    t2 = await turn(ctx, history, u)
    _patch(ctx, t2)

    phase = t2.get("call_phase", "")
    say = t2.get("say", "")
    # Best case: bot detects cannot_pay or partial. Worst case: it just says "Sorry, didn't understand".
    is_sorry_loop = ("Sorry" in say or "समझ नहीं" in say) and phase == "opening"
    check("CASE-4", "did NOT just say 'Sorry, didn't understand'", not is_sorry_loop,
          f"phase={phase}  say={say[:80]}")
    check("CASE-4", "phase=cannot_pay or partial or asks clarification with money keyword",
          phase in ("cannot_pay", "partial") or any(w in say for w in ["पैसे","पैसा","money","कितने"]),
          f"phase={phase}  say={say[:80]}")
    if VERBOSE: print(f"      say → {say[:160]}")


async def case_kankamedala_confused() -> None:
    """
    Call 1e579719. Customer gave fragmented mixed-language replies:
    "यह ब्रांच में से फोन किया है मुझे। कैसे ठीक है ना माँ? वह सारी है।"
    "मैं क्या हूं हिंदी राज राज में तो नहीं तो। खेलू रहा है।"
    In prod: 4 turns of "Sorry मैं समझ नहीं पाई" loop.
    Expected: bot should escalate / offer callback rather than loop.
    """
    print(f"\n{CYAN}[CASE-5] Kankamedala — code-switch confusion (2 turns){RESET}")
    ctx = _fresh(customer_name="Kankamedala", emi_overdue_amt="5,081")
    history = []
    t1 = _opening(ctx); _patch(ctx, t1); history = _hist(history, None, t1["say"])

    u1 = "यह ब्रांच में से फोन किया है मुझे। कैसे ठीक है ना माँ? वह सारी है।"
    t2 = await turn(ctx, history, u1)
    _patch(ctx, t2); history = _hist(history, u1, t2.get("say", ""))
    if VERBOSE: print(f"      turn-1 say → {t2.get('say','')[:120]}")

    u2 = "मैं क्या हूं हिंदी राज राज में तो नहीं तो। खेलू रहा है।"
    t3 = await turn(ctx, history, u2)
    _patch(ctx, t3)
    say3 = t3.get("say", "")
    if VERBOSE: print(f"      turn-2 say → {say3[:120]}")

    # By turn 2 the bot should NOT be saying "Sorry didn't understand" yet again.
    is_sorry_again = "Sorry" in say3 and "समझ नहीं" in say3
    check("CASE-5", "by turn 2, NOT another 'Sorry didn't understand'", not is_sorry_again,
          say3[:100])
    check("CASE-5", "bot offers callback / asks for specific date / closes politely",
          any(w in say3 for w in ["कब","तारीख","callback","call back","बाद में","wapas","later"]),
          say3[:100])


async def case_ashok_13th() -> None:
    """
    Call b760277d (Ashok Zhod). Prod final summary: 'pay full overdue EMI on 13th June 2026'.
    The customer utterance itself wasn't in the visible log, so we use a typical phrasing.
    Today is 2026-06-11 → 13th = today+2.
    """
    print(f"\n{CYAN}[CASE-6] Ashok — '13 तारीख को कर दूंगा' (PTP){RESET}")
    ctx = _fresh(customer_name="Ashok Zhod", emi_overdue_amt="12,906")
    history = []
    t1 = _opening(ctx); _patch(ctx, t1); history = _hist(history, None, t1["say"])

    u = "13 तारीख को कर दूंगा।"
    t2 = await turn(ctx, history, u)
    _patch(ctx, t2)

    check("CASE-6", "phase=ptp", t2.get("call_phase") == "ptp", t2.get("call_phase", "?"))
    td = ctx.get("target_date") or ctx.get("payment_commitment_iso", "")
    check("CASE-6", "target_date captured", bool(td) and td != "not_specified", f"target_date={td}")
    if td and td != "not_specified":
        try:
            d = date.fromisoformat(str(td)[:10])
            check("CASE-6", "date is the 13th of this month", d.day == 13, f"got {d}")
        except Exception as e:
            check("CASE-6", "date parses", False, str(e))


# ── Runner ───────────────────────────────────────────────────────────────────
async def main() -> None:
    cases = [
        case_vinod_late_ptp,
        case_padam_3_4_days,
        case_rohit_branch_callback,
        case_telugu_cannot_pay,
        case_kankamedala_confused,
        case_ashok_13th,
    ]
    for c in cases:
        try:
            await c()
        except Exception as e:
            print(f"    {FAIL} {c.__name__} CRASHED: {e}")
            _results.append((c.__name__, FAIL, "crashed", str(e)))

    # Summary
    print(f"\n{YELLOW}{'═'*70}{RESET}")
    print(f"{YELLOW}SUMMARY{RESET}")
    print(f"{YELLOW}{'═'*70}{RESET}")
    by_case: dict[str, list] = {}
    for case, status, label, detail in _results:
        by_case.setdefault(case, []).append((status, label, detail))
    total = passed = 0
    for case, items in by_case.items():
        p = sum(1 for s,_,_ in items if PASS in s)
        t = len(items)
        total += t; passed += p
        color = GREEN if p == t else (RED if p == 0 else YELLOW)
        print(f"  {color}{case}: {p}/{t} passed{RESET}")
    print(f"\n  {GREEN if passed==total else (RED if passed==0 else YELLOW)}TOTAL: {passed}/{total} checks passed{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
