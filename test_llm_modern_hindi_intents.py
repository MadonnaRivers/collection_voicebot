from __future__ import annotations

import asyncio
import io
import json
import sys
from datetime import date, timedelta

from llm_orchestrator import run_conversation_turn
from scripts import build_default_ctx, build_opening_greeting


if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _is_hindi(text: str) -> bool:
    return sum(1 for c in text if "\u0900" <= c <= "\u097f") >= 3


def _hist_add(history: list[dict[str, str]], role: str, content: str) -> list[dict[str, str]]:
    out = list(history)
    out.append({"role": role, "content": content})
    return out


def _merge_ctx(ctx: dict[str, str], out: dict) -> None:
    patch = out.get("context_patch", {})
    if isinstance(patch, dict):
        for k, v in patch.items():
            ctx[str(k)] = str(v)


def _assert_schema(out: dict) -> None:
    for k in ("say", "context_patch", "end_call", "hangup_reason", "call_phase"):
        assert k in out, f"missing key: {k}"
    assert isinstance(out["say"], str)
    assert isinstance(out["context_patch"], dict)
    assert isinstance(out["end_call"], bool)
    assert isinstance(out["hangup_reason"], str)
    assert isinstance(out["call_phase"], str)


async def _turn(ctx: dict[str, str], history: list[dict[str, str]], msg: str) -> dict:
    out = await run_conversation_turn(ctx, history, msg)
    _assert_schema(out)
    assert _is_hindi(out["say"]), f"non-hindi response: {out['say']}"
    return out


async def run_all() -> None:
    print("Modern Hindi intent tests started...")
    today = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    # 0) opening template
    ctx = build_default_ctx()
    opening = build_opening_greeting(ctx)
    assert _is_hindi(opening), "opening is not hindi"
    assert "अदिति" in opening, "opening missing bot identity"
    assert "EMI" in opening or "रुपये" in opening, "opening missing EMI context"
    print("PASS opening template")

    # 1) payment_confirm (today)
    ctx = build_default_ctx()
    history: list[dict[str, str]] = [{"role": "assistant", "content": opening}]
    out = await _turn(ctx, history, "जी, मैं आज ही पूरा भुगतान कर दूंगा।")
    _merge_ctx(ctx, out)
    assert out["call_phase"] == "payment_confirm", out
    assert out["end_call"] is True, out
    assert out["hangup_reason"] == "payment_today_confirmed", out
    print("PASS intent payment_confirm")

    # 2) ptp (future date)
    ctx = build_default_ctx()
    history = [{"role": "assistant", "content": opening}]
    out = await _turn(ctx, history, "मैं कल तक भुगतान कर दूंगा।")
    _merge_ctx(ctx, out)
    assert out["call_phase"] == "ptp", out
    assert out["end_call"] is True, out
    assert out["hangup_reason"] == "ptp_confirmed", out
    assert ctx.get("target_date"), f"target_date missing: {json.dumps(out, ensure_ascii=False)}"
    print("PASS intent ptp")

    # 3) cannot_pay (no partial offer in this flow now)
    ctx = build_default_ctx()
    history = [{"role": "assistant", "content": opening}]
    out1 = await _turn(ctx, history, "इस समय मेरी नौकरी चली गई है, मैं EMI नहीं दे पा रहा हूँ।")
    _merge_ctx(ctx, out1)
    history = _hist_add(history, "user", "इस समय मेरी नौकरी चली गई है, मैं EMI नहीं दे पा रहा हूँ।")
    history = _hist_add(history, "assistant", out1["say"])
    if out1["end_call"]:
        assert out1["call_phase"] == "cannot_pay", out1
        assert out1["hangup_reason"] == "cannot_pay_acknowledged", out1
        assert "partial" not in out1["say"].lower(), out1["say"]
    else:
        out2 = await _turn(ctx, history, "घर में मेडिकल खर्च बहुत ज्यादा है, इसलिए अभी संभव नहीं है।")
        _merge_ctx(ctx, out2)
        assert out2["call_phase"] == "cannot_pay", out2
        assert out2["end_call"] is True, out2
        assert out2["hangup_reason"] == "cannot_pay_acknowledged", out2
        assert "partial" not in out2["say"].lower(), out2["say"]
    print("PASS intent cannot_pay")

    # 4) partial (customer self-offers partial)
    ctx = build_default_ctx()
    history = [{"role": "assistant", "content": opening}]
    out1 = await _turn(ctx, history, "मैं अभी 2000 रुपये दे सकता हूँ।")
    _merge_ctx(ctx, out1)
    assert out1["call_phase"] == "partial", out1
    history = _hist_add(history, "user", "मैं अभी 2000 रुपये दे सकता हूँ।")
    history = _hist_add(history, "assistant", out1["say"])
    out2 = await _turn(ctx, history, "बाकी राशि मैं " + tomorrow + " तक दे दूंगा।")
    _merge_ctx(ctx, out2)
    assert out2["call_phase"] == "partial", out2
    assert out2["end_call"] is True, out2
    assert out2["hangup_reason"] == "partial_confirmed", out2
    assert ctx.get("partial_amount"), f"partial_amount missing: {ctx}"
    assert ctx.get("target_date"), f"target_date missing: {ctx}"
    print("PASS intent partial")

    # 5) already_paid (within 90 days)
    ctx = build_default_ctx()
    history = [{"role": "assistant", "content": opening}]
    paid_date = (date.today() - timedelta(days=1)).strftime("%d %B %Y")
    out1 = await _turn(ctx, history, f"मैंने {paid_date} को भुगतान कर दिया था।")
    _merge_ctx(ctx, out1)
    history = _hist_add(history, "user", f"मैंने {paid_date} को भुगतान कर दिया था।")
    history = _hist_add(history, "assistant", out1["say"])
    if out1["end_call"]:
        assert out1["call_phase"] == "already_paid", out1
        assert out1["hangup_reason"] == "already_paid_noted", out1
    else:
        out2 = await _turn(ctx, history, "UPI से किया था।")
        _merge_ctx(ctx, out2)
        assert out2["call_phase"] == "already_paid", out2
        assert out2["end_call"] is True, out2
        assert out2["hangup_reason"] == "already_paid_noted", out2
    print("PASS intent already_paid")

    # 6) deceased
    ctx = build_default_ctx()
    history = [{"role": "assistant", "content": opening}]
    out = await _turn(ctx, history, "जिनके नाम पर लोन है उनका निधन हो चुका है।")
    _merge_ctx(ctx, out)
    assert out["call_phase"] == "deceased", out
    assert out["end_call"] is True, out
    assert out["hangup_reason"] == "deceased", out
    print("PASS intent deceased")

    # 7) no_response silent tokens
    ctx = build_default_ctx()
    history = [{"role": "assistant", "content": opening}]
    s1 = await _turn(ctx, history, "[SILENCE_1]")
    history = _hist_add(history, "user", "[SILENCE_1]")
    history = _hist_add(history, "assistant", s1["say"])
    assert s1["end_call"] is False, s1
    s2 = await _turn(ctx, history, "[SILENCE_2]")
    history = _hist_add(history, "user", "[SILENCE_2]")
    history = _hist_add(history, "assistant", s2["say"])
    assert s2["end_call"] is False, s2
    s3 = await _turn(ctx, history, "[SILENCE_3]")
    assert s3["call_phase"] == "no_response", s3
    assert s3["end_call"] is True, s3
    assert s3["hangup_reason"] == "no_response", s3
    print("PASS silence flow")

    print("ALL PASS modern Hindi intent tests")


if __name__ == "__main__":
    asyncio.run(run_all())
