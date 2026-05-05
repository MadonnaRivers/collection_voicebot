"""
Legacy classifier tests — the scripted FSM and per-state classify() labels were removed.

Conversation logic lives in llm_orchestrator.run_conversation_turn().
See classifier.finalize_call_variables for post-call CRM JSON.

Manual check:
  python -c "import asyncio; from llm_orchestrator import run_conversation_turn; from scripts import build_default_ctx; asyncio.run(run_conversation_turn(build_default_ctx(), [], 'हाँ'))"
"""
from __future__ import annotations

import asyncio
import sys


async def _smoke() -> None:
    from llm_orchestrator import run_conversation_turn
    from scripts import build_default_ctx

    ctx = build_default_ctx()
    out = await run_conversation_turn(ctx, [], "[EVENT: CALL_CONNECTED — borrower answered.]")
    assert "say" in out and "end_call" in out
    print("smoke ok:", {k: out[k] for k in ("call_phase", "end_call", "hangup_reason")})


if __name__ == "__main__":
    try:
        asyncio.run(_smoke())
    except Exception as exc:
        print("smoke failed (needs OPENAI_API_KEY and network):", exc, file=sys.stderr)
        sys.exit(1)
