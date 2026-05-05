"""
Legacy FSM simulation — removed when the flow moved to llm_orchestrator.

The model now drives each turn (say, context_patch, end_call) in call_handler.llm_conversation_loop.

To exercise the LLM: run test_classifier.py (smoke) or place a real test call.
"""
from __future__ import annotations

if __name__ == "__main__":
    print("FSM test_cases are obsolete. Use: python test_classifier.py")
    print("or integration tests against llm_orchestrator.run_conversation_turn().")
