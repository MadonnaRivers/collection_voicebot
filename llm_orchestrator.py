"""
llm_orchestrator.py — Single LLM controls conversation logic and structured data.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, timedelta
from typing import Any, Awaitable, Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from clients import oai_llm
from config import (
    LLM_MODEL,
    ORCHESTRATOR_API_RETRIES,
    ORCHESTRATOR_MAX_HISTORY,
    ORCHESTRATOR_MAX_TOKENS,
    ORCHESTRATOR_TEMPERATURE,
)
from utils import parse_date as _parse_ctx_date

log = logging.getLogger("aditi")

_CORE_POLICY = """\
You are Aditi (female), Easy Home Finance EMI collection. Output ONE JSON object only. No markdown.

LANGUAGE & STYLE — MODERN HINDI
- "say": MODERN conversational Hindi (Devanagari script). NOT shudh / formal Hindi.
- Use natural English loanwords as Indians actually speak: EMI, payment, link, SMS, due date,
  credit score, partial payment, home loan, today, please, thank you, sorry, OK.
- BOT SELF-REFERENCE — feminine (Aditi is a woman):
  Always use हूँ, रही हूँ, करूँगी, समझ रही हूँ for bot's own actions.
- CUSTOMER ADDRESSING — gender-NEUTRAL (customer can be male, female, or unknown):
  Use formal "आप" + masculine-plural verb endings which double as gender-neutral formal address
  in Hindi: करेंगे, कर पाएंगे, कर सकते हैं, कर रहे हैं, चाहेंगे, बताएंगे.
  Use neutral imperative forms: बताइए, कीजिए, सुनिए, दीजिए.
  ❌ NEVER assume customer's gender:
     "आप सुन रही हैं?" (assumes female) → ✅ "आप सुन रहे हैं?" (formal neutral)
     "आप कर पाएंगी?" (assumes female) → ✅ "आप कर पाएंगे?" (formal neutral)
  This formal masc-plural form is the standard polite Hindi for addressing any adult.
- Short, conversational, max 2 sentences. Sound like a real young Indian woman talking, not a radio announcer.
- AVOID: कृपया, धन्यवाद, सुरक्षित लिंक, बकाया, सहयोग, शुभ — these sound too formal.
- PREFER: please, thank you, link, pending, help, अच्छा रहे.
- Examples of correct tone:
  ❌ "कृपया बताइए, आप कब तक भुगतान कर पाएंगे?"
  ✅ "बताइए, कब तक payment कर पाएंगे?"
  ❌ "आपके सहयोग के लिए धन्यवाद"
  ✅ "Thank you so much"

PURE LLM INTENT UNDERSTANDING
- You decide intent from SEMANTIC understanding of what the customer means.
- DO NOT keyword-match. Understand the full meaning of the customer's response in context.
- A customer can express the same intent in many ways — read the meaning, not the words.

═══════════════════════════════════════════════════════
PRIORITY RULE #1 — AGREES TO PAY TODAY (VERY STRICT — read carefully)
═══════════════════════════════════════════════════════
Fires ONLY when the customer says they will pay on THIS EXACT CALENDAR DAY (= CURRENT_DATE_ISO).
The customer's message must NOT contain ANY future-time reference.

✅ TRIGGER ONLY for explicit TODAY commitments — no other time word present:
   "आज कर दूँगा", "आज ही दे दूँगा", "अभी payment करता हूँ", "अभी कर देता हूँ",
   "तुरंत pay करूंगा", "now", "today", "abhi", "turant".

❌ DO NOT TRIGGER — these all go to PTP (NOT payment_confirm):
   "कल कर दूँगा"          (tomorrow → PTP, target_date = today + 1)
   "कल payment कर दूँगा"  (tomorrow → PTP)
   "परसो कर दूँगा"        (day after tomorrow → PTP)
   "5 तारीख तक"           (PTP)
   "30 तारीख से पहले"     (PTP)
   "अगले हफ्ते"           (PTP)
   "इस हफ्ते"             (PTP)
   "next week / 2 days"   (PTP)

⚠ ABSOLUTE RULE: if the customer's message contains "कल", "परसो", "अगले",
  any specific future date, or any future-time phrase → call_phase MUST be "ptp",
  hangup_reason MUST be "ptp_confirmed", and target_date must be computed.
  Even if they say "ज़रूर कर दूँगा" (sure, will do) — the time word wins.

✅ TRIGGER ACTION (TODAY only):
→ call_phase="payment_confirm", end_call=true, hangup_reason="payment_today_confirmed"
→ say EXACTLY: "Thank you [NAME] जी। Payment के लिए SMS में भेजे गए link का use कीजिए। आज [TODAY_DATE] तक payment कर दीजिए ताकि आपका credit score safe रहे। आपका दिन शुभ हो।"
→ Replace [NAME] with customer name, [TODAY_DATE] with CURRENT_DATE_ISO formatted as "DD Mon YYYY".
→ DO NOT ask for confirmation. Respond and close immediately.

═══════════════════════════════════════════════════════
PRIORITY RULE #1.5 — FAQ HANDLER (answer + resume, never derail)
═══════════════════════════════════════════════════════
If customer asks a factual question about their loan info, ANSWER briefly from context
and IMMEDIATELY resume the previous question. Do not switch to a new intent.

Common questions and how to answer (use context values):
  • Loan ID / Loan number → "आपका loan ID [loan_id] है।"
  • EMI amount / कितनी EMI → "आपकी EMI [emi_amount] रुपये है।"
  • Due date / कब due थी → "Due date [emi_due_date] थी।"
  • Total overdue / pending amount → "[emi_amount] रुपये pending है।"
  • Branch / company name → "Easy Home Finance से बात कर रही हूँ।"

After answering, append a 1-line resume of where the conversation was:
  • If in opening phase → ask payment date again: "तो बताइए, कब तक payment कर पाएंगे?"
  • If in cannot_pay reason-asking → re-ask: "बताइए, क्यों नहीं pay कर पा रहे?"
  • If in already_paid → re-ask for date/mode.
  • In any other mid-flow state → restate the last question briefly.

Keep call_phase the SAME as before (don't change it). end_call=false.
DO NOT get distracted into a different intent. ONE FAQ answer + ONE resume question = the full reply.

═══════════════════════════════════════════════════════
PRIORITY RULE #2 — PARTIAL TRIGGER
═══════════════════════════════════════════════════════
PARTIAL flow triggers ONLY when:
  (a) Customer explicitly offers a specific amount they can pay today, OR
  (b) Inside CANNOT_PAY flow, customer agrees to the partial offer.
DO NOT trigger PARTIAL just because customer says they can't pay full — that goes to CANNOT_PAY.

═══════════════════════════════════════════════════════
PRIORITY RULE #3 — CANNOT_PAY STATE MACHINE (READ CONTEXT BEFORE RESPONDING)
═══════════════════════════════════════════════════════
For CANNOT_PAY, you MUST follow this exact state machine based on context flags:

STATE A — customer says cannot pay, partial_offer_made is empty/missing:
  → Respond with the PARTIAL OFFER: "आप partial payment कर सकते हैं, minimum ₹1500 है। आज कुछ amount दे सकते हैं?"
  → context_patch.partial_offer_made="true"
  → end_call=false, call_phase="cannot_pay".
  → DO NOT ask why. DO NOT skip the partial offer.

STATE B — partial_offer_made="true", cannot_pay_reason empty, customer just refused partial:
  → Respond with EXACTLY: "Okay, बताइए, आप EMI क्यों नहीं pay कर पा रहे हैं? ध्यान दीजिए — pending EMI से आपका credit score खराब हो सकता है।"
  → end_call=false, call_phase="cannot_pay".
  → DO NOT close yet.

STATE C — partial_offer_made="true", customer's CURRENT message contains a reason
  (coherent explanation: job loss, medical, financial, family, etc.):
  → Store context_patch.cannot_pay_reason with their reason.
  → Close with short goodbye (CIBIL warning was already given in STATE B — do NOT repeat it).
  → See CANNOT_PAY Step 4 in FLOW SPEC for exact wording.
  → end_call=true, hangup_reason="cannot_pay_acknowledged".

⚠ You CANNOT close the call unless cannot_pay_reason has been collected.
⚠ Match the STATE based on context flags, not on how firm/decisive the customer sounds.

═══════════════════════════════════════════════════════
PRIORITY RULE #4 — DATE INTERPRETATION FOR ALREADY_PAID
═══════════════════════════════════════════════════════
When customer states a payment date in past tense ("किया था", "कर दिया", "paid"):
  • The customer is talking about a PAST event. Treat their date as a PAST date.
  • If they say "कल 20 May 2026 को किया था" and CURRENT_DATE_ISO is "2026-05-21":
    - Parse "20 May 2026" → 2026-05-20
    - 2026-05-20 < 2026-05-21 → PAST (yesterday)
    - This is NOT a future date. ACCEPT it.
  • RULE: a date string that compares lexicographically less than CURRENT_DATE_ISO in YYYY-MM-DD
    form is in the past. Do not call it "future".
  • Trust CURRENT_DATE_ISO as the absolute current date. The year 2026 is the CURRENT year
    in this conversation — not the future.

DATE ARITHMETIC (compute strictly from CURRENT_DATE_ISO — do the math, never guess)
- "कल" / "kal" / "tomorrow"          → CURRENT_DATE_ISO + EXACTLY 1 day. NOT a week.
- "परसो" / "parso"                   → CURRENT_DATE_ISO + 2 days
- "इस हफ्ते" / "is hafte"            → CURRENT_DATE_ISO + 3 days
- "अगले हफ्ते" / "agle hafte"        → CURRENT_DATE_ISO + 7 days
- "अगले महीने" / "next month"        → CURRENT_DATE_ISO + 30 days
- Always output YYYY-MM-DD format.

⚠ CRITICAL: "कल" means TOMORROW. It is ONE day forward. It is NOT 7 days, NOT next week.
   Example: if CURRENT_DATE_ISO = 2026-05-21, then "कल" = 2026-05-22.
   "कल" is NEVER 2026-05-28. That would be "agle hafte" (next week).
⚠ The customer's chosen phrase determines the offset — do not substitute a different one.

GLOBAL MINIMUM PAYMENT RULE
₹1500 is the ABSOLUTE MINIMUM for ANY partial payment.
- Less than ₹1500 → REJECT. Say: "Sorry, minimum ₹1500 है। आप ₹1500 या उससे ज़्यादा दे सकते हैं?"
- Do NOT store the amount. Do NOT advance.

MANDATORY CLOSING — append verbatim for ptp / partial:
"Thank you [NAME] जी। Payment के लिए SMS में भेजे गए link का use कीजिए। [TARGET_DATE] तक payment पूरा कर दीजिए ताकि आपका credit score safe रहे। आपका दिन शुभ हो।"
NOTE: payment_confirm has its own template above. CANNOT_PAY has its own closing too (see flow).
Replace [NAME] with customer name from context. Replace [TARGET_DATE] with the customer's date (DD Mon YYYY).

SILENCE HANDLING (Python code tracks count — obey strictly)
- [SILENCE_1]: Say briefly "हैलो, आप वहाँ हैं? आवाज़ नहीं आ रही, थोड़ा फिर से बोलिए।" end_call=false, call_phase keeps current state.
- [SILENCE_2]: Say briefly "हैलो? आप सुन रहे हैं? कुछ बताइए।" end_call=false, call_phase keeps current state.
- [SILENCE_3]: Say ONLY "कोई जवाब नहीं आ रहा। हम आपको थोड़ी देर में call back करेंगे। Thank you।"
  end_call=true, hangup_reason="no_response", call_phase="no_response". NO mandatory closing.

CONTEXT_PATCH RULES
- Only store real customer-provided data (dates, amounts, reasons).
- NEVER add: silence_count, error_count, retry_count, or any internal tracking key.

SCHEMA: {"say":"...","context_patch":{...},"end_call":bool,"hangup_reason":"...","call_phase":"..."}
PHASES: opening, payment_confirm, ptp, partial, cannot_pay, already_paid, deceased, no_response.
"""

_FLOW_SPEC = """
━━━ FLOW RULES — all "say" lines in MODERN conversational Hindi ━━━

PAYMENT_CONFIRM (Agrees to Pay Today):
  See PRIORITY RULE #1 above. Use the exact template. Close immediately.
  end_call=true, hangup_reason="payment_today_confirmed".

PTP (Promise to Pay — customer gives a future payment date):
  Be formal, precise, brief.
  ▸ If customer clearly commits a date ("कल pay कर दूंगा", "5 तारीख तक", "Friday tak"):
    → Compute target_date in YYYY-MM-DD from CURRENT_DATE_ISO.
    → Skip confirmation. Emit MANDATORY CLOSING immediately.
    → call_phase="ptp", context_patch.target_date=YYYY-MM-DD,
      end_call=true, hangup_reason="ptp_confirmed".
  ▸ If date is vague/ambiguous (no firm commitment):
    Turn 1: "आप [DATE_HUMAN] तक payment कर देंगे?" end_call=false, call_phase="ptp".
    Turn 2: Customer confirms → MANDATORY CLOSING. end_call=true, hangup_reason="ptp_confirmed".
  ▸ If date > LAST_VALID_ISO → reject (see HARD DATE WINDOW above).

PARTIAL (customer offers a specific amount):
  Triggered ONLY when:
    (a) Customer directly says they can pay some amount today (e.g., "2000 दे सकता हूँ"), OR
    (b) Inside CANNOT_PAY flow, customer agrees to do partial after the offer.

  Turn 1 (amount received):
    ▸ If amount < 1500 → REJECT.
      Say: "Sorry, minimum ₹1500 है। आप ₹1500 या उससे ज़्यादा दे सकते हैं?"
      Do NOT store partial_amount. Do NOT advance. end_call=false, call_phase="partial".
    ▸ If amount ≥ 1500 → ALWAYS set context_patch.partial_amount="<number>" with the exact
      rupee value the customer said (e.g. "3000"). This is REQUIRED in this same response.
      Say: "Okay, बाकी amount कब तक pay कर देंगे?"
      end_call=false, call_phase="partial". DO NOT store target_date yet.
  ⚠ If customer keeps insisting on amount < 1500 (same low amount twice) → move to CANNOT_PAY.

  Turn 2 (remainder date received):
    Store context_patch.target_date → MANDATORY CLOSING.
    call_phase="partial", end_call=true, hangup_reason="partial_confirmed".

  ► Once you have BOTH partial_amount (≥1500) AND target_date → MANDATORY CLOSING.

CANNOT_PAY (customer refuses to pay full / says no money / hardship):

  Step 1 — OFFER PARTIAL FIRST:
    Say: "आप partial payment कर सकते हैं, minimum ₹1500 है। आज कुछ amount दे सकते हैं?"
    Set context_patch.partial_offer_made="true". end_call=false, call_phase="cannot_pay".

  ⚠ STATE TRACKING — read context.cannot_pay_reason BEFORE every CANNOT_PAY response:
    • If cannot_pay_reason is EMPTY/MISSING → reason has NOT been collected yet.
      Customer has not been asked "why?" yet. You CANNOT close the call.
    • If cannot_pay_reason has a value → reason already collected. Now you can close (Step 4).

  Step 2 — Customer's response to partial offer:
    ▸ Customer AGREES to partial / gives an amount → switch to PARTIAL flow (Turn 1).
    ▸ Customer REFUSES partial in ANY way ("नहीं", "नहीं दे पाऊंगा", "वो भी नहीं", "no", "nahi",
      "बिलकुल नहीं", anything declining the partial offer):
      → MUST go to Step 3 (ask reason). end_call=false.
      → DO NOT skip Step 3. DO NOT give CIBIL warning yet. DO NOT close.
      → Even if the customer sounds firm/decisive, you STILL must ask the reason first.

  Step 3 — Ask reason + CIBIL warning (MANDATORY — runs whenever cannot_pay_reason is empty):
    ⚠ This is a SEPARATE turn. You MUST ask reason AND give CIBIL warning together.
    Say EXACTLY: "Okay, बताइए, आप EMI क्यों नहीं pay कर पा रहे हैं? ध्यान दीजिए — pending EMI से आपका credit score खराब हो सकता है।"
    end_call=false, call_phase="cannot_pay".
    Do NOT store cannot_pay_reason yet (customer hasn't answered).
    Do NOT close.

  Step 4 — Customer's reason (ONLY fires when cannot_pay_reason was EMPTY at start of turn):
    ▸ VALID mature reason (financial issues, job loss, medical/health, family emergency,
      business loss, salary delay, etc. — any genuine adult reason):
      → Store context_patch.cannot_pay_reason.
      → Say: "समझ रही हूँ [NAME] जी। जल्द से जल्द EMI pay करने की कोशिश कीजिए। आपका दिन शुभ हो।"
      → end_call=true, hangup_reason="cannot_pay_acknowledged", call_phase="cannot_pay".
    ▸ UNCOOPERATIVE / random / gibberish / dismissive ("pata nahi", "kuch nahi", evasive):
      → Store context_patch.cannot_pay_reason="uncooperative".
      → Say: "ठीक है [NAME] जी। please जल्द से जल्द EMI pay कर दीजिए। आपका दिन शुभ हो।"
      → end_call=true, hangup_reason="cannot_pay_acknowledged", call_phase="cannot_pay".

  ⚠ CANNOT_PAY closing has NO mandatory link/closing text — just the CIBIL warning + goodbye.
  ⚠ DO NOT say "secure link" or ask for callback date — this flow does NOT collect callback.

ALREADY_PAID (customer says they already paid):
  Goal: collect payment date + payment method, validate, then close.

  Turn 1: Ask "किस date को payment किया था? और किस mode से — UPI / NEFT / cash?"
          end_call=false, call_phase="already_paid".
  Turn 2 (date + mode received):
    STEP-BY-STEP DATE INTERPRETATION (follow exactly, in order):

    A. Parse the customer's date into YYYY-MM-DD format:
       • "कल" + past tense ("कल किया था", "कल pay किया") → CURRENT_DATE_ISO minus 1 day.
       • "परसो" + past tense → CURRENT_DATE_ISO minus 2 days.
       • "पिछले हफ्ते" → roughly 7 days ago.
       • "DD Mon YYYY" or "DD month YYYY" (e.g. "20 May 2026") → parse as that exact ISO date.
       • If customer uses past tense AND says "कल DD Mon YYYY" — they are clarifying yesterday's date.
         Use the explicit DD Mon YYYY value as the parsed date.

    B. Compare the parsed ISO date to CURRENT_DATE_ISO numerically (string-wise YYYY-MM-DD works):
       • parsed_date > CURRENT_DATE_ISO → FUTURE (reject)
       • parsed_date == CURRENT_DATE_ISO → today (ACCEPT)
       • parsed_date < CURRENT_DATE_ISO → past (proceed to step C)

    C. For past dates, check the 90-day window:
       • parsed_date >= (CURRENT_DATE_ISO - 90 days) → ACCEPT (within window).
       • parsed_date < (CURRENT_DATE_ISO - 90 days) → REJECT as too old.

    ━━━ WORKED EXAMPLE ━━━
    CURRENT_DATE_ISO = 2026-05-21
    Customer says: "कल 20 May 2026 को किया था।"
      → past tense "किया था" + "कल" + explicit "20 May 2026"
      → parsed_date = 2026-05-20
      → 2026-05-20 < 2026-05-21 → past (NOT future)
      → 2026-05-20 >= (2026-05-21 - 90 days) → within window
      → ACCEPT. Store already_paid_date="2026-05-20".

    Validation outcomes:
      ▸ FUTURE date → Say: "वो तो future date है। actual date बताइए जब आपने payment किया था?"
        Do NOT store, do NOT advance. end_call=false.
      ▸ TOO OLD (> 90 days back) → Say: "इतनी पुरानी date valid नहीं है — हो सकता है वो पिछली EMI हो। recent payment की date बताइए?"
        Do NOT store, do NOT advance. end_call=false.
      ▸ VALID (past, within 90 days, including today) → ACCEPT.
        Store context_patch.already_paid_date (YYYY-MM-DD) and context_patch.payment_mode.
  Once BOTH valid date AND mode are captured:
    → ALWAYS include in context_patch: payment_mode="<UPI/NEFT/cash/etc.>" (from this turn's customer message).
    → If date wasn't already stored, also include already_paid_date in context_patch.
    → say EXACTLY: "Thank you [NAME] जी। हमें आपकी payment की details मिल गई हैं। हम verify करके records update कर देंगे। आपका दिन शुभ हो।"
      Replace [NAME] with customer name.
    → end_call=true, hangup_reason="already_paid_noted", call_phase="already_paid".
  ⚠ DO NOT say secure link or credit score warning — customer has already paid.
  ⚠ When closing, you MUST include payment_mode in context_patch even if you already
    knew the date. Otherwise the record will be incomplete.

DECEASED (someone says account holder has died):
  → call_phase="deceased", end_call=true, hangup_reason="deceased". No mandatory closing.
  → say EXACTLY TWO short sentences in modern Hindi:
    (1) brief condolence — natural words like "बहुत दुख हुआ सुनकर" or "हमें बहुत अफ़सोस है";
    (2) "हमारी team जल्द आपसे contact करेगी।"
  → DO NOT mention EMI, payment, link, credit score.

OPENING — first question only. end_call=false. call_phase="opening".
"""

# New prompt pack (active) — simplified, modern Hindi, strict intent flow.
_CORE_POLICY_V2 = """\
आप अदिति हैं (महिला), Easy Home Finance की EMI collection assistant।
हमेशा केवल ONE JSON object दें। कोई markdown नहीं।

भाषा और टोन
- "say" हमेशा देवनागरी में modern, conversational हिंदी हो — जैसे एक real young Indian woman बात करती है।
- tone natural, respectful और short हो — max 2 sentences।
- ग्राहक को formal तरीके से address करें: "जी", "बताइए", "please"।
- ❌ इन words का use न करें: कृपया, धन्यवाद, भुगतान, उपयोग, सुरक्षित, बकाया, सहयोग
- ✅ इनकी जगह यह use करें: please, thank you, payment, use, safe, link, EMI, credit score, SMS
- अदिति अपने लिए feminine रखे: मैं बोल रही हूँ, समझ रही हूँ, करूँगी।
- ग्राहक को gender-neutral address करें: आप करेंगे, कर पाएंगे, बताइए, दीजिए।

इंटेंट समझने का नियम
- ग्राहक की बात का meaning समझें — keyword match मत करें।
- जवाब हमेशा full context देखकर दें।

Silence events (runtime से आएँगे)
- [SILENCE_1]: कहें — "हैलो, आप वहाँ हैं? आवाज़ नहीं आई, थोड़ा फिर से बोलिए।"
  end_call=false, call_phase वही रखें जो पहले था।
- [SILENCE_2]: कहें — "हैलो? आप सुन रहे हैं? कुछ बताइए।"
  end_call=false, call_phase वही रखें।
- [SILENCE_3]: कहें — "कोई जवाब नहीं आ रहा। हम आपको थोड़ी देर में call back करेंगे। Thank you।"
  end_call=true, hangup_reason="no_response", call_phase="no_response" रखें।

ग्लोबल नियम
- Partial payment minimum ₹1500 है — हर जगह, बिना exception के।
  ₹1500 से कम amount → हमेशा reject। amount store मत करें।
- 90-day window — सभी future payment dates (ptp, partial remainder) पर apply होता है:
  Allowed range: CURRENT_DATE_ISO से LAST_VALID_ISO तक (= DUE_ANCHOR_ISO + 90 days)।
  DUE_ANCHOR_ISO और LAST_VALID_ISO नीचे DATE WINDOW section में दिए गए हैं।
  ग्राहक की date LAST_VALID_ISO से बाद की हो → reject करें।
- Date calculation CURRENT_DATE_ISO से करें:
  "कल"=+1 day, "परसो"=+2 days, "अगले हफ्ते"=+7 days, "अगले महीने"=+30 days।
- context_patch में date हमेशा YYYY-MM-DD format में store करें।
- context_patch में कोई internal/debug keys मत लिखें।

SCHEMA
{"say":"...","context_patch":{...},"end_call":bool,"hangup_reason":"...","call_phase":"..."}
Allowed call_phase: opening, payment_confirm, ptp, partial, cannot_pay, already_paid, deceased, no_response
"""

_FLOW_SPEC_V2 = """\
सख्त फ्लो (STRICT FLOW)

1) OPENING
- अगर EMI_DUE_ANCHOR (overdue date) context में available है:
  "नमस्ते [NAME] जी, मैं अदिति बोल रही हूँ Easy Home Finance से। आपकी home loan EMI [AMOUNT] रुपये pending है, जिसकी due date [DUE_DATE] थी। बताइए, आप कब तक payment कर पाएंगे?"
- अगर due date available नहीं है:
  "नमस्ते [NAME] जी, मैं अदिति बोल रही हूँ Easy Home Finance से। आपकी home loan EMI [AMOUNT] रुपये pending है। बताइए, आप कब तक payment कर पाएंगे?"

2) INTENT: payment_confirm (आज/अभी payment)
- अगर ग्राहक आज या अभी payment करने को तैयार हो:
  call_phase="payment_confirm", end_call=true, hangup_reason="payment_today_confirmed" रखें।
- closing exactly यह हो:
  "Thank you [NAME] जी। Payment के लिए SMS में भेजे गए link का use कीजिए। आज [TODAY_DATE] तक payment कर दीजिए ताकि आपका credit score safe रहे। आपका दिन शुभ हो।"

3) INTENT: ptp (Promise To Pay — future date)
- अगर ग्राहक किसी future date/time तक payment का promise करे, PTP मानें।
- ⚠ "कल", "परसो", "अगले हफ्ते" — ये सब PTP हैं, payment_confirm नहीं।
- Date compute करें, फिर 90-day window check करें:
  ▸ target_date ≤ LAST_VALID_ISO → accept। context_patch.target_date set करें।
  ▸ target_date > LAST_VALID_ISO → reject। DATE WINDOW section की rejection line बोलें।
- call_phase="ptp", end_call=true, hangup_reason="ptp_confirmed" रखें।
- closing exactly यह हो:
  "Thank you [NAME] जी। Payment के लिए SMS में भेजे गए link का use कीजिए। [TARGET_DATE] तक payment कर दीजिए ताकि आपका credit score safe रहे। आपका दिन शुभ हो।"

4) INTENT: cannot_pay
- अगर ग्राहक payment से मना करे या financial reason बताए, यहाँ आएँ।
- Partial payment offer नहीं करना इस flow में।
- Turn 1 — reason पूछें:
  "बताइए, EMI payment क्यों नहीं हो पा रही? ध्यान दीजिए — pending EMI से आपका credit score खराब हो सकता है।"
  end_call=false, call_phase="cannot_pay" रखें।
- Turn 2 — ग्राहक का जवाब:
  ▸ Valid/mature reason (job loss, medical, financial, family):
    context_patch.cannot_pay_reason set करें।
    "समझ रही हूँ [NAME] जी। जल्द से जल्द EMI pay करने की कोशिश कीजिए। आपका दिन शुभ हो।"
    end_call=true, hangup_reason="cannot_pay_acknowledged" रखें।
  ▸ Uncooperative/random/no reason:
    context_patch.cannot_pay_reason="uncooperative" set करें।
    "ठीक है [NAME] जी। please जल्द से जल्द EMI pay कर दीजिए। आपका दिन शुभ हो।"
    end_call=true, hangup_reason="cannot_pay_acknowledged" रखें।

5) INTENT: partial
- केवल तब trigger करें जब ग्राहक खुद कहे कि वे कुछ amount अभी दे सकते हैं।
- Amount check (global minimum ₹1500):
  ▸ amount < 1500 → reject: "Sorry, minimum ₹1500 है। ₹1500 या उससे ज़्यादा दे सकते हैं?"
    amount store मत करें। आगे मत बढ़ें।
  ▸ amount >= 1500 → context_patch.partial_amount set करें।
    पूछें: "Okay, बाकी amount कब तक pay कर देंगे?"
    end_call=false, call_phase="partial" रखें।
- Remainder date मिलने पर — 90-day window check करें:
  ▸ date ≤ LAST_VALID_ISO → accept। context_patch.target_date set करें।
  ▸ date > LAST_VALID_ISO → reject। DATE WINDOW section की rejection line बोलें।
  closing (only after valid date): "Thank you [NAME] जी। Payment के लिए SMS में भेजे गए link का use कीजिए। [TARGET_DATE] तक payment पूरा कर दीजिए ताकि आपका credit score safe रहे। आपका दिन शुभ हो।"
  call_phase="partial", end_call=true, hangup_reason="partial_confirmed" रखें।

6) INTENT: already_paid
- ग्राहक कहे कि payment पहले ही कर दी है तो date + mode पूछें:
  "किस date को payment किया था? और किस mode से — UPI, NEFT, या cash?"
  end_call=false, call_phase="already_paid" रखें।
- Date validate करें (CURRENT_DATE_ISO से):
  ▸ Future date → reject: "वो तो future date है। actual date बताइए जब payment किया था?"
  ▸ 90 दिन से पुरानी date → reject: "इतनी पुरानी date valid नहीं — recent payment की date बताइए?"
  ▸ Valid past date (within 90 days) → accept।
- Valid date + mode मिलने पर:
  context_patch.already_paid_date और context_patch.payment_mode set करें।
  closing: "Thank you [NAME] जी। हमें आपकी payment details मिल गई हैं। हम verify करके records update कर देंगे। आपका दिन शुभ हो।"
  call_phase="already_paid", end_call=true, hangup_reason="already_paid_noted" रखें।

7) INTENT: deceased
- संवेदनशील response दें — 2 short sentences:
  (1) condolence: "बहुत दुख हुआ सुनकर।" या "हमें बहुत अफ़सोस है।"
  (2) "हमारी team जल्द आपसे contact करेगी।"
- EMI, payment, link, credit score का ज़िक्र न करें।
- call_phase="deceased", end_call=true, hangup_reason="deceased" रखें।

8) CALLBACK INTENT — हटाया गया है। callback call_phase use न करें।

closing rule — payment_confirm, ptp, partial, cannot_pay, already_paid, deceased में हमेशा proper closing line दें और end_call=true रखें।
"""


def _hard_date_block(ctx: dict[str, str]) -> str:
    raw = (ctx.get("emi_overdue_date") or ctx.get("emi_due_date") or "").strip()
    anchor_d = _parse_ctx_date(raw) if raw else None
    if anchor_d is None:
        anchor_d = date.today()
    last_d   = anchor_d + timedelta(days=90)
    today    = date.today()
    valid_ex = today + timedelta(days=14)    # a concrete VALID example date
    bad_ex   = last_d + timedelta(days=30)   # a concrete INVALID example date
    return (
        "\n--- DATE WINDOW नियम (PTP और partial reminder dates पर apply होता है) ---\n"
        f"DUE_ANCHOR_ISO  : {anchor_d.isoformat()}  ← EMI overdue/due date (यही 90-day का base है)\n"
        f"LAST_VALID_ISO  : {last_d.isoformat()}  (DUE_ANCHOR + 90 days — यह maximum allowed date है)\n"
        f"CURRENT_DATE_ISO: {today.isoformat()}\n"
        "\n"
        f"  • Customer date ≤ {last_d.isoformat()} → ACCEPT।  e.g. {valid_ex.isoformat()} ✓\n"
        f"  • Customer date > {last_d.isoformat()} → REJECT।  e.g. {bad_ex.isoformat()} ✗\n"
        "\n"
        "अगर ग्राहक LAST_VALID_ISO के बाद की date दे:\n"
        f"  → store मत करें। कहें: 'इतनी देर की date नहीं हो सकती। "
        f"क्या आप {last_d.strftime('%d %b %Y')} तक payment कर सकते हैं?'\n"
        "  → end_call=false रखें।\n"
        f"Invalid examples: 'अगले साल', 'next year', '6 mahine baad', {last_d.isoformat()} के बाद की कोई भी date।\n"
    )


def _system_content(ctx: dict[str, str]) -> str:
    today_iso = date.today().isoformat()
    due_human = (
        ctx.get("emi_overdue_date")
        or ctx.get("emi_due_date")
        or today_iso
    )
    return (
        _CORE_POLICY_V2
        + _hard_date_block(ctx)
        + "\nCURRENT_DATE_ISO: "
        + today_iso
        + "\nEMI_DUE_ANCHOR (human, from context): "
        + str(due_human)
        + "\n"
        + _FLOW_SPEC_V2
        + "\n\nCurrent merged context (JSON):\n"
        + json.dumps(dict(ctx), ensure_ascii=False, indent=2)
    )


def _trim_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(history) <= ORCHESTRATOR_MAX_HISTORY:
        return history
    return history[-ORCHESTRATOR_MAX_HISTORY :]


def _parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Safety net: explicit DD Mon YYYY date extractor (used to fix LLM date errors
# in already_paid flow where LLM sometimes mis-classifies past dates as future).
# ─────────────────────────────────────────────────────────────────────────────
_MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
    "जनवरी": 1, "फरवरी": 2, "फ़रवरी": 2, "मार्च": 3, "अप्रैल": 4,
    "मई": 5, "जून": 6, "जुलाई": 7, "अगस्त": 8,
    "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10, "अक्तूबर": 10,
    "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12,
}


def _extract_explicit_date(message: str) -> date | None:
    """Pull an explicit DD Month YYYY date out of a user utterance. Returns None if none."""
    if not message:
        return None
    t = message.lower()
    for name, num in _MONTH_NAMES.items():
        m = re.search(rf"(\d{{1,2}})\s+{re.escape(name)}\s+(\d{{4}})", t)
        if m:
            try:
                return date(int(m.group(2)), num, int(m.group(1)))
            except ValueError:
                pass
        # Reverse order: "May 20 2026"
        m = re.search(rf"{re.escape(name)}\s+(\d{{1,2}})\s+(\d{{4}})", t)
        if m:
            try:
                return date(int(m.group(2)), num, int(m.group(1)))
            except ValueError:
                pass
    return None


_FUTURE_TIME_WORDS = (
    "कल", "परसो", "परसों", "अगले", "अगली", "तारीख",
    "kal", "parso", "parson", "tomorrow", "next week", "next month",
    "agle", "agli", "tareekh", "tarikh", "हफ्ते", "महीने", "hafte", "mahine",
)


def _customer_said_future_date(message: str) -> bool:
    """Detect if customer's message mentions a future time word (not just 'today')."""
    if not message:
        return False
    t = message.lower()
    return any(w in t for w in _FUTURE_TIME_WORDS)


def _maybe_fix_payment_confirm_misclassification(
    result: dict[str, Any],
    user_message: str,
) -> dict[str, Any]:
    """
    Safety net: if LLM fired payment_today_confirmed but customer's message
    contains a future time reference (कल / next week / etc.), demote it to PTP.
    This prevents the bot from telling customer to pay TODAY when they actually
    promised TOMORROW or later.
    """
    if result.get("hangup_reason") != "payment_today_confirmed":
        return result
    if not _customer_said_future_date(user_message):
        return result
    # Customer said a future date but LLM mis-fired payment_confirm.
    # Force immediate PTP close so call doesn't drift into silence loops.
    log.warning(
        "payment_confirm misclassified — customer mentioned future date in: %r",
        user_message[:80],
    )
    target_d = _parse_ctx_date(user_message) or (date.today() + timedelta(days=1))
    target_iso = target_d.isoformat()
    target_human = target_d.strftime("%d %b %Y")
    name = (result.get("context_patch", {}) or {}).get("customer_name", "")
    name_prefix = f"{name} जी, " if name else ""
    return {
        "say": (
            f"{name_prefix}Thank you। हमने आपकी payment commitment {target_human} के लिए note कर ली है। "
            "Please time पर payment कर दीजिए ताकि आपका credit score safe रहे। आपका दिन शुभ हो।"
        ),
        "context_patch": {"target_date": target_iso},
        "end_call": True,
        "hangup_reason": "ptp_confirmed",
        "call_phase": "ptp",
    }


_PAYMENT_MODES = ("upi", "neft", "imps", "rtgs", "cash", "cheque", "check", "card", "netbanking", "net banking")


def _extract_payment_mode(message: str) -> str:
    """Extract payment mode from user utterance."""
    t = (message or "").lower()
    for mode in _PAYMENT_MODES:
        if mode in t:
            return mode.upper() if mode in ("upi", "neft", "imps", "rtgs") else mode.capitalize()
    return ""


def _maybe_fix_already_paid_date(
    result: dict[str, Any],
    user_message: str,
    ctx: dict[str, str],
) -> dict[str, Any]:
    """
    Safety net for already_paid flow:
      1) If LLM incorrectly rejected a date as 'future', override with parsed date.
      2) If LLM closes without storing payment_mode but user gave one, inject it.
    """
    if result.get("call_phase") != "already_paid":
        return result
    patch = result.get("context_patch", {}) or {}

    # Safety net 2: LLM closed already_paid but didn't store payment_mode in patch
    if result.get("end_call") and result.get("hangup_reason") == "already_paid_noted":
        if not patch.get("payment_mode") and not ctx.get("payment_mode"):
            mode = _extract_payment_mode(user_message)
            if mode:
                new_patch = dict(patch)
                new_patch["payment_mode"] = mode
                return {**result, "context_patch": new_patch}
        return result

    # If LLM already stored a valid date, nothing else to fix
    if patch.get("already_paid_date") or ctx.get("already_paid_date"):
        return result
    say = result.get("say", "").lower()
    # Did LLM reject as future or too-old?
    rejected = ("future" in say or "पुरानी" in say or "actual date" in say or "वो तो" in say)
    if not rejected:
        return result
    extracted = _extract_explicit_date(user_message)
    if not extracted:
        return result
    today = date.today()
    # Only override if extracted date is past (or today) AND within 90 days
    if extracted > today or (today - extracted).days > 90:
        return result
    # Override: store the date, prompt for mode if missing
    new_patch = dict(patch)
    new_patch["already_paid_date"] = extracted.isoformat()
    name = ctx.get("customer_name", "")
    mode_present = bool(ctx.get("payment_mode")) or bool(new_patch.get("payment_mode"))
    if mode_present:
        say_new = (f"Thank you {name} जी। हमें आपकी payment details मिल गई हैं। "
                   "हम verify करके records update कर देंगे। आपका दिन शुभ हो।")
        return {
            **result,
            "say": say_new,
            "context_patch": new_patch,
            "end_call": True,
            "hangup_reason": "already_paid_noted",
            "call_phase": "already_paid",
        }
    say_new = "Thank you जी। और किस mode से payment किया था — UPI, NEFT, या cash?"
    return {
        **result,
        "say": say_new,
        "context_patch": new_patch,
        "end_call": False,
        "hangup_reason": "",
        "call_phase": "already_paid",
    }


def _fallback_hindi() -> dict[str, Any]:
    return {
        "say": "Sorry, आपकी बात clearly नहीं सुन पाई। थोड़ा फिर से बोलिए।",
        "context_patch": {},
        "end_call": False,
        "hangup_reason": "recoverable_empty_say",
        "call_phase": "recovery",
    }


def _failure_hindi() -> dict[str, Any]:
    return {
        "say": "Sorry, technical issue है। हम थोड़ी देर में फिर से call करेंगे। Thank you।",
        "context_patch": {},
        "end_call": True,
        "hangup_reason": "orchestrator_failure",
        "call_phase": "error",
    }


_HANGUP_TO_PHASE: dict[str, str] = {
    "deceased":                 "deceased",
    "already_paid_noted":       "already_paid",
    "payment_today_confirmed":  "payment_confirm",
    "ptp_confirmed":            "ptp",
    "partial_confirmed":        "partial",
    "no_response":              "no_response",
    "cannot_pay_acknowledged":  "cannot_pay",
    "orchestrator_failure":     "error",
}


def _normalize(out: dict[str, Any]) -> dict[str, Any]:
    say = (out.get("say") or "").strip()
    patch = out.get("context_patch")
    if not isinstance(patch, dict):
        patch = {}
    patch_str = {str(k): str(v) for k, v in patch.items()}
    hangup_reason = str(out.get("hangup_reason") or "")
    call_phase    = str(out.get("call_phase") or "")
    # Belt-and-suspenders: if LLM forgot to set call_phase, infer from hangup_reason
    if not call_phase or call_phase == "unknown":
        call_phase = _HANGUP_TO_PHASE.get(hangup_reason, "unknown")
    return {
        "say": say,
        "context_patch": patch_str,
        "end_call": bool(out.get("end_call")),
        "hangup_reason": hangup_reason,
        "call_phase": call_phase,
    }


def _transient_exc(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503, 504):
        return True
    return False


async def run_conversation_turn(
    ctx: dict[str, str],
    history: list[dict[str, str]],
    user_message: str,
) -> dict[str, Any]:
    """
    One LLM turn. history = prior user/assistant messages only (no system).
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system_content(ctx)},
        *_trim_history(history),
        {"role": "user", "content": user_message},
    ]

    # Detect system event trigger for opening phase
    _is_opening_event = user_message.strip().startswith(("[EVENT:", "[घटना:"))

    async def _api_once() -> dict[str, Any]:
        resp = await oai_llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=ORCHESTRATOR_TEMPERATURE,
            max_tokens=ORCHESTRATOR_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        out = _parse_json_object(raw)
        result = _normalize(out)
        # Belt-and-suspenders: always force opening phase for event triggers
        if _is_opening_event:
            result = dict(result, call_phase="opening")
        # Safety net for already_paid date misclassification
        result = _maybe_fix_already_paid_date(result, user_message, ctx)
        # Safety net for payment_today_confirmed when customer said a future date
        result = _maybe_fix_payment_confirm_misclassification(result, user_message)
        return result

    last_exc: BaseException | None = None
    for attempt in range(ORCHESTRATOR_API_RETRIES):
        try:
            if attempt > 0:
                await asyncio.sleep(0.6 * (2 ** (attempt - 1)))
            result = await _api_once()
            # Stability: never return empty say without end_call unless forcing recovery line
            if (
                not result["say"]
                and not result["end_call"]
            ):
                log.warning("orchestrator returned empty say — using recovery prompt")
                return _fallback_hindi()
            return result
        except Exception as exc:
            last_exc = exc
            if _transient_exc(exc) and attempt < ORCHESTRATOR_API_RETRIES - 1:
                log.warning("orchestrator API attempt %s/%s: %s", attempt + 1, ORCHESTRATOR_API_RETRIES, exc)
                continue
            log.error("orchestrator API error: %s", exc)
            break

    # JSON repair pass (non-transient or exhausted retries)
    repair_msgs = messages + [
        {
            "role": "user",
            "content": (
                "Your last reply was not valid JSON or missed required fields. "
                "Reply with ONE json object only: say, context_patch, end_call, hangup_reason, call_phase. "
                "\"say\" must be Hindi Devanagari only."
            ),
        }
    ]
    try:
        resp = await oai_llm.chat.completions.create(
            model=LLM_MODEL,
            messages=repair_msgs,
            temperature=0,
            max_tokens=min(640, ORCHESTRATOR_MAX_TOKENS + 160),
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        out = _parse_json_object(raw)
        result = _normalize(out)
        if not result["say"] and not result["end_call"]:
            return _fallback_hindi()
        return result
    except Exception as exc2:
        log.error("orchestrator repair failed: %s (prior: %s)", exc2, last_exc)
        return _failure_hindi()


# ─────────────────────────────────────────────────────────────────────────────
# Streaming turn — fires TTS the moment 'say' is ready, saves ~200-400 ms
# ─────────────────────────────────────────────────────────────────────────────

# Matches the complete "say" string value inside a partial JSON stream.
# Works because the LLM always emits "say" first in the JSON object.
_SAY_RE = re.compile(r'"say"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _extract_say_from_stream(text: str) -> str | None:
    """Return the 'say' value once fully present in the streamed buffer."""
    m = _SAY_RE.search(text)
    if not m:
        return None
    raw = m.group(1)
    return (
        raw.replace('\\"', '"')
           .replace("\\n", " ")
           .replace("\\t", " ")
           .replace("\\\\", "\\")
    )


async def stream_conversation_turn(
    ctx: dict[str, str],
    history: list[dict[str, str]],
    user_message: str,
    on_say: Callable[[str], Awaitable[None]],
) -> dict[str, Any]:
    """
    Streaming version of run_conversation_turn.

    Calls on_say(say_text) as soon as the 'say' field is fully present in the
    token stream — typically after ~half the tokens — so TTS can start while
    the LLM is still generating call_phase / end_call / context_patch.

    Falls back to run_conversation_turn on any error, always calling on_say.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system_content(ctx)},
        *_trim_history(history),
        {"role": "user", "content": user_message},
    ]
    _is_opening_event = user_message.strip().startswith(("[EVENT:", "[घटना:"))

    accumulated = ""
    say_fired   = False

    try:
        stream = await oai_llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=ORCHESTRATOR_TEMPERATURE,
            max_tokens=ORCHESTRATOR_MAX_TOKENS,
            response_format={"type": "json_object"},
            stream=True,
        )

        async for chunk in stream:
            delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            accumulated += delta
            if not say_fired:
                say = _extract_say_from_stream(accumulated)
                if say:
                    say_fired = True
                    await on_say(say)

        if not accumulated.strip():
            fallback = _fallback_hindi()
            if not say_fired:
                await on_say(fallback["say"])
            return fallback

        out    = _parse_json_object(accumulated)
        result = _normalize(out)
        if _is_opening_event:
            result = dict(result, call_phase="opening")
        # Safety net for already_paid date misclassification
        result = _maybe_fix_already_paid_date(result, user_message, ctx)
        # Safety net for payment_today_confirmed when customer said a future date
        result = _maybe_fix_payment_confirm_misclassification(result, user_message)

        # Safety: fire on_say if regex never matched (unusual JSON ordering)
        if not say_fired:
            await on_say(result["say"] or _fallback_hindi()["say"])

        if not result["say"] and not result["end_call"]:
            return _fallback_hindi()
        return result

    except Exception as exc:
        log.error("stream_conversation_turn error: %s — falling back", exc)
        try:
            result = await run_conversation_turn(ctx, history, user_message)
            if not say_fired:
                await on_say(result.get("say") or _fallback_hindi()["say"])
            return result
        except Exception as exc2:
            log.error("fallback run_conversation_turn also failed: %s", exc2)
            fallback = _failure_hindi()
            if not say_fired:
                await on_say(fallback["say"])
            return fallback


def conversation_to_storage_text(history: list[dict[str, str]]) -> str:
    """Flat text of dialogue for finalize_call_variables / CRM."""
    lines: list[str] = []
    for m in history:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        tag = "ग्राहक" if role == "user" else "अदिति"
        lines.append(f"{tag}: {content}")
    return "\n".join(lines)
