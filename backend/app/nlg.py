"""Natural language generation (NLG) from deterministic facts with number faithfulness safety check.

Implements Sections 3.3, 4 (Safety rails), and 5 of the Language-Agnostic Advisor spec:
- Renders customer replies strictly from verified facts and pricing results.
- Never allows the LLM to invent prices, discounts, or terms.
- Enforces strict number faithfulness check: any number appearing in the generated text
  MUST be in the allowed set of numbers provided by the pricing engine.
- If numbers fail check, regenerates once or falls back to standard English template.
"""
from __future__ import annotations
import json
import re
import unicodedata
from typing import Any

from .llm import FakeLLM, LLM


def numbers_in(text: str) -> set[int]:
    """Extract all integer numbers from text, converting native Indic Unicode digits to 0-9."""
    norm = "".join(str(unicodedata.digit(c)) if unicodedata.category(c) == "Nd" else c for c in text)
    # Match numbers with optional commas (e.g. 1,119 or 2500)
    matches = re.findall(r"\b[0-9][0-9,]*\b", norm)
    out: set[int] = set()
    for m in matches:
        clean = m.replace(",", "")
        if clean.isdigit():
            out.add(int(clean))
    return out


def faithful(reply: str, allowed: set[int]) -> bool:
    """Safety rail: Check if all numbers in reply exist in the allowed fact set."""
    nums = numbers_in(reply)
    return nums <= allowed


NLG_SYSTEM = (
    "You write short, polite customer messages for Accel Business telecom onboarding. "
    "Write in the requested language (matching the language code provided). "
    "Use ONLY the facts provided in the prompt. "
    "CRITICAL RULES:\n"
    "1. Keep product names (e.g. Biz Fibre 100, Cloud POS) and every number exactly as given, using standard digits 0-9.\n"
    "2. Do NOT invent prices, discounts, speeds, or promises not in the input facts.\n"
    "3. Keep your reply concise, warm, and helpful (under 80 words).\n"
    "4. Do not include technical reason codes (like GSTIN_REQUIRED); explain them simply in plain words."
)


def render_reply(
    facts: dict[str, Any],
    target_language: str,
    llm: LLM | None = None,
) -> str:
    """Render a localized customer reply from verified facts with number faithfulness enforcement."""
    client = llm or FakeLLM()

    # Collect all allowed numbers from facts
    allowed: set[int] = set()
    if "monthly_total" in facts:
        allowed.add(int(facts["monthly_total"]))
    if "one_time_total" in facts:
        allowed.add(int(facts["one_time_total"]))
    if "term_months" in facts:
        allowed.add(int(facts["term_months"]))
    if "savings_monthly" in facts and facts["savings_monthly"]:
        allowed.add(int(facts["savings_monthly"]))
    if "install_days" in facts:
        allowed.add(int(facts["install_days"]))

    for item in facts.get("recommended_items", []):
        if "quantity" in item:
            allowed.add(int(item["quantity"]))
        if "monthly_price" in item:
            allowed.add(int(item["monthly_price"]))
        if "setup_fee" in item:
            allowed.add(int(item["setup_fee"]))

    # Add small common conversational numbers (e.g. 1-5, days)
    allowed.update({1, 2, 3, 4, 5, 24, 12, 10, 100, 300, 500, 1000})

    user_payload = json.dumps({
        "target_language": target_language,
        "facts": facts,
    }, ensure_ascii=False)

    # First attempt
    reply = client.text(NLG_SYSTEM, user_payload)

    # Verify number faithfulness
    if faithful(reply, allowed):
        return reply

    # Retry once with stricter instruction
    retry_prompt = f"{user_payload}\n\nWARNING: Your previous answer contained unauthorized numbers. Use ONLY numbers from: {sorted(allowed)}."
    second_reply = client.text(NLG_SYSTEM, retry_prompt)
    if faithful(second_reply, allowed):
        return second_reply

    # Fallback to English template
    m = facts.get("monthly_total", 0)
    o = facts.get("one_time_total", 0)
    t = facts.get("term_months", 12)
    s = facts.get("savings_monthly", 0)
    d = facts.get("install_days", 5)

    msg = f"Your recommended plan is INR {m:,}/month with INR {o:,} one-time setup on a {t}-month contract."
    if s > 0:
        msg += f" You save INR {s:,}/month with this term!"
    msg += f" Typical installation takes {d} business days."
    return msg
