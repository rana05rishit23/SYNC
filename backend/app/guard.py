"""Grounding guards for the advisor (ADR-014): the LLM may phrase facts, it may never change them.

Three rules, all enforced in code rather than in the prompt:
1. Every number in the text the customer reads must come from a tool result, the catalogue, or the customer's
   own messages. Otherwise the reply is retried once and then replaced by a deterministic summary.
2. Slot values the LLM proposes (employees, sites, city tier, term, quantities) are bounded; bad values fall back
   to what the session already knows.
3. GST status is never taken from the LLM. Only a registry check sets it.
"""
from __future__ import annotations

import json
from typing import Any

from .nlg import numbers_in

SMALL_NUMBERS = frozenset(range(0, 6))      # "1-2 questions", "3 minutes": harmless, prices are never this small
MAX_QTY, MAX_EMPLOYEES, MAX_SITES = 500, 500, 50


def bounded_int(value: Any, default: int, lo: int, hi: int) -> int:
    """int(value) if it parses and lies in [lo, hi]; otherwise `default`. Never raises."""
    if isinstance(value, bool):
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if lo <= n <= hi else default


def allowed_numbers(tool_results: list[dict[str, Any]], catalogue: dict, user_texts: list[str],
                    context_numbers: set[int]) -> set[int]:
    nums = set(SMALL_NUMBERS) | set(context_numbers)
    for r in tool_results:
        nums |= numbers_in(json.dumps(r, ensure_ascii=False, default=str))
    for p in catalogue.values():
        nums |= numbers_in(f"{p.name} {p.description}")        # e.g. 'Biz Fibre 100'
    for t in user_texts:
        nums |= numbers_in(t)                                  # figures the customer stated
    return nums


def ungrounded_numbers(text: str, allowed: set[int]) -> set[int]:
    return numbers_in(text) - allowed


def fallback_message(q12: Any, q24: Any) -> str:
    """Deterministic English summary built only from the quote. Used when the LLM text fails the gate."""
    if q12 is None:
        return ("Sorry, I could not prepare that reliably. Please tell me your business type and city, "
                "and I will fetch your plan from our price list.")
    lines = [f"- {l['name']} x {l['quantity']}: \u20b9{l['monthly']:,}/month" for l in q12.line_items]
    out = ["Here is your plan, with prices straight from our price list:", *lines,
           f"12-month term: \u20b9{q12.monthly_total:,}/month + \u20b9{q12.one_time_total:,} one-time setup."]
    if q24 is not None:
        out.append(f"24-month term: \u20b9{q24.monthly_total:,}/month.")
    out.append("Choose a term or add-ons below, or continue to KYC.")
    return "\n".join(out)
