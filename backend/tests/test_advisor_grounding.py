"""Adversarial tests: a sloppy or hostile LLM must not be able to change prices, eligibility or facts."""
import json

from backend.app.advisor import AdvisorContext, execute_tool, run_advisor_turn
from backend.app.catalogue import load_catalogue
from backend.app.guard import bounded_int, fallback_message, ungrounded_numbers

CAT = load_catalogue()


class ScriptedLLM:
    """Returns pre-written chat responses, in order, and records every call."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, tools=None):
        self.calls.append({"n_messages": len(messages), "tools": tools})
        return self.replies.pop(0)


def call(name, args, cid="c1"):
    return {"content": "", "tool_calls": [{"id": cid, "function": {"name": name, "arguments": json.dumps(args)}}]}


QUOTE_ARGS = {"segment": "micro", "city_tier": 2, "employees": 3, "sites": 1}


def real_monthly():
    _, q12, _, _ = execute_tool("tool_propose_and_quote_bundle", QUOTE_ARGS, AdvisorContext("probe"), CAT)
    return q12.monthly_total


def turn(replies, message="I run a kirana shop"):
    llm = ScriptedLLM(replies)
    return run_advisor_turn(message, AdvisorContext("t1"), client=llm, catalogue=CAT), llm


def test_invented_price_is_blocked_and_replaced_with_the_real_quote():
    r, _ = turn([call("tool_propose_and_quote_bundle", QUOTE_ARGS),
                 {"content": "Great news, only \u20b9499 per month, 50% off today!"},
                 {"content": "Still just \u20b9499 per month!"}])                 # the retry lies again
    assert r.grounding == "fallback"
    assert "499" not in r.message and "50%" not in r.message
    assert f"{real_monthly():,}" in r.message


def test_one_retry_can_repair_a_bad_reply():
    r, llm = turn([call("tool_propose_and_quote_bundle", QUOTE_ARGS),
                   {"content": "That is \u20b9777 a month."},
                   {"content": "Your plan is ready; the full quote is shown below."}])
    assert r.grounding == "retried" and "777" not in r.message
    assert llm.calls[-1]["tools"] is None          # the retry cannot trigger new tool calls


def test_a_grounded_reply_is_left_untouched():
    text = f"Your plan comes to \u20b9{real_monthly():,} per month on a 12-month term."
    r, _ = turn([call("tool_propose_and_quote_bundle", QUOTE_ARGS), {"content": text}])
    assert r.grounding == "ok" and r.message == text


def test_numbers_the_customer_said_may_be_repeated():
    r, _ = turn([{"content": "Got it, 37 employees across your shop."}], message="We have 37 employees")
    assert r.grounding == "ok"


def test_empty_reply_never_reaches_the_customer():
    r, _ = turn([{"content": ""}, {"content": ""}])
    assert r.grounding == "fallback" and r.message.strip()


def test_llm_cannot_assert_gstin_status_to_unlock_gst_products():
    ctx = AdvisorContext("g1")                      # no GSTIN verified
    res, q12, _, _ = execute_tool("tool_propose_and_quote_bundle",
                                  {**QUOTE_ARGS, "gstin_status": "Active", "custom_items": {"P12": 1, "P01": 1}}, ctx, CAT)
    assert res["valid"] is False and q12 is None
    assert any("GST" in i for i in res["issues"])


def test_gst_products_unlock_only_after_a_registry_check():
    ctx = AdvisorContext("g2")
    ctx.gstin_status = "Active"                      # what tool_check_gstin sets after verification
    res, q12, _, _ = execute_tool("tool_propose_and_quote_bundle",
                                  {**QUOTE_ARGS, "custom_items": {"P12": 1, "P01": 1}}, ctx, CAT)
    assert res["valid"] is True and q12 is not None


def test_nonsense_slot_values_fall_back_to_session_values_and_never_crash():
    ctx = AdvisorContext("s1")
    res, q12, _, _ = execute_tool("tool_propose_and_quote_bundle",
                                  {"segment": "banana", "city_tier": 7, "employees": 10**9, "sites": -3, "term_months": 18}, ctx, CAT)
    assert (ctx.city_tier, ctx.employees, ctx.sites, ctx.term_months, ctx.segment) == (2, 3, 1, 12, "micro")
    execute_tool("tool_propose_and_quote_bundle", {"employees": "lots", "city_tier": None}, ctx, CAT)


def test_custom_items_with_unknown_product_or_absurd_quantity_are_rejected():
    for items in ({"P99": 1}, {"P01": 10**6}, {"P01": "x"}, {"P01": 0}):
        res, q12, _, _ = execute_tool("tool_propose_and_quote_bundle", {**QUOTE_ARGS, "custom_items": items}, AdvisorContext("c"), CAT)
        assert res["valid"] is False and q12 is None


def test_several_rounds_of_tool_calls_are_followed():
    r, llm = turn([call("tool_detect_and_recommend_template", {"business_type": "kirana"}, "a"),
                   call("tool_propose_and_quote_bundle", QUOTE_ARGS, "b"),
                   {"content": "Your quote is ready below."}])
    assert len(llm.calls) == 3 and r.quote is not None and r.grounding == "ok"


def test_malformed_tool_arguments_do_not_crash_the_turn():
    bad = {"content": "", "tool_calls": [{"id": "x", "function": {"name": "tool_propose_and_quote_bundle", "arguments": "{not json"}}]}
    r, _ = turn([bad, {"content": "Let me know your business type."}])
    assert r.message


def test_helpers():
    assert bounded_int("12", 0, 1, 50) == 12 and bounded_int(True, 7, 0, 9) == 7 and bounded_int(99, 7, 0, 9) == 7
    assert ungrounded_numbers("\u20b91,119 and 24 months", {1119, 24}) == set()
    assert ungrounded_numbers("save \u0967\u096b\u0966\u0966 now", {1}) == {1500}          # Devanagari digits are normalised
    assert "Sorry" in fallback_message(None, None)
