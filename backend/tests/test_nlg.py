import pytest
from backend.app.nlg import numbers_in, faithful, render_reply
from backend.app.llm import FakeLLM


def test_numbers_in_handles_latin_and_indic_digits():
    # Standard Arabic/Latin digits
    assert numbers_in("Plan price is 1,119 per month for 12 months") == {1119, 12}
    # Devanagari digits: १,११९ (1119) and १२ (12)
    assert numbers_in("आपका प्लान १,११९ रुपये प्रति माह है १२ महीने के लिए") == {1119, 12}


def test_wrong_price_in_translation_is_rejected():
    allowed = {1119, 3, 12}
    # Faithful reply
    assert faithful("Aapka plan 1,119 rupaye mahine ka hai, 12 mahine ke liye", allowed)
    # Unfaithful reply hallucinating 1,199 instead of 1,119
    assert not faithful("Aapka plan 1,199 rupaye mahine ka hai", allowed)


def test_render_reply_falls_back_if_unfaithful():
    # Fake LLM that returns hallucinated price 9999
    fake = FakeLLM(text_replies=["Wrong price 9999 per month", "Still wrong 8888"])
    facts = {
        "monthly_total": 1119,
        "one_time_total": 0,
        "term_months": 12,
        "savings_monthly": 0,
        "install_days": 5,
    }
    reply = render_reply(facts, target_language="hi", llm=fake)
    # Must fallback to safe English template with exact fact numbers
    assert "1,119" in reply
    assert "9999" not in reply
