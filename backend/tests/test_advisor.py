import pytest
from backend.app.advisor import (
    AdvisorContext,
    MockAdvisorLLMClient,
    execute_tool,
    run_advisor_turn,
)
from backend.app.catalogue import load_catalogue
from backend.app.pricing import build_quote

CAT = load_catalogue()


def test_advisor_greeting_turn():
    ctx = AdvisorContext(session_id="test-1")
    result = run_advisor_turn("Hello, I want internet for my business", ctx)
    assert "Accel Business" in result.message or "Namaste" in result.message
    assert len(result.quick_chips) >= 3
    assert any("Kirana" in c for c in result.quick_chips)
    assert ctx.turn_count == 1


def test_advisor_profiling_turn_detects_segment():
    ctx = AdvisorContext(session_id="test-2")
    # Turn 1
    run_advisor_turn("Hi", ctx)
    # Turn 2: User specifies Kirana shop
    result = run_advisor_turn("I have a small kirana grocery store in Lucknow", ctx)
    assert ctx.segment == "micro"
    assert "micro" in result.message.lower() or "broadband" in result.message.lower() or "team" in result.message.lower()


def test_advisor_quoting_turn_generates_verified_quote():
    ctx = AdvisorContext(session_id="test-3", segment="micro", employees=3, sites=1, city_tier=2)
    result = run_advisor_turn("We have 3 staff members, please give me the quote", ctx)
    assert result.quote is not None
    assert result.quote_24m is not None
    assert result.quote.term_months == 12
    assert result.quote_24m.term_months == 24
    # 24 month plan has monthly savings
    assert result.quote_24m.saving_vs_baseline_monthly > 0
    # Add-ons are computed
    assert result.available_addons is not None
    assert len(result.available_addons) > 0
    # Next step is proceeding to KYC
    assert result.next_step == "proceed_to_kyc"


def test_advisor_quote_matches_pricing_engine_exactly():
    ctx = AdvisorContext(session_id="test-4", segment="micro", employees=3, sites=1, city_tier=2)
    result = run_advisor_turn("Give me a quote for 3 staff", ctx)
    expected_quote = build_quote(ctx.to_profile(), result.bundle, 12, CAT)
    assert result.quote.monthly_total == expected_quote.monthly_total
    assert result.quote.one_time_total == expected_quote.one_time_total
    assert result.quote.contract_value == expected_quote.contract_value


def test_tool_check_gstin_updates_context():
    ctx = AdvisorContext(session_id="test-5")
    # Valid active GSTIN
    res, _, _, _ = execute_tool("tool_check_gstin", {"gstin": "32TTNXB7384T1ZW"}, ctx, CAT)
    assert res["valid"] is True
    assert res["status"] == "Active"
    assert ctx.has_gstin is True
    assert ctx.gstin_status == "Active"


def test_tool_propose_and_quote_rejects_invalid_custom_bundle():
    ctx = AdvisorContext(session_id="test-6", employees=2, city_tier=2)
    # CCTV without internet dependency
    res, q12, q24, _ = execute_tool(
        "tool_propose_and_quote_bundle",
        {
            "segment": "micro",
            "city_tier": 2,
            "employees": 2,
            "sites": 1,
            "custom_items": {"P23": 1},
        },
        ctx,
        CAT,
    )
    assert res["valid"] is False
    assert any("internet plan" in issue for issue in res["issues"])
    assert q12 is None


def test_end_to_end_adaptive_flow_turn_sequence():
    ctx = AdvisorContext(session_id="test-e2e")
    # Turn 1: Welcome
    r1 = run_advisor_turn("Hello", ctx)
    assert r1.next_step == "select_business_type"

    # Turn 2: Select business type
    r2 = run_advisor_turn("Kirana / Grocery Shop", ctx)
    assert ctx.segment == "micro"

    # Turn 3: Answer questions and get quote
    r3 = run_advisor_turn("3 staff, 1 shop, ready for quote", ctx)
    assert r3.quote is not None
    assert r3.next_step == "proceed_to_kyc"
    assert "Switch to 24-Month Plan" in r3.quick_chips or "Proceed to KYC" in r3.quick_chips
