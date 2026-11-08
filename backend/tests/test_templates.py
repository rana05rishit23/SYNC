import pytest
from backend.app.catalogue import BusinessProfile, load_catalogue, validate_bundle
from backend.app.pricing import build_quote
from backend.app.templates import (
    TEMPLATES,
    detect_segment,
    get_segment_template,
    propose_bundle,
)

CAT = load_catalogue()


def test_segment_detection():
    assert detect_segment("I run a Kirana shop in Lucknow") == "micro"
    assert detect_segment("Bakery and cafe") == "micro"
    assert detect_segment("Dental clinic with 10 staff") == "prof"
    assert detect_segment("CA and Tax consulting office") == "prof"
    assert detect_segment("Retail chain with 4 branches") == "multi"
    assert detect_segment("Auto parts manufacturing unit") == "mfg"
    assert detect_segment("Unknown general store") == "micro"


def test_all_templates_exist():
    assert set(TEMPLATES.keys()) == {"micro", "prof", "multi", "mfg"}
    for t in TEMPLATES.values():
        assert len(t.quick_chips) >= 3
        assert len(t.curated_addons) >= 3
        assert t.name
        assert t.tagline


def test_micro_bundle_is_rule_valid_for_small_shop():
    prof = BusinessProfile(city_tier=2, employees=3, sites=1, gstin_status=None)
    bundle = propose_bundle("micro", prof, catalogue=CAT)
    assert "P01" in bundle
    assert "P07" in bundle
    assert validate_bundle(prof, bundle, CAT) == []

    quote = build_quote(prof, bundle, 12, CAT)
    assert quote.monthly_total > 0


def test_professional_office_bundle_validity():
    prof = BusinessProfile(city_tier=1, employees=12, sites=1, gstin_status="Active")
    bundle = propose_bundle("prof", prof, {"needs_telephony": True}, catalogue=CAT)
    assert "P02" in bundle or "P01" in bundle
    assert "P16" in bundle
    assert "P11" in bundle
    assert "P12" in bundle
    assert validate_bundle(prof, bundle, CAT) == []

    quote = build_quote(prof, bundle, 24, CAT)
    assert quote.saving_vs_baseline_monthly > 0


def test_multisite_bundle_scales_with_sites():
    prof = BusinessProfile(city_tier=1, employees=30, sites=4, gstin_status="Active")
    bundle = propose_bundle("multi", prof, {"needs_cctv": True}, catalogue=CAT)
    assert bundle["P02"] == 4
    assert bundle["P09"] == 4
    assert bundle["P10"] == 4  # SD-WAN for 3+ sites
    assert bundle["P23"] == 8  # 2 cameras per site
    assert validate_bundle(prof, bundle, CAT) == []


def test_mfg_bundle_respects_leased_line_and_firewall_rules():
    # 15 employees, Active GSTIN -> gets Leased line (P05) and Firewall (P21)
    prof = BusinessProfile(city_tier=2, employees=15, sites=1, gstin_status="Active")
    bundle = propose_bundle("mfg", prof, {"needs_iot": True}, catalogue=CAT)
    assert "P05" in bundle
    assert "P21" in bundle
    assert "P24" in bundle  # IoT
    assert validate_bundle(prof, bundle, CAT) == []


def test_mfg_bundle_gracefully_degrades_if_no_gstin():
    # If manufacturer has no GSTIN, P05 & P21 cannot be purchased; propose_bundle auto-adapts
    prof = BusinessProfile(city_tier=2, employees=15, sites=1, gstin_status=None)
    bundle = propose_bundle("mfg", prof, catalogue=CAT)
    assert "P05" not in bundle
    assert "P21" not in bundle
    assert validate_bundle(prof, bundle, CAT) == []
