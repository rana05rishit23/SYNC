import pytest
from backend.app.catalogue import BusinessProfile, load_catalogue
from backend.app.pricing import (InvalidBundle, InvalidTerm, available_addons, build_quote,
                                 term_options, unit_price)

CAT = load_catalogue()
T1 = BusinessProfile(city_tier=1, employees=20, sites=1, gstin_status="Active")


def test_unit_price_by_term_for_fibre_100():
    p = CAT["P01"]                       # range 999-1199
    assert unit_price(p, 12) == 1119
    assert unit_price(p, 24) == 1039


def test_every_price_stays_inside_catalogue_range_and_longer_term_is_cheaper():
    for p in CAT.values():
        p12, p24 = unit_price(p, 12), unit_price(p, 24)
        assert p.price_min <= p24 <= p12 <= p.price_max
        if p.price_max > p.price_min:
            assert p24 < p12


def test_setup_fee_is_per_unit_and_quantity_multiplies_price():
    q = build_quote(T1, {"P01": 1, "P19": 2}, 12, CAT)      # POS: setup 2500 per counter
    pos = next(l for l in q.lines if l.product_id == "P19")
    assert pos.setup == 5000 and pos.monthly == 2 * pos.unit_price
    assert q.one_time_total == 5000
    assert q.monthly_total == sum(l.monthly for l in q.lines)


def test_quote_totals_are_consistent():
    q = build_quote(T1, {"P02": 1, "P23": 4}, 24, CAT)
    assert q.first_payment == q.monthly_total + q.one_time_total
    assert q.contract_value == q.monthly_total * 24 + q.one_time_total
    assert q.saving_vs_baseline_monthly > 0
    assert q.install_days == max(CAT["P02"].install_days, CAT["P23"].install_days)


def test_12_month_quote_has_no_saving():
    assert build_quote(T1, {"P01": 1}, 12, CAT).saving_vs_baseline_monthly == 0


def test_invalid_bundle_is_refused_with_reasons():
    with pytest.raises(InvalidBundle) as e:
        build_quote(T1, {"P23": 1}, 12, CAT)                 # CCTV with no internet plan
    assert {i.code for i in e.value.issues} == {"MISSING_DEPENDENCY"}


def test_empty_bundle_is_refused():
    with pytest.raises(InvalidBundle):
        build_quote(T1, {}, 12, CAT)


def test_term_must_respect_longest_minimum_contract():
    items = {"P05": 1}                                        # leased line, 12 months min
    assert term_options(items, CAT) == [12, 24]
    items = {"P10": 1, "P01": 1}                              # SD-WAN needs 24 months
    prof = BusinessProfile(1, 20, 3, "Active")
    assert term_options(items, CAT) == [24]
    with pytest.raises(InvalidTerm):
        build_quote(prof, items, 12, CAT)


def test_addons_respect_dependencies_and_eligibility():
    with_net = {p.product_id for p in available_addons(T1, {"P01": 1}, CAT)}
    assert {"P23", "P22"} <= with_net and "P01" not in with_net
    assert "P10" not in with_net                              # SD-WAN needs 3+ sites
    no_net = {p.product_id for p in available_addons(T1, {"P07": 1}, CAT)}
    assert "P23" not in no_net                                # CCTV needs internet
