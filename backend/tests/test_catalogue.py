import pandas as pd
from pathlib import Path
from backend.app.catalogue import BusinessProfile, load_catalogue, validate_bundle, eligible_products

CAT = load_catalogue()
DATA = Path(__file__).resolve().parents[2] / "data"
codes = lambda issues: {i.code for i in issues}


def test_catalogue_loads_25_products_with_parsed_rules():
    assert len(CAT) == 25
    assert CAT["P03"].city_tiers == frozenset({1})
    assert CAT["P23"].requires_one_of == frozenset({"P01", "P02", "P03", "P04", "P05", "P06"})


def test_kirana_without_gstin_cannot_buy_leased_line():
    kirana = BusinessProfile(city_tier=2, employees=3, sites=1, gstin_status=None)
    assert codes(validate_bundle(kirana, {"P05": 1}, CAT)) == {"MIN_EMPLOYEES", "GSTIN_REQUIRED"}


def test_cancelled_gstin_is_not_the_same_as_missing_gstin():
    p = BusinessProfile(1, 20, 1, "Cancelled")
    assert codes(validate_bundle(p, {"P12": 3, "P01": 1}, CAT)) == {"GSTIN_INACTIVE"}


def test_tier_gating():
    assert codes(validate_bundle(BusinessProfile(2, 20, 1, "Active"), {"P03": 1}, CAT)) == {"TIER"}
    assert validate_bundle(BusinessProfile(1, 20, 1, "Active"), {"P03": 1}, CAT) == []


def test_dependency_is_resolved_at_bundle_level():
    p = BusinessProfile(1, 5, 1, None)
    assert codes(validate_bundle(p, {"P23": 2}, CAT)) == {"MISSING_DEPENDENCY"}   # CCTV alone
    assert validate_bundle(p, {"P23": 2, "P01": 1}, CAT) == []                     # CCTV + fibre


def test_unknown_product_and_bad_quantity():
    p = BusinessProfile(1, 5, 1, "Active")
    assert codes(validate_bundle(p, {"P99": 1, "P01": 0}, CAT)) == {"UNKNOWN_PRODUCT", "BAD_QUANTITY"}


def test_eligible_products_for_tiny_shop_excludes_gated_items():
    ids = {p.product_id for p in eligible_products(BusinessProfile(2, 2, 1, None), CAT)}
    assert {"P01", "P04", "P07"} <= ids
    assert not ids & {"P03", "P05", "P10", "P21"}


def test_characterisation_of_historical_sales():
    """Of the 368 bundles the field force actually sold, 56 break catalogue rules.
    Pinned so a rules change that alters this number is a conscious decision."""
    L = pd.read_csv(DATA / "smb_leads.csv")
    G = pd.read_csv(DATA / "mock_gstin_registry.csv").set_index("gstin").status
    bad = 0
    for _, r in L[L.outcome == "Activated"].iterrows():
        st = G.get(r.gstin) if r.has_gstin == "Y" else None
        items = {x.split("x")[0]: int(x.split("x")[1]) for x in r.bundle_products.split(";")}
        bad += bool(validate_bundle(BusinessProfile(int(r.city_tier), int(r.employee_count), int(r.num_sites), st), items, CAT))
    assert bad == 56
