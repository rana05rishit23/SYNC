"""Golden tests on the official specimen documents (synthetic data supplied with the case)."""
from pathlib import Path

import pytest

from backend.app.docscan import MIN_SHARPNESS, scan
from backend.app.kyc_case import verify_case
from backend.app.validators import repair_gstin_structured

OFFICIAL = Path(__file__).resolve().parents[2] / "data" / "kyc_official"
pytestmark = pytest.mark.skipif(not OFFICIAL.exists(), reason="official specimens not present")
_cache: dict[str, object] = {}


def case(k):
    if k not in _cache:
        _cache[k] = verify_case([scan(p) for p in sorted(OFFICIAL.glob(f"{k}_*.png"))])
    return _cache[k]


def test_blurred_scan_is_detected_by_sharpness_alone():
    assert scan(OFFICIAL / "K04_gst_certificate.png").sharpness < MIN_SHARPNESS
    assert scan(OFFICIAL / "K01_udyam_certificate.png").sharpness > 3 * MIN_SHARPNESS


def test_watermarked_values_are_read_by_label():
    d = scan(OFFICIAL / "K01_gst_certificate.png")
    assert d.fields["trade_name"] == "Vindhya Tours & Travels"
    assert d.fields["gstin"] == "37IHKXV5167L1ZS"
    assert d.fields["date_of_liability"] == "2025-06-07"


def test_skewed_scan_is_straightened_before_reading():
    d = scan(OFFICIAL / "K02_gst_certificate.png")
    assert d.fields["trade_name"] == "Harmony Dental Care" and d.fields["legal_name"].lower().startswith("joseph shah")


def test_signatory_id_is_recognised_and_parsed():
    d = scan(OFFICIAL / "K01_signatory_id.png")
    assert d.doc_type == "signatory_id" and d.fields["name"] == "Neha Kaur" and d.fields["valid_till"] == "2028-03-26"


@pytest.mark.parametrize("k,status,code", [
    ("K01", "rejected", "GSTIN_CANCELLED"),
    ("K04", "reupload", "BLURRY_DOC"),
    ("K05", "incomplete", "MISSING_DOC"),
    ("K07", "review", "GST_LEGAL_VS_TRADE"),
    ("K11", "reupload", "ADDR_STALE"),
])
def test_planted_problems_are_caught(k, status, code):
    r = case(k)
    assert r.status == status and code in {f.code for f in r.findings}


@pytest.mark.parametrize("k", ["K08", "K12"])
def test_clean_applicants_are_approved_without_owner_confirmation(k):
    r = case(k)
    assert r.status == "approved" and not r.confirm_gstin_with_owner


def test_structured_repair_fixes_z_read_as_seven_or_two():
    assert "37IHKXV5167L1ZS" in repair_gstin_structured("37IHKXV5167L12S")
    assert "24XORXG9365M1Z3" in repair_gstin_structured("24X0RXG9365M173")
