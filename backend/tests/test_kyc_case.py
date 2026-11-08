"""Case-level KYC rules, tested on hand-built documents (fast, no OCR) plus golden official specimens."""
from dataclasses import replace
from datetime import date

import pytest

from backend.app.docscan import MIN_SHARPNESS, ScannedDoc
from backend.app.kyc_case import TODAY, verify_case
from backend.app.registries import GSTINRecord, UdyamRecord

GST = "37IHKXV5167L1ZS"
REG = {GST: GSTINRecord(GST, "Neha Kaur (Prop. Vindhya Tours & Travels)", "Vindhya Tours & Travels", "Proprietorship",
                        "Shop/Unit 1, Tech Park Road, Visakhapatnam, Andhra Pradesh - 118383", "Andhra Pradesh", "2025-06-07", "Active")}
UREG = {"UDYAM-AP-00-1": UdyamRecord("UDYAM-AP-00-1", "Vindhya Tours & Travels", "Micro", "Services", "Travel", "Andhra Pradesh", "Visakhapatnam", "2024-03-24")}
ADDR = "Shop/Unit 1, Tech Park Road, Visakhapatnam, Andhra Pradesh - 118383"


def docs(**over):
    base = {
        "gst_certificate": ScannedDoc("gst_certificate", {"gstin": GST, "gstin_candidates": GST, "gstin_raw": GST, "trade_name": "Vindhya Tours & Travels",
                                      "legal_name": "Neha Kaur (Prop. Vindhya Tours & Travels)", "address": ADDR}, 1300),
        "udyam_certificate": ScannedDoc("udyam_certificate", {"enterprise_name": "Vindhya Tours & Travels", "city": "Visakhapatnam", "state": "Andhra Pradesh"}, 1000),
        "signatory_id": ScannedDoc("signatory_id", {"name": "Neha Kaur", "business": "Vindhya Tours & Travels", "valid_till": "2028-03-26"}, 1700),
        "address_proof": ScannedDoc("address_proof", {"billed_to": "Vindhya Tours & Travels", "address": ADDR, "bill_date": "2026-06-26"}, 1050),
    }
    for k, v in over.items():
        if v is None:
            base.pop(k)
        else:
            base[k] = ScannedDoc(base[k].doc_type, {**base[k].fields, **v.get("fields", {})}, v.get("sharpness", base[k].sharpness))
    return list(base.values())


def run(**over):
    return verify_case(docs(**over), TODAY, REG, UREG)


def codes(r):
    return {f.code for f in r.findings}


def test_clean_case_is_approved():
    r = run()
    assert r.status == "approved" and r.findings == []
    assert r.gstin_masked == "37IH********1ZS" and r.udyam_masked is not None


def test_cancelled_gstin_is_rejected_not_reviewed():
    reg = {GST: replace(REG[GST], status="Cancelled")}
    r = verify_case(docs(), TODAY, reg, UREG)
    assert r.status == "rejected" and "GSTIN_CANCELLED" in codes(r)


def test_missing_document_is_incomplete():
    r = run(udyam_certificate=None)
    assert r.status == "incomplete" and codes(r) == {"MISSING_DOC"}


def test_blurred_document_is_reupload_and_never_guessed():
    r = run(gst_certificate={"sharpness": MIN_SHARPNESS - 1})
    assert r.status == "reupload" and "BLURRY_DOC" in codes(r) and r.gstin_masked is None


@pytest.mark.parametrize("bill_date,expected", [("2026-04-16", "approved"), ("2026-04-15", "reupload")])
def test_address_proof_age_boundary_is_90_days(bill_date, expected):
    assert (date(2026, 7, 15) - date.fromisoformat(bill_date)).days in (90, 91)
    assert run(address_proof={"fields": {"bill_date": bill_date}}).status == expected


def test_future_dated_bill_goes_to_review():
    assert "ADDR_FUTURE_DATED" in codes(run(address_proof={"fields": {"bill_date": "2026-08-01"}}))


def test_expired_signatory_id_is_reupload():
    r = run(signatory_id={"fields": {"valid_till": "2026-07-14"}})
    assert r.status == "reupload" and "ID_EXPIRED" in codes(r)


def test_trade_name_contradicting_legal_name_goes_to_review():
    r = run(gst_certificate={"fields": {"trade_name": "Neelgiri Physio Clinic", "legal_name": "Neha Banerjee (Prop. Shri Physio Clinic)"}})
    assert "GST_LEGAL_VS_TRADE" in codes(r)


def test_cross_document_mismatches_go_to_review():
    assert "UDYAM_NAME_MISMATCH" in codes(run(udyam_certificate={"fields": {"enterprise_name": "Other Traders"}}))
    assert "ID_BUSINESS_MISMATCH" in codes(run(signatory_id={"fields": {"business": "Other Traders"}}))
    assert "ADDR_NAME_MISMATCH" in codes(run(address_proof={"fields": {"billed_to": "Other Traders"}}))
    assert "ADDR_MISMATCH" in codes(run(address_proof={"fields": {"address": "Shop 9, Main Road, Pune, Maharashtra - 411001"}}))


def test_same_name_in_another_city_is_not_a_udyam_match():
    r = run(udyam_certificate={"fields": {"city": "Mumbai", "state": "Maharashtra"}})
    assert "UDYAM_NOT_FOUND" in codes(r) and r.status == "review"


def test_near_twin_udyam_names_use_exact_name_tiebreak():
    twin = UdyamRecord("UDYAM-AP-00-2", "Vindhya Tours & Travels LLP", "Micro", "Services", "Travel", "Andhra Pradesh", "Visakhapatnam", "2024-01-01")
    r = verify_case(docs(), TODAY, REG, {**UREG, "UDYAM-AP-00-2": twin})
    assert "UDYAM_NOT_FOUND" not in codes(r)


def test_gstin_reconciled_by_name_and_pincode_asks_owner_to_confirm():
    noisy = {"fields": {"gstin_candidates": None, "gstin_raw": "37THKXV5167L12S"}}
    r = run(gst_certificate=noisy)
    assert r.status == "approved" and r.confirm_gstin_with_owner and "GSTIN_CONFIRM" in codes(r)
    far = {"fields": {"gstin_candidates": None, "gstin_raw": "99ZZZZZ0000Z9Z9"}}
    assert "GSTIN_UNRESOLVED" in codes(run(gst_certificate=far))
