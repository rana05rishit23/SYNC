import pytest
from backend.app.registries import (
    load_gstin_registry,
    load_udyam_registry,
    lookup_gstin,
    lookup_udyam,
    verify_gstin_with_repair,
)


def test_load_registries():
    gst_reg = load_gstin_registry()
    udyam_reg = load_udyam_registry()
    assert len(gst_reg) == 3534
    assert len(udyam_reg) == 2723


def test_active_gstin_lookup():
    rec = lookup_gstin("32TTNXB7384T1ZW")
    assert rec is not None
    assert rec.trade_name == "Balaji Dental Care LLP"
    assert rec.is_active is True
    assert rec.state == "Kerala"
    assert rec.masked_gstin.startswith("32TT")
    assert rec.masked_gstin.endswith("1ZW")


def test_suspended_and_cancelled_gstins_distinguished():
    gst_reg = load_gstin_registry()
    cancelled = [r for r in gst_reg.values() if r.status == "Cancelled"]
    suspended = [r for r in gst_reg.values() if r.status == "Suspended"]
    assert len(cancelled) == 167
    assert len(suspended) == 101

    c_rec = lookup_gstin(cancelled[0].gstin)
    assert c_rec.status == "Cancelled"
    assert not c_rec.is_active

    s_rec = lookup_gstin(suspended[0].gstin)
    assert s_rec.status == "Suspended"
    assert not s_rec.is_active


def test_udyam_lookup():
    rec = lookup_udyam("UDYAM-KL-00-3078801")
    assert rec is not None
    assert rec.enterprise_name == "Balaji Dental Care LLP"
    assert rec.enterprise_type == "Micro"
    assert rec.major_activity == "Services"
    assert rec.city == "Kochi"


def test_verify_gstin_with_ocr_repair():
    # Original valid GSTIN: 03PJNXS4821G1ZB
    # Simulate OCR confusing '0' with 'O' or '8' with 'B'
    corrupted = "O3PJNXS4821G1ZB"  # 'O' instead of '0'
    rec, notes = verify_gstin_with_repair(corrupted)
    assert rec is not None
    assert rec.gstin == "03PJNXS4821G1ZB"
    assert rec.trade_name == "Saffron Bakers"
    assert any("Repaired" in n for n in notes)


def test_verify_gstin_unknown_or_bad():
    rec, notes = verify_gstin_with_repair("99ZZZZZZZZZZZZZ")
    assert rec is None
