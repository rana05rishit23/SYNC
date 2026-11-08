import pandas as pd
from pathlib import Path
from backend.app.validators import is_valid_gstin, is_valid_udyam, repair_gstin, mask_id

DATA = Path(__file__).resolve().parents[2] / "data"


def test_every_registry_gstin_passes_checksum():
    gst = pd.read_csv(DATA / "mock_gstin_registry.csv")
    assert gst.gstin.map(is_valid_gstin).all()          # README claim, verified


def test_gstin_rejects_bad_checksum_and_format():
    good = "32TTNXB7384T1ZW"
    assert is_valid_gstin(good)
    assert not is_valid_gstin(good[:-1] + "X")           # wrong checksum
    assert not is_valid_gstin(good[:-1])                  # too short
    assert not is_valid_gstin(None)


def test_every_registry_udyam_passes_format():
    udy = pd.read_csv(DATA / "mock_udyam_registry.csv")
    assert udy.udyam_number.map(is_valid_udyam).all()
    assert not is_valid_udyam("UDYAM-KL-00-30788")


def test_repair_fixes_single_ocr_confusion():
    good = "32TTNXB7384T1ZW"
    garbled = good.replace("B", "8", 1)                   # OCR read B as 8
    assert garbled != good and not is_valid_gstin(garbled)
    assert good in repair_gstin(garbled)


def test_repair_gives_up_on_heavily_garbled_text():
    assert repair_gstin("19E0GKH4} SyiZ,") == []          # real OCR output from K02: needs layer 2/3


def test_mask_id():
    assert mask_id("32TTNXB7384T1ZW") == "32TT********1ZW"
    assert mask_id("abc") == "***"
    assert mask_id(None) == ""
