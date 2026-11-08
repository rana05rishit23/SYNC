"""Unit tests for OCR extraction, image preprocessing, and field parsing."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from backend.app.ocr import (
    ExtractedDocument,
    detect_doc_type,
    extract_document,
    extract_text,
    parse_fields,
    preprocess_image,
)


def test_preprocess_image_returns_binary_or_grayscale():
    # Create a small RGB test image with background and simulated text/watermark
    img = Image.new("RGB", (100, 100), color=(240, 240, 240))
    result = preprocess_image(img)
    assert isinstance(result, Image.Image)
    assert result.mode in ("L", "1")
    assert result.size == (100, 100)


def test_detect_doc_type():
    # GSTIN Certificate
    assert detect_doc_type("Government of India\nGoods and Services Tax Certificate") == "gstin_cert"
    assert detect_doc_type("Registration under GSTIN: 32TTNXB7384T1ZW") == "gstin_cert"

    # Udyam Certificate
    assert detect_doc_type("UDYAM REGISTRATION CERTIFICATE\nUDYAM-KL-00-3078801") == "udyam_cert"
    assert detect_doc_type("Ministry of MSME - Udyog Aadhaar Memorandum") == "udyam_cert"

    # Signatory ID
    assert detect_doc_type("INCOME TAX DEPARTMENT - PERMANENT ACCOUNT NUMBER CARD") == "signatory_id"
    assert detect_doc_type("Unique Identification Authority of India - Aadhaar Card") == "signatory_id"
    assert detect_doc_type("Election Commission of India - Voter Identity Card") == "signatory_id"
    assert detect_doc_type("PAN Card No: ABCDE1234F") == "signatory_id"

    # Address Proof
    assert detect_doc_type("Electricity Distribution Bill - Oct 2026") == "address_proof"
    assert detect_doc_type("Utility Bill for Water Connection") == "address_proof"
    assert detect_doc_type("Residential Rent Agreement / Lease Deed") == "address_proof"

    # Unknown
    assert detect_doc_type("Random invoice receipt $500") == "unknown"
    assert detect_doc_type("") == "unknown"


def test_parse_fields_gstin_cert():
    raw_text = """
    Government of India
    Form GST REG-06
    Registration Certificate
    Registration Number (GSTIN) : 32TTNXB7384T1ZW
    1. Legal Name : Balaji Dental Care LLP
    2. Trade Name, if any : Balaji Dental Care
    3. Address of Principal Place of Business :
       12/4, MG Road, Ernakulam,
       Kochi, Kerala - 682016
    State : Kerala
    Date of Registration : 01/07/2017
    """
    fields = parse_fields(raw_text)
    assert fields["extracted_gstin"] == "32TTNXB7384T1ZW"
    assert fields["extracted_name"] == "Balaji Dental Care LLP"
    assert "12/4, MG Road, Ernakulam" in (fields["extracted_address"] or "")
    assert "Kochi, Kerala - 682016" in (fields["extracted_address"] or "")
    assert fields["extracted_udyam"] is None


def test_parse_fields_lenient_gstin_with_ocr_confusables():
    # OCR mistook leading '0' for 'O' in 03PJNXS4821G1ZB
    raw_text = """
    GOODS AND SERVICES TAX
    GSTIN: O3PJNXS4821G1ZB
    Legal Name: Saffron Bakers
    Principal Place of Business: 45 Mall Road, Ludhiana
    """
    fields = parse_fields(raw_text)
    assert fields["extracted_gstin"] == "O3PJNXS4821G1ZB"
    assert fields["extracted_name"] == "Saffron Bakers"
    assert fields["extracted_address"] == "45 Mall Road, Ludhiana"


def test_parse_fields_udyam_cert():
    raw_text = """
    UDYAM REGISTRATION CERTIFICATE
    UDYAM REGISTRATION NUMBER : UDYAM-KL-00-3078801
    NAME OF ENTERPRISE : Balaji Dental Care LLP
    MAJOR ACTIVITY : Services
    ADDRESS OF ENTERPRISE :
    12/4, MG Road, Ernakulam, Kochi
    State : Kerala
    """
    fields = parse_fields(raw_text)
    assert fields["extracted_udyam"] == "UDYAM-KL-00-3078801"
    assert fields["extracted_name"] == "Balaji Dental Care LLP"
    assert "12/4, MG Road, Ernakulam, Kochi" in (fields["extracted_address"] or "")
    assert fields["extracted_gstin"] is None


def test_parse_fields_lenient_udyam():
    # OCR mistook '0' for 'O'
    raw_text = "UDYAM NUMBER: UDYAM-MH-OO-1234567"
    fields = parse_fields(raw_text)
    assert fields["extracted_udyam"] == "UDYAM-MH-00-1234567"


def test_parse_fields_signatory_id():
    raw_text = """
    INCOME TAX DEPARTMENT
    GOVT OF INDIA
    Permanent Account Number
    Name: Rajesh Kumar
    Father's Name: Suresh Kumar
    DOB: 12/05/1985
    """
    fields = parse_fields(raw_text)
    assert fields["extracted_name"] == "Rajesh Kumar"
    assert fields["extracted_gstin"] is None
    assert fields["extracted_udyam"] is None


def test_parse_fields_empty_text():
    fields = parse_fields("")
    assert fields["extracted_gstin"] is None
    assert fields["extracted_udyam"] is None
    assert fields["extracted_name"] is None
    assert fields["extracted_address"] is None


def test_extract_text_missing_file():
    text, conf = extract_text(Path("non_existent_document_file.jpg"))
    assert text == ""
    assert conf == 0.0


def test_extract_text_with_mocked_pytesseract(tmp_path: Path):
    test_img_path = tmp_path / "test.png"
    Image.new("RGB", (50, 50), color="white").save(test_img_path)

    with patch("backend.app.ocr.pytesseract") as mock_pytesseract:
        mock_pytesseract.image_to_string.return_value = "GSTIN: 32TTNXB7384T1ZW"
        mock_pytesseract.image_to_data.return_value = {
            "conf": ["95", "90", "-1"],
            "text": ["GSTIN:", "32TTNXB7384T1ZW", ""],
        }

        text, conf = extract_text(test_img_path)
        assert "32TTNXB7384T1ZW" in text
        assert conf == 92.5


def test_extract_text_graceful_on_tesseract_error(tmp_path: Path):
    test_img_path = tmp_path / "test.png"
    Image.new("RGB", (50, 50), color="white").save(test_img_path)

    with patch("backend.app.ocr.pytesseract") as mock_pytesseract:
        mock_pytesseract.image_to_string.side_effect = OSError("Tesseract not installed")

        text, conf = extract_text(test_img_path)
        assert text == ""
        assert conf == 0.0


def test_extract_document_integration(tmp_path: Path):
    test_img_path = tmp_path / "gstin_doc.png"
    Image.new("RGB", (60, 60), color="white").save(test_img_path)

    mock_ocr_output = """
    Goods and Services Tax
    GSTIN: 32TTNXB7384T1ZW
    Legal Name: Balaji Dental Care LLP
    Principal Place of Business: Kochi, Kerala
    """
    with patch("backend.app.ocr.extract_text") as mock_extract:
        mock_extract.return_value = (mock_ocr_output, 88.5)

        doc = extract_document(test_img_path)
        assert isinstance(doc, ExtractedDocument)
        assert doc.doc_type == "gstin_cert"
        assert doc.extracted_gstin == "32TTNXB7384T1ZW"
        assert doc.extracted_name == "Balaji Dental Care LLP"
        assert doc.extracted_address == "Kochi, Kerala"
        assert doc.confidence == 88.5


# ---------------------------------------------------------------------------
# Real Image Integration Tests (using data/kyc_documents/ synthetic images)
# ---------------------------------------------------------------------------
KYC_DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "kyc_documents"
requires_kyc_docs = pytest.mark.skipif(
    not KYC_DOCS_DIR.exists() or not (KYC_DOCS_DIR / "K01_gstin_active.png").exists(),
    reason="data/kyc_documents synthetic specimens not present",
)


@requires_kyc_docs
def test_real_image_k01_active_gstin():
    from backend.app.registries import verify_gstin_with_repair

    doc = extract_document(KYC_DOCS_DIR / "K01_gstin_active.png")
    assert doc.doc_type == "gstin_cert"
    assert doc.extracted_gstin == "32TTNXB7384T1ZW"
    assert "Balaji Dental Care" in (doc.extracted_name or "")
    assert doc.confidence >= 85.0

    rec, notes = verify_gstin_with_repair(doc.extracted_gstin)
    assert rec is not None
    assert rec.status == "Active"
    assert "Balaji Dental Care" in rec.trade_name


@requires_kyc_docs
def test_real_image_k02_cancelled_gstin_with_repair():
    from backend.app.registries import verify_gstin_with_repair

    doc = extract_document(KYC_DOCS_DIR / "K02_gstin_cancelled.png")
    assert doc.doc_type == "gstin_cert"
    assert "Neha Kaur" in (doc.extracted_name or "")
    assert doc.confidence >= 85.0

    # The OCR text is repaired through structured confusable recovery to the cancelled registry entry
    rec, notes = verify_gstin_with_repair(doc.extracted_gstin or "")
    assert rec is not None
    assert rec.status == "Cancelled"
    assert "Vindhya Tours" in rec.trade_name


@requires_kyc_docs
def test_real_image_k03_udyam_active():
    from backend.app.registries import lookup_udyam

    doc = extract_document(KYC_DOCS_DIR / "K03_udyam_active.png")
    assert doc.doc_type == "udyam_cert"
    assert doc.extracted_udyam == "UDYAM-KL-00-3078801"
    assert "Balaji Dental Care" in (doc.extracted_name or "")
    assert doc.confidence >= 85.0

    rec = lookup_udyam(doc.extracted_udyam)
    assert rec is not None
    assert rec.enterprise_name == "Balaji Dental Care LLP"
    assert rec.city == "Kochi"


@requires_kyc_docs
def test_real_image_k04_watermarked_gstin_suppressed():
    from backend.app.registries import verify_gstin_with_repair

    # Watermarked certificate: Otsu / adaptive thresholding suppresses background watermark
    doc = extract_document(KYC_DOCS_DIR / "K04_gstin_watermarked.png")
    assert doc.doc_type == "gstin_cert"
    assert doc.extracted_gstin == "32TTNXB7384T1ZW"
    assert "Balaji Dental Care" in (doc.extracted_name or "")
    assert doc.confidence >= 80.0

    rec, _ = verify_gstin_with_repair(doc.extracted_gstin)
    assert rec is not None
    assert rec.status == "Active"


@requires_kyc_docs
def test_real_image_k05_cancelled_clean():
    from backend.app.registries import verify_gstin_with_repair

    doc = extract_document(KYC_DOCS_DIR / "K05_gstin_cancelled_clean.png")
    assert doc.doc_type == "gstin_cert"
    assert "Ishita Menon" in (doc.extracted_name or "")

    rec, notes = verify_gstin_with_repair(doc.extracted_gstin or "")
    assert rec is not None
    assert rec.status == "Cancelled"
    assert "Prime General Store" in rec.trade_name
