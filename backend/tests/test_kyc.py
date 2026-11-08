"""Unit and integration tests for the KYC orchestrator and review queue."""
from pathlib import Path
from unittest.mock import patch
import pytest

from backend.app.kyc import (
    KYCResult,
    ReviewItem,
    REVIEW_QUEUE,
    add_to_review,
    get_pending_reviews,
    resolve_review,
    run_kyc_pipeline,
)
from backend.app.ocr import ExtractedDocument
from backend.app.sessions import SessionStore


@pytest.fixture(autouse=True)
def clean_review_queue():
    """Clear review queue before each test."""
    REVIEW_QUEUE.clear()
    yield
    REVIEW_QUEUE.clear()


@pytest.fixture
def mock_session_store():
    store = SessionStore()
    store.record_consent(
        session_id="test_session_1",
        owner_name="Test Owner",
        phone="9876543210",
        purpose="KYC verification",
    )
    return store


# ---------------------------------------------------------------------------
# Consent Gate Tests
# ---------------------------------------------------------------------------

def test_consent_required_raises_value_error():
    empty_store = SessionStore()
    with pytest.raises(ValueError, match="Consent required before KYC processing"):
        run_kyc_pipeline(
            session_id="unconsented_session",
            image_path=Path("dummy.png"),
            sessions=empty_store,
        )


def test_consent_passed_with_mock_store(mock_session_store):
    with patch("backend.app.kyc.extract_document") as mock_extract:
        mock_extract.return_value = ExtractedDocument(
            doc_type="gstin_cert",
            raw_text="GSTIN 32TTNXB7384T1ZW Balaji Dental Care LLP",
            extracted_gstin="32TTNXB7384T1ZW",
            extracted_name="Balaji Dental Care LLP",
            confidence=95.0,
        )
        res = run_kyc_pipeline(
            session_id="test_session_1",
            image_path=Path("cert.png"),
            sessions=mock_session_store,
        )
        assert res.status == "approved"
        assert res.session_id == "test_session_1"
        assert res.registry_match is not None
        assert res.registry_match["trade_name"] == "Balaji Dental Care LLP"


# ---------------------------------------------------------------------------
# Review Queue Management Tests
# ---------------------------------------------------------------------------

def test_add_to_review_creates_item():
    item = add_to_review(
        session_id="s123",
        doc_type="gstin_cert",
        reason="Low confidence",
        extracted={"name": "Acme Corp"},
        registry={"name": "Acme Inc"},
    )
    assert isinstance(item, ReviewItem)
    assert item.item_id in REVIEW_QUEUE
    assert item.status == "pending"
    assert item.reason == "Low confidence"
    assert item.reviewed_at is None
    assert item.reviewer_note is None


def test_get_pending_reviews():
    item1 = add_to_review("s1", "gstin_cert", "reason 1", {}, {})
    item2 = add_to_review("s2", "udyam_cert", "reason 2", {}, {})

    pending = get_pending_reviews()
    assert len(pending) == 2
    assert {it.item_id for it in pending} == {item1.item_id, item2.item_id}

    resolve_review(item1.item_id, approved=True, note="Looks legit")
    pending_after = get_pending_reviews()
    assert len(pending_after) == 1
    assert pending_after[0].item_id == item2.item_id


def test_resolve_review():
    item = add_to_review("s1", "gstin_cert", "reason", {}, {})
    resolved = resolve_review(item.item_id, approved=False, note="Fraudulent doc")

    assert resolved is not None
    assert resolved.status == "rejected"
    assert resolved.reviewer_note == "Fraudulent doc"
    assert resolved.reviewed_at is not None

    # Test resolving non-existent item
    assert resolve_review("non_existent_id", approved=True) is None


# ---------------------------------------------------------------------------
# GSTIN Pipeline Tests
# ---------------------------------------------------------------------------

def test_gstin_active_high_match_approved(mock_session_store):
    with patch("backend.app.kyc.extract_document") as mock_extract:
        mock_extract.return_value = ExtractedDocument(
            doc_type="gstin_cert",
            raw_text="32TTNXB7384T1ZW Balaji Dental Care",
            extracted_gstin="32TTNXB7384T1ZW",
            extracted_name="Balaji Dental Care",
            confidence=90.0,
        )
        res = run_kyc_pipeline("test_session_1", Path("doc.png"), sessions=mock_session_store)
        assert res.status == "approved"
        assert res.name_match_score is not None
        assert res.name_match_score >= 0.8
        assert len(REVIEW_QUEUE) == 0


def test_gstin_repaired_approved(mock_session_store):
    # 'O' instead of '0' in '03PJNXS4821G1ZB' -> Saffron Bakers
    with patch("backend.app.kyc.extract_document") as mock_extract:
        mock_extract.return_value = ExtractedDocument(
            doc_type="gstin_cert",
            raw_text="O3PJNXS4821G1ZB Saffron Bakers",
            extracted_gstin="O3PJNXS4821G1ZB",
            extracted_name="Saffron Bakers",
            confidence=88.0,
        )
        res = run_kyc_pipeline("test_session_1", Path("doc.png"), sessions=mock_session_store)
        assert res.status == "approved"
        assert res.registry_match is not None
        assert res.registry_match["gstin"] == "03PJNXS4821G1ZB"
        assert any("Repaired" in note for note in res.audit_trail)


def test_gstin_cancelled_or_suspended_rejected(mock_session_store):
    # Lookup a cancelled GSTIN from registry
    from backend.app.registries import load_gstin_registry

    reg = load_gstin_registry()
    cancelled_gstin = next(r.gstin for r in reg.values() if r.status == "Cancelled")
    cancelled_rec = reg[cancelled_gstin]

    with patch("backend.app.kyc.extract_document") as mock_extract:
        mock_extract.return_value = ExtractedDocument(
            doc_type="gstin_cert",
            raw_text=f"{cancelled_gstin} {cancelled_rec.legal_name}",
            extracted_gstin=cancelled_gstin,
            extracted_name=cancelled_rec.legal_name,
            confidence=95.0,
        )
        res = run_kyc_pipeline("test_session_1", Path("doc.png"), sessions=mock_session_store)
        assert res.status == "rejected"
        assert "Cancelled" in res.audit_trail[-1] or any("Cancelled" in n for n in res.audit_trail)
        assert len(REVIEW_QUEUE) == 0


def test_gstin_unrepairable_rejected(mock_session_store):
    with patch("backend.app.kyc.extract_document") as mock_extract:
        mock_extract.return_value = ExtractedDocument(
            doc_type="gstin_cert",
            raw_text="INVALID_GARBLED_GSTIN Some Company",
            extracted_gstin="INVALID_GARBLED_GSTIN",
            extracted_name="Some Company",
            confidence=75.0,
        )
        res = run_kyc_pipeline("test_session_1", Path("doc.png"), sessions=mock_session_store)
        assert res.status == "rejected"
        assert any("unrepairable" in n.lower() or "rejected" in n.lower() for n in res.audit_trail)


def test_gstin_active_low_confidence_sent_to_review(mock_session_store):
    with patch("backend.app.kyc.extract_document") as mock_extract:
        mock_extract.return_value = ExtractedDocument(
            doc_type="gstin_cert",
            raw_text="32TTNXB7384T1ZW Balaji Dental Care LLP",
            extracted_gstin="32TTNXB7384T1ZW",
            extracted_name="Balaji Dental Care LLP",
            confidence=55.0,  # Below 60
        )
        res = run_kyc_pipeline("test_session_1", Path("doc.png"), sessions=mock_session_store)
        assert res.status == "review"
        assert len(REVIEW_QUEUE) == 1
        item = list(REVIEW_QUEUE.values())[0]
        assert "confidence" in item.reason.lower()


def test_gstin_active_medium_name_match_sent_to_review(mock_session_store):
    # Balaji Dental Care LLP vs Balaji Dental Clinic (partial similarity in 0.6-0.79 range)
    with patch("backend.app.kyc.extract_document") as mock_extract, patch(
        "backend.app.kyc.match_name"
    ) as mock_match:
        mock_extract.return_value = ExtractedDocument(
            doc_type="gstin_cert",
            raw_text="32TTNXB7384T1ZW Balaji Healthcare",
            extracted_gstin="32TTNXB7384T1ZW",
            extracted_name="Balaji Healthcare",
            confidence=85.0,
        )
        from backend.app.kyc_matcher import MatchResult
        mock_match.return_value = MatchResult(
            ocr_name="Balaji Healthcare",
            registry_name="Balaji Dental Care LLP",
            score=0.72,
            verdict="review",
        )
        res = run_kyc_pipeline("test_session_1", Path("doc.png"), sessions=mock_session_store)
        assert res.status == "review"
        assert res.name_match_score == 0.72
        assert len(REVIEW_QUEUE) == 1


def test_gstin_active_name_mismatch_rejected(mock_session_store):
    with patch("backend.app.kyc.extract_document") as mock_extract, patch(
        "backend.app.kyc.match_name"
    ) as mock_match:
        mock_extract.return_value = ExtractedDocument(
            doc_type="gstin_cert",
            raw_text="32TTNXB7384T1ZW Completely Different Pharmacy",
            extracted_gstin="32TTNXB7384T1ZW",
            extracted_name="Completely Different Pharmacy",
            confidence=92.0,
        )
        from backend.app.kyc_matcher import MatchResult
        mock_match.return_value = MatchResult(
            ocr_name="Completely Different Pharmacy",
            registry_name="Balaji Dental Care LLP",
            score=0.25,
            verdict="mismatch",
        )
        res = run_kyc_pipeline("test_session_1", Path("doc.png"), sessions=mock_session_store)
        assert res.status == "rejected"
        assert res.name_match_score == 0.25


# ---------------------------------------------------------------------------
# Udyam Pipeline Tests
# ---------------------------------------------------------------------------

def test_udyam_active_approved(mock_session_store):
    with patch("backend.app.kyc.extract_document") as mock_extract:
        mock_extract.return_value = ExtractedDocument(
            doc_type="udyam_cert",
            raw_text="UDYAM-KL-00-3078801 Balaji Dental Care LLP",
            extracted_udyam="UDYAM-KL-00-3078801",
            extracted_name="Balaji Dental Care LLP",
            confidence=95.0,
        )
        res = run_kyc_pipeline("test_session_1", Path("doc.png"), sessions=mock_session_store)
        assert res.status == "approved"
        assert res.extracted_udyam == "UDYAM-KL-00-3078801"
        assert res.registry_match is not None
        assert res.registry_match["enterprise_name"] == "Balaji Dental Care LLP"


def test_udyam_not_found_rejected(mock_session_store):
    with patch("backend.app.kyc.extract_document") as mock_extract:
        mock_extract.return_value = ExtractedDocument(
            doc_type="udyam_cert",
            raw_text="UDYAM-XX-99-9999999 Some Enterprise",
            extracted_udyam="UDYAM-XX-99-9999999",
            extracted_name="Some Enterprise",
            confidence=90.0,
        )
        res = run_kyc_pipeline("test_session_1", Path("doc.png"), sessions=mock_session_store)
        assert res.status == "rejected"


# ---------------------------------------------------------------------------
# Edge Case: No Identifiers Extracted
# ---------------------------------------------------------------------------

def test_no_identifiers_sent_to_review(mock_session_store):
    with patch("backend.app.kyc.extract_document") as mock_extract:
        mock_extract.return_value = ExtractedDocument(
            doc_type="unknown",
            raw_text="Some blurry image with no numbers",
            extracted_gstin=None,
            extracted_udyam=None,
            extracted_name=None,
            confidence=40.0,
        )
        res = run_kyc_pipeline("test_session_1", Path("doc.png"), sessions=mock_session_store)
        assert res.status == "review"
        assert len(REVIEW_QUEUE) == 1
        item = list(REVIEW_QUEUE.values())[0]
        assert "no identifiers" in item.reason.lower()
