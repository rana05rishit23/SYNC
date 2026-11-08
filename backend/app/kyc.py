"""KYC Orchestrator and Human Review Queue for Project Udyam Digital.

Deterministic 4-layer KYC document verification pipeline:
1. Consent gate: Explicit user consent verified before document processing
2. OCR Extraction: Tesseract-based extraction with confidence scoring
3. Checksum & Confusable Repair: Strict checksum validation + character repair
4. Registry Cross-Check & Name Matching: Real mock registry lookup + fuzzy match
5. Human Review Queue: Discrepancies, edge cases, and low confidence routed to review
"""
from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, is_dataclass
import logging
from pathlib import Path
from typing import Any
import uuid

from .ocr import extract_document
from .kyc_matcher import match_name
from .registries import lookup_udyam, verify_gstin_with_repair
from .sessions import SESSIONS, SessionStore
from .validators import is_valid_gstin, mask_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class KYCResult:
    session_id: str
    doc_type: str
    status: str  # 'approved' | 'review' | 'rejected'
    extracted_gstin: str | None
    extracted_udyam: str | None
    extracted_name: str | None
    registry_match: dict | None  # registry record data if matched
    name_match_score: float | None
    audit_trail: list[str]  # step-by-step log of what happened
    created_at: str


@dataclass
class ReviewItem:
    item_id: str  # UUID
    session_id: str
    doc_type: str
    reason: str  # Why it's flagged
    extracted_data: dict  # What OCR found
    registry_data: dict  # What registry says (if anything)
    status: str  # 'pending' | 'approved' | 'rejected'
    created_at: str
    reviewed_at: str | None
    reviewer_note: str | None


# ---------------------------------------------------------------------------
# Review Queue Management
# ---------------------------------------------------------------------------

REVIEW_QUEUE: dict[str, ReviewItem] = {}


def add_to_review(
    session_id: str,
    doc_type: str,
    reason: str,
    extracted: dict,
    registry: dict,
) -> ReviewItem:
    """Add a flagged document to the module-level human review queue."""
    item_id = str(uuid.uuid4())
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    item = ReviewItem(
        item_id=item_id,
        session_id=session_id,
        doc_type=doc_type,
        reason=reason,
        extracted_data=extracted if extracted is not None else {},
        registry_data=registry if registry is not None else {},
        status="pending",
        created_at=now,
        reviewed_at=None,
        reviewer_note=None,
    )
    REVIEW_QUEUE[item_id] = item
    logger.info(
        "[REVIEW] Added item=%s session=%s doc_type=%s reason='%s'",
        item_id,
        session_id,
        doc_type,
        reason,
    )
    return item


def get_pending_reviews() -> list[ReviewItem]:
    """Retrieve all pending review items awaiting human review."""
    return [item for item in REVIEW_QUEUE.values() if item.status == "pending"]


def resolve_review(item_id: str, approved: bool, note: str = "") -> ReviewItem | None:
    """Resolve an item in the human review queue with approval or rejection."""
    item = REVIEW_QUEUE.get(item_id)
    if not item:
        logger.warning("[REVIEW] Attempted to resolve unknown review item_id=%s", item_id)
        return None

    item.status = "approved" if approved else "rejected"
    item.reviewed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    item.reviewer_note = note
    logger.info(
        "[REVIEW] Resolved item=%s status=%s note='%s'",
        item_id,
        item.status,
        note,
    )
    return item


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _to_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {"raw": str(obj)}


def _extract_field(obj: Any, *field_names: str, default: Any = None) -> Any:
    for name in field_names:
        if isinstance(obj, dict):
            if name in obj and obj[name] is not None:
                return obj[name]
        else:
            val = getattr(obj, name, None)
            if val is not None:
                return val
    return default


def _compute_match_score(name: str, record: Any) -> float:
    try:
        res = match_name(name, record)
    except Exception:
        reg_name = ""
        if isinstance(record, str):
            reg_name = record
        elif hasattr(record, "legal_name"):
            reg_name = getattr(record, "legal_name", "")
        elif hasattr(record, "enterprise_name"):
            reg_name = getattr(record, "enterprise_name", "")
        res = match_name(name, reg_name)

    if isinstance(res, (int, float)):
        return float(res)
    if hasattr(res, "score"):
        return float(res.score)
    if isinstance(res, dict) and "score" in res:
        return float(res["score"])
    return float(res)


# ---------------------------------------------------------------------------
# Main KYC Pipeline
# ---------------------------------------------------------------------------

def run_kyc_pipeline(
    session_id: str,
    image_path: Path,
    sessions: SessionStore | None = None,
) -> KYCResult:
    """Run the 4-layer KYC document verification pipeline.

    1. Verify user consent
    2. Extract document via OCR (extract_document)
    3. Verify and repair identifiers (GSTIN or Udyam) against mock registries
    4. Match extracted name against registry records (match_name)
    5. Determine status ('approved', 'review', 'rejected') and flag for review if needed
    """
    path = Path(image_path)
    store = sessions if sessions is not None else SESSIONS

    # Layer 0: Privacy consent gate
    if not store.has_consent(session_id):
        logger.warning("[KYC] Consent check failed for session_id=%s", session_id)
        raise ValueError("Consent required before KYC processing")

    audit_trail: list[str] = [
        f"Consent verified for session {session_id}",
        f"Starting OCR extraction for image: {path.name}",
    ]

    # Layer 1: OCR Extraction
    doc = extract_document(path)
    extracted_data = _to_dict(doc)

    doc_type = str(_extract_field(doc, "doc_type", "type", "document_type", default="unknown"))
    extracted_gstin = _extract_field(
        doc, "extracted_gstin", "gstin", "gst_number"
    )
    if extracted_gstin:
        extracted_gstin = str(extracted_gstin).strip().upper()

    extracted_udyam = _extract_field(
        doc, "extracted_udyam", "udyam", "udyam_number", "udyam_registration_number"
    )
    if extracted_udyam:
        extracted_udyam = str(extracted_udyam).strip().upper()

    extracted_name = _extract_field(
        doc, "extracted_name", "name", "legal_name", "enterprise_name", "trade_name"
    )
    if extracted_name:
        extracted_name = str(extracted_name).strip()

    confidence_val = _extract_field(
        doc, "confidence", "ocr_confidence", "score", default=100.0
    )
    try:
        confidence = float(confidence_val) if confidence_val is not None else 100.0
    except (ValueError, TypeError):
        confidence = 100.0

    audit_trail.append(
        f"OCR extraction completed: doc_type='{doc_type}', "
        f"GSTIN={mask_id(extracted_gstin)}, Udyam={mask_id(extracted_udyam)}, "
        f"name='{extracted_name or ''}', confidence={confidence:.1f}%"
    )

    status: str
    reason: str | None = None
    registry_match: dict | None = None
    name_match_score: float | None = None

    # Layer 2 & 3: Verification & Registry Cross-Check
    if not extracted_gstin and not extracted_udyam:
        status = "review"
        reason = "No identifiers extracted"
        audit_trail.append("No identifiers (GSTIN or Udyam) extracted from document")

    elif extracted_gstin:
        audit_trail.append(f"Verifying extracted GSTIN: {mask_id(extracted_gstin)}")
        rec, notes = verify_gstin_with_repair(extracted_gstin)
        audit_trail.extend(notes)

        if rec is None:
            status = "rejected"
            if not is_valid_gstin(extracted_gstin):
                reason = "GSTIN invalid and unrepairable"
                audit_trail.append("GSTIN format/checksum invalid and unrepairable. Rejected.")
            else:
                reason = "GSTIN not found in registry"
                audit_trail.append("GSTIN checksum valid but not found in mock registry. Rejected.")
        else:
            registry_match = asdict(rec)
            if rec.status in ("Cancelled", "Suspended"):
                status = "rejected"
                reason = f"GSTIN is {rec.status}"
                audit_trail.append(
                    f"GSTIN {mask_id(rec.gstin)} status in registry is {rec.status}. Rejected."
                )
            else:
                audit_trail.append(
                    f"GSTIN verified Active in registry for legal name: '{rec.legal_name}'"
                )
                if extracted_name:
                    name_match_score = round(_compute_match_score(extracted_name, rec), 4)
                    audit_trail.append(f"Name match evaluated: score={name_match_score:.2f}")

                    if confidence < 60.0:
                        status = "review"
                        reason = f"OCR confidence low ({confidence:.1f} < 60)"
                        audit_trail.append(
                            f"OCR confidence ({confidence:.1f}%) < 60%. Flagged for review."
                        )
                    elif 0.6 <= name_match_score < 0.8:
                        status = "review"
                        reason = f"Name match score requires review ({name_match_score:.2f})"
                        audit_trail.append(
                            f"Name match score ({name_match_score:.2f}) in review range [0.6, 0.8). Flagged for review."
                        )
                    elif name_match_score >= 0.8:
                        status = "approved"
                        audit_trail.append(
                            f"Active GSTIN and name match {name_match_score:.2f} >= 0.8. Approved."
                        )
                    else:  # name_match_score < 0.6
                        status = "rejected"
                        reason = f"Name mismatch with registry ({name_match_score:.2f} < 0.6)"
                        audit_trail.append(
                            f"Name match score {name_match_score:.2f} < 0.6. Rejected."
                        )
                else:
                    name_match_score = None
                    status = "review"
                    reason = "No name extracted from document for verification"
                    audit_trail.append(
                        "No business name extracted to verify against registry. Flagged for review."
                    )

    else:
        # Udyam extracted
        audit_trail.append(f"Looking up Udyam number: {mask_id(extracted_udyam)}")
        rec_udyam = lookup_udyam(extracted_udyam)

        if rec_udyam is None:
            status = "rejected"
            reason = "Udyam number not found in registry"
            audit_trail.append("Udyam number not found in mock registry. Rejected.")
        else:
            registry_match = asdict(rec_udyam)
            audit_trail.append(
                f"Udyam record found for enterprise: '{rec_udyam.enterprise_name}'"
            )
            if extracted_name:
                name_match_score = round(_compute_match_score(extracted_name, rec_udyam), 4)
                audit_trail.append(f"Name match evaluated: score={name_match_score:.2f}")

                if confidence < 60.0:
                    status = "review"
                    reason = f"OCR confidence low ({confidence:.1f} < 60)"
                    audit_trail.append(
                        f"OCR confidence ({confidence:.1f}%) < 60%. Flagged for review."
                    )
                elif 0.6 <= name_match_score < 0.8:
                    status = "review"
                    reason = f"Name match score requires review ({name_match_score:.2f})"
                    audit_trail.append(
                        f"Name match score ({name_match_score:.2f}) in review range [0.6, 0.8). Flagged for review."
                    )
                elif name_match_score >= 0.8:
                    status = "approved"
                    audit_trail.append(
                        f"Valid Udyam registration and name match {name_match_score:.2f} >= 0.8. Approved."
                    )
                else:  # name_match_score < 0.6
                    status = "rejected"
                    reason = f"Name mismatch with registry ({name_match_score:.2f} < 0.6)"
                    audit_trail.append(
                        f"Name match score {name_match_score:.2f} < 0.6. Rejected."
                    )
            else:
                name_match_score = None
                status = "review"
                reason = "No name extracted from document for verification"
                audit_trail.append(
                    "No enterprise name extracted to verify against registry. Flagged for review."
                )

    # Layer 4: Human Review Queue Routing
    if status == "review":
        item_reason = reason or "Flagged for manual review"
        add_to_review(
            session_id=session_id,
            doc_type=doc_type,
            reason=item_reason,
            extracted=extracted_data,
            registry=registry_match or {},
        )
        audit_trail.append(f"Routed to human review queue: {item_reason}")

    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = KYCResult(
        session_id=session_id,
        doc_type=doc_type,
        status=status,
        extracted_gstin=extracted_gstin,
        extracted_udyam=extracted_udyam,
        extracted_name=extracted_name,
        registry_match=registry_match,
        name_match_score=name_match_score,
        audit_trail=audit_trail,
        created_at=created_at,
    )

    logger.info(
        "[KYC] Completed session=%s status=%s doc_type=%s reason=%s",
        session_id,
        status,
        doc_type,
        reason,
    )
    return result
