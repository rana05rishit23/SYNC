"""FastAPI backend application for Project Udyam Digital.

Provides REST endpoints for:
- Advisor chat with Groq tool-calling loop (or mock in test/sandbox)
- Deterministic pricing engine and add-on toggling
- Product catalogue and segment templates
- Mock registry checks (GSTIN and Udyam) with OCR repair
- Consent gate for privacy compliance
"""
from __future__ import annotations
from dataclasses import asdict
import hmac
import os
from pathlib import Path
import tempfile
from typing import Annotated, Any
import uuid
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints

from .advisor import (
    AdvisorContext,
    GroqLLMClient,
    MockAdvisorLLMClient,
    run_advisor_turn,
)
from .catalogue import BusinessProfile, load_catalogue, validate_bundle
from .config import get_settings
from .i18n import get_translations
from .kyc import (
    KYCResult,
    ReviewItem,
    get_pending_reviews,
    resolve_review,
    run_kyc_pipeline,
)
from .language import detect
from .pricing import available_addons, build_quote, term_options, unit_price
from .registries import lookup_gstin, lookup_udyam, verify_gstin_with_repair
from .sessions import SESSIONS
from .templates import TEMPLATES, get_segment_template, propose_bundle

app = FastAPI(
    title="Project Udyam Digital API",
    description="Accel Business SMB Onboarding: Advisor -> Quote -> Smart KYC -> Tracker",
    version="2.0.0",
)

# The UI is served by this app (same origin), so CORS only matters for a separately hosted frontend.
# Origins come from CORS_ORIGINS (comma-separated); no wildcard, no credentials.
_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Admin-Key"],
)

CATALOGUE = load_catalogue()
SETTINGS = get_settings()

# Input limits for the KYC upload endpoint.
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
SessionId = Annotated[str, StringConstraints(pattern=SESSION_ID_PATTERN)]
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf"}


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    """Guard for reviewer-only endpoints. Disabled (always 403) unless ADMIN_API_KEY is set."""
    expected = os.getenv("ADMIN_API_KEY")
    if not expected or not x_admin_key or not hmac.compare_digest(x_admin_key, expected):
        raise HTTPException(status_code=403, detail="Reviewer access requires a valid X-Admin-Key.")


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: SessionId = Field(..., description="Unique user session ID")
    message: str = Field(..., description="User message or quick-chip selection")
    preferred_language: str = Field(default="English", description="Preferred language (Hindi, English, etc.)")


class ProfileSchema(BaseModel):
    city_tier: int = Field(default=2, ge=1, le=2)
    employees: int = Field(default=3, ge=1)
    sites: int = Field(default=1, ge=1)
    gstin_status: str | None = Field(default=None)


class QuoteRequest(BaseModel):
    profile: ProfileSchema
    items: dict[str, int] = Field(..., description="Map of product_id to quantity")
    term_months: int = Field(default=12, description="12 or 24 months")


class ToggleAddonRequest(BaseModel):
    profile: ProfileSchema
    current_items: dict[str, int]
    product_id: str
    action: str = Field(default="toggle", description="'add', 'remove', or 'toggle'")
    term_months: int = 12


class ConsentRequest(BaseModel):
    session_id: SessionId
    owner_name: str
    phone: str
    purpose: str = "Digital onboarding & KYC verification"


class ReviewResolutionRequest(BaseModel):
    approved: bool = Field(..., description="True to approve, False to reject")
    note: str = Field(default="", description="Reviewer reason or notes")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "Project Udyam Digital",
        "catalogue_products": len(CATALOGUE),
        "advisor_model": SETTINGS.advisor_model,
        "groq_configured": bool(SETTINGS.groq_api_key),
    }


@app.get("/api/catalogue")
def get_catalogue():
    return [
        {
            "product_id": p.product_id,
            "category": p.category,
            "name": p.name,
            "description": p.description,
            "unit": p.unit,
            "price_min": p.price_min,
            "price_max": p.price_max,
            "setup_fee": p.setup_fee,
            "min_contract_months": p.min_contract_months,
            "available_city_tiers": list(sorted(p.city_tiers)),
            "min_employees": p.min_employees,
            "min_sites": p.min_sites,
            "requires_gstin": p.requires_gstin,
            "install_days": p.install_days,
            "target_segments": list(sorted(p.target_segments)),
        }
        for p in CATALOGUE.values()
    ]


@app.get("/api/templates")
def get_templates():
    return [
        {
            "segment_id": t.segment_id,
            "name": t.name,
            "tagline": t.tagline,
            "quick_chips": list(t.quick_chips),
            "curated_addons": list(t.curated_addons),
        }
        for t in TEMPLATES.values()
    ]


@app.get("/api/i18n/{language}")
def get_ui_translations(language: str):
    """Retrieve UI translations for any language, cached in SQLite."""
    return get_translations(language)


@app.post("/api/chat")
def chat_turn(req: ChatRequest):
    ctx = SESSIONS.get_or_create_context(req.session_id, req.preferred_language)

    # Automatic language detection (Layer A + Layer B)
    detection = detect(req.message)
    if detection.confidence >= 0.7:
        ctx.preferred_language = detection.language

    # Use Groq client if API key is provided and not disabled, else use Mock client
    is_live = bool(SETTINGS.groq_api_key)
    if is_live:
        client = GroqLLMClient(api_key=SETTINGS.groq_api_key, model=SETTINGS.advisor_model)
    else:
        client = MockAdvisorLLMClient(CATALOGUE)

    result = run_advisor_turn(req.message, ctx, client=client, catalogue=CATALOGUE)
    SESSIONS.save_context(ctx)

    return {
        "session_id": req.session_id,
        "message": result.message,
        "detected_language": detection.language,
        "language_confidence": detection.confidence,
        "quick_chips": result.quick_chips,
        "next_step": result.next_step,
        "bundle": result.bundle,
        "quote": result.quote,
        "quote_24m": result.quote_24m,
        "available_addons": result.available_addons,
        "validation_issues": result.validation_issues,
        "stage": ctx.stage,
        "segment": ctx.segment,
        "groq_configured": is_live,
        "is_mock": not is_live,
    }


@app.post("/api/quote")
def calculate_quote(req: QuoteRequest):
    prof = BusinessProfile(
        city_tier=req.profile.city_tier,
        employees=req.profile.employees,
        sites=req.profile.sites,
        gstin_status=req.profile.gstin_status,
    )
    issues = validate_bundle(prof, req.items, CATALOGUE)
    if issues:
        return {
            "valid": False,
            "issues": [i.message for i in issues],
            "quote": None,
        }

    q = build_quote(prof, req.items, req.term_months, CATALOGUE)
    addons = available_addons(prof, req.items, CATALOGUE)

    return {
        "valid": True,
        "term_months": q.term_months,
        "monthly_total": q.monthly_total,
        "one_time_total": q.one_time_total,
        "first_payment": q.first_payment,
        "contract_value": q.contract_value,
        "saving_vs_baseline_monthly": q.saving_vs_baseline_monthly,
        "install_days": q.install_days,
        "lines": [
            {
                "product_id": l.product_id,
                "name": l.name,
                "unit": l.unit,
                "quantity": l.quantity,
                "unit_price": l.unit_price,
                "monthly": l.monthly,
                "setup": l.setup,
            }
            for l in q.lines
        ],
        "available_addons": [
            {
                "product_id": p.product_id,
                "name": p.name,
                "category": p.category,
                "unit": p.unit,
                "price_12m": unit_price(p, 12),
                "price_24m": unit_price(p, 24),
                "setup_fee": p.setup_fee,
            }
            for p in addons
        ],
    }


@app.post("/api/quote/toggle-addon")
def toggle_addon(req: ToggleAddonRequest):
    items = dict(req.current_items)
    pid = req.product_id

    if req.action == "add":
        items[pid] = items.get(pid, 0) + 1
    elif req.action == "remove":
        items.pop(pid, None)
    else:  # toggle
        if pid in items:
            items.pop(pid)
        else:
            items[pid] = 1

    prof = BusinessProfile(
        city_tier=req.profile.city_tier,
        employees=req.profile.employees,
        sites=req.profile.sites,
        gstin_status=req.profile.gstin_status,
    )
    issues = validate_bundle(prof, items, CATALOGUE)
    if issues:
        return {
            "valid": False,
            "issues": [i.message for i in issues],
            "current_items": req.current_items,
        }

    q = build_quote(prof, items, req.term_months, CATALOGUE)
    addons = available_addons(prof, items, CATALOGUE)

    return {
        "valid": True,
        "current_items": items,
        "monthly_total": q.monthly_total,
        "one_time_total": q.one_time_total,
        "first_payment": q.first_payment,
        "contract_value": q.contract_value,
        "saving_vs_baseline_monthly": q.saving_vs_baseline_monthly,
        "install_days": q.install_days,
        "available_addons": [
            {
                "product_id": p.product_id,
                "name": p.name,
                "price_12m": unit_price(p, 12),
                "price_24m": unit_price(p, 24),
                "setup_fee": p.setup_fee,
            }
            for p in addons
        ],
    }


@app.get("/api/registry/gstin/{gstin}")
def get_gstin(gstin: str):
    rec, notes = verify_gstin_with_repair(gstin)
    if not rec:
        raise HTTPException(status_code=404, detail="GSTIN not found or invalid checksum")
    return {
        "gstin": rec.gstin,
        "masked_gstin": rec.masked_gstin,
        "trade_name": rec.trade_name,
        "legal_name": rec.legal_name,
        "constitution": rec.constitution,
        "state": rec.state,
        "status": rec.status,
        "is_active": rec.is_active,
        "audit_notes": notes,
    }


@app.get("/api/registry/udyam/{udyam}")
def get_udyam(udyam: str):
    rec = lookup_udyam(udyam)
    if not rec:
        raise HTTPException(status_code=404, detail="Udyam number not found in registry")
    return {
        "udyam_number": rec.udyam_number,
        "masked_udyam": rec.masked_udyam,
        "enterprise_name": rec.enterprise_name,
        "enterprise_type": rec.enterprise_type,
        "major_activity": rec.major_activity,
        "city": rec.city,
        "state": rec.state,
    }


@app.delete("/api/consent/{session_id}")
def withdraw_consent(session_id: Annotated[str, PathParam(pattern=SESSION_ID_PATTERN)]):
    """DPDP: the owner can withdraw consent at any time; this also erases their KYC result."""
    if not SESSIONS.withdraw_consent(session_id):
        raise HTTPException(status_code=404, detail="No consent recorded for this session")
    return {"status": "withdrawn", "session_id": session_id}


@app.post("/api/consent")
def submit_consent(req: ConsentRequest):
    record = SESSIONS.record_consent(
        session_id=req.session_id,
        owner_name=req.owner_name,
        phone=req.phone,
        purpose=req.purpose,
    )
    return {
        "status": "consented",
        "session_id": record.session_id,
        "masked_phone": record.masked_phone,
        "consented_at": record.consented_at,
        "purpose": record.purpose,
    }


# ---------------------------------------------------------------------------
# Smart KYC Document Pipeline Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/kyc/upload")
async def upload_kyc_document(
    session_id: str = Form(..., pattern=SESSION_ID_PATTERN, description="Active session ID with recorded consent"),
    file: UploadFile = File(..., description="Uploaded document image (PNG/JPEG/PDF), max 8 MB"),
):
    """Intake KYC document, enforce privacy consent gate, and run 4-layer verification pipeline."""
    if not SESSIONS.has_consent(session_id):
        raise HTTPException(
            status_code=403,
            detail="Consent required before document upload and KYC processing.",
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG or PDF file.")

    # The server picks the filename: nothing user-controlled goes into the path.
    temp_dir = Path(tempfile.gettempdir()) / "udyam_kyc_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4().hex}{suffix}"

    try:
        written = 0
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File too large (limit 8 MB).")
                buffer.write(chunk)

        result = run_kyc_pipeline(session_id=session_id, image_path=temp_path)
        SESSIONS.save_kyc_result(session_id, result)
        return asdict(result)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass  # best effort; the OS temp directory is cleaned separately


@app.get("/api/kyc/status/{session_id}")
def get_kyc_status(session_id: Annotated[str, PathParam(pattern=SESSION_ID_PATTERN)]):
    """Retrieve the latest KYC verification outcome for a session."""
    res = SESSIONS.get_kyc_result(session_id)
    if not res:
        raise HTTPException(status_code=404, detail="No KYC submission found for session")
    return asdict(res)


@app.get("/api/kyc/review", dependencies=[Depends(require_admin)])
def list_pending_reviews():
    """Retrieve all KYC submissions routed to the human review queue."""
    return [asdict(item) for item in get_pending_reviews()]


@app.post("/api/kyc/review/{item_id}", dependencies=[Depends(require_admin)])
def resolve_review_endpoint(item_id: str, req: ReviewResolutionRequest):
    """Resolve an item in the KYC review queue (manual approval or rejection)."""
    item = resolve_review(item_id, approved=req.approved, note=req.note)
    if not item:
        raise HTTPException(status_code=404, detail="Review item not found")
    return asdict(item)


# Mount frontend single-page application
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

