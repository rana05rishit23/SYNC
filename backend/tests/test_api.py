import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["catalogue_products"] == 25


def test_api_catalogue():
    res = client.get("/api/catalogue")
    assert res.status_code == 200
    products = res.json()
    assert len(products) == 25
    pids = {p["product_id"] for p in products}
    assert "P01" in pids
    assert "P23" in pids


def test_api_templates():
    res = client.get("/api/templates")
    assert res.status_code == 200
    templates = res.json()
    assert len(templates) == 4
    seg_ids = {t["segment_id"] for t in templates}
    assert seg_ids == {"micro", "prof", "multi", "mfg"}


def test_api_chat_flow():
    # Turn 1: Greeting
    res1 = client.post("/api/chat", json={
        "session_id": "api-sess-1",
        "message": "Hello",
        "preferred_language": "English",
    })
    assert res1.status_code == 200
    d1 = res1.json()
    assert len(d1["quick_chips"]) > 0
    assert d1["next_step"] == "select_business_type"

    # Turn 2: Provide business type
    res2 = client.post("/api/chat", json={
        "session_id": "api-sess-1",
        "message": "I run a Kirana shop",
        "preferred_language": "English",
    })
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["segment"] == "micro"

    # Turn 3: Finalize quote
    res3 = client.post("/api/chat", json={
        "session_id": "api-sess-1",
        "message": "3 staff, give me quote",
        "preferred_language": "English",
    })
    assert res3.status_code == 200
    d3 = res3.json()
    assert d3["quote"] is not None
    assert d3["quote_24m"] is not None
    assert d3["quote_24m"]["saving_vs_baseline_monthly"] > 0


def test_api_calculate_quote():
    req = {
        "profile": {"city_tier": 2, "employees": 3, "sites": 1, "gstin_status": None},
        "items": {"P01": 1, "P07": 1},
        "term_months": 12,
    }
    res = client.post("/api/quote", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is True
    assert data["monthly_total"] > 0
    assert len(data["available_addons"]) > 0


def test_api_calculate_quote_invalid_bundle():
    # CCTV without internet dependency
    req = {
        "profile": {"city_tier": 2, "employees": 3, "sites": 1, "gstin_status": None},
        "items": {"P23": 1},
        "term_months": 12,
    }
    res = client.post("/api/quote", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is False
    assert len(data["issues"]) > 0


def test_api_toggle_addon():
    req = {
        "profile": {"city_tier": 1, "employees": 5, "sites": 1, "gstin_status": "Active"},
        "current_items": {"P01": 1},
        "product_id": "P23",  # Cloud CCTV
        "action": "toggle",
        "term_months": 12,
    }
    # Toggle ON
    res = client.post("/api/quote/toggle-addon", json=req)
    assert res.status_code == 200
    d1 = res.json()
    assert d1["valid"] is True
    assert "P23" in d1["current_items"]

    # Toggle OFF
    req["current_items"] = d1["current_items"]
    res2 = client.post("/api/quote/toggle-addon", json=req)
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["valid"] is True
    assert "P23" not in d2["current_items"]


def test_api_registry_gstin_and_udyam():
    # GSTIN lookup
    res_gst = client.get("/api/registry/gstin/32TTNXB7384T1ZW")
    assert res_gst.status_code == 200
    d_gst = res_gst.json()
    assert d_gst["trade_name"] == "Balaji Dental Care LLP"
    assert d_gst["is_active"] is True
    assert d_gst["masked_gstin"] != d_gst["gstin"]

    # Udyam lookup
    res_udyam = client.get("/api/registry/udyam/UDYAM-KL-00-3078801")
    assert res_udyam.status_code == 200
    d_udyam = res_udyam.json()
    assert d_udyam["enterprise_name"] == "Balaji Dental Care LLP"


def test_api_consent_recording():
    res = client.post("/api/consent", json={
        "session_id": "consent-test-1",
        "owner_name": "Ramesh Kumar",
        "phone": "9876543210",
        "purpose": "Airtel Accel Business digital verification",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "consented"
    assert data["masked_phone"] == "98******10"


def test_api_kyc_upload_without_consent_fails(tmp_path):
    # Missing consent should return 403 Forbidden
    test_img = tmp_path / "test_doc.png"
    test_img.write_bytes(b"dummy image bytes")

    with open(test_img, "rb") as f:
        res = client.post(
            "/api/kyc/upload",
            data={"session_id": "no-consent-session"},
            files={"file": ("test_doc.png", f, "image/png")},
        )
    assert res.status_code == 403
    assert "Consent required" in res.json()["detail"]


def test_api_kyc_upload_and_status(tmp_path, monkeypatch):
    from backend.app.ocr import ExtractedDocument
    from backend.app.main import SESSIONS

    session_id = "kyc-api-test-sess"
    SESSIONS.record_consent(session_id, "Balaji Dental", "9876543210")

    # Mock extract_document to return predictable document fields
    def mock_extract(path):
        return ExtractedDocument(
            doc_type="gstin_cert",
            raw_text="GSTIN: 32TTNXB7384T1ZW Legal Name: Balaji Dental Care LLP",
            extracted_gstin="32TTNXB7384T1ZW",
            extracted_udyam=None,
            extracted_name="Balaji Dental Care LLP",
            confidence=95.0,
        )

    monkeypatch.setattr("backend.app.kyc.extract_document", mock_extract)

    test_img = tmp_path / "doc.png"
    test_img.write_bytes(b"mock image content")

    with open(test_img, "rb") as f:
        res = client.post(
            "/api/kyc/upload",
            data={"session_id": session_id},
            files={"file": ("doc.png", f, "image/png")},
        )

    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == session_id
    assert data["status"] == "approved"
    assert data["extracted_gstin"] == "32TTNXB7384T1ZW"

    # Status check endpoint
    res_status = client.get(f"/api/kyc/status/{session_id}")
    assert res_status.status_code == 200
    st_data = res_status.json()
    assert st_data["status"] == "approved"
    assert st_data["session_id"] == session_id


def test_api_kyc_review_flow(monkeypatch):
    from backend.app.kyc import REVIEW_QUEUE, add_to_review

    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    admin = {"X-Admin-Key": "test-admin-key"}

    # Add item directly to review
    item = add_to_review(
        session_id="review-api-sess",
        doc_type="gstin_cert",
        reason="Ambiguous OCR match",
        extracted={"extracted_name": "Balaji Clinic"},
        registry={"legal_name": "Balaji Dental Care LLP"},
    )

    # 1. List reviews
    res_list = client.get("/api/kyc/review", headers=admin)
    assert res_list.status_code == 200
    items = res_list.json()
    assert any(i["item_id"] == item.item_id for i in items)

    # 2. Resolve review
    res_resolve = client.post(
        f"/api/kyc/review/{item.item_id}",
        json={"approved": True, "note": "Verified manually via physical registry"},
        headers=admin,
    )
    assert res_resolve.status_code == 200
    resolved = res_resolve.json()
    assert resolved["status"] == "approved"
    assert resolved["reviewer_note"] == "Verified manually via physical registry"

