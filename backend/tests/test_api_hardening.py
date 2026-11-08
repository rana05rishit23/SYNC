"""Regression tests for input validation and access control on the public API."""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.app.main as main
from backend.app.sessions import SESSIONS

client = TestClient(main.app, raise_server_exceptions=False)
PNG = ("scan.png", io.BytesIO(b"not-really-a-png"), "image/png")


@pytest.fixture
def consented():
    sid = "hardening-sess-1"
    SESSIONS.record_consent(sid, "Test Owner", "9876543210")
    yield sid
    SESSIONS.withdraw_consent(sid)


@pytest.mark.parametrize("bad", ["../../tmp/pwn", "a/b", "x" * 65, "has space", ""])
def test_session_ids_with_path_characters_are_rejected(bad):
    r = client.post("/api/consent", json={"session_id": bad, "owner_name": "x", "phone": "9999999999"})
    assert r.status_code == 422
    r = client.post("/api/kyc/upload", data={"session_id": bad}, files={"file": PNG})
    assert r.status_code == 422


def test_upload_rejects_unexpected_file_types(consented):
    r = client.post("/api/kyc/upload", data={"session_id": consented},
                    files={"file": ("payload.exe", io.BytesIO(b"MZ"), "application/octet-stream")})
    assert r.status_code == 415


def test_upload_rejects_oversized_files(consented, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/api/kyc/upload", data={"session_id": consented},
                    files={"file": ("big.png", io.BytesIO(b"x" * 50), "image/png")})
    assert r.status_code == 413


def test_upload_path_is_server_chosen_and_inside_upload_dir(consented, monkeypatch):
    seen = {}

    def spy(session_id, image_path):
        seen["path"] = Path(image_path).resolve()
        raise RuntimeError("stop after capturing the path")

    monkeypatch.setattr(main, "run_kyc_pipeline", spy)
    client.post("/api/kyc/upload", data={"session_id": consented}, files={"file": PNG})
    upload_dir = (Path(main.tempfile.gettempdir()) / "udyam_kyc_uploads").resolve()
    assert seen["path"].parent == upload_dir
    assert consented not in seen["path"].name
    assert not seen["path"].exists()            # temp file removed even on failure


def test_review_endpoints_need_the_admin_key(monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    assert client.get("/api/kyc/review").status_code == 403                                  # not configured: closed
    monkeypatch.setenv("ADMIN_API_KEY", "k")
    assert client.get("/api/kyc/review").status_code == 403                                  # no header
    assert client.get("/api/kyc/review", headers={"X-Admin-Key": "wrong"}).status_code == 403
    assert client.get("/api/kyc/review", headers={"X-Admin-Key": "k"}).status_code == 200
    assert client.post("/api/kyc/review/x", json={"approved": True}).status_code == 403


def test_withdrawing_consent_blocks_further_uploads_and_erases_results():
    sid = "withdraw-sess-1"
    SESSIONS.record_consent(sid, "Owner", "9876543210")
    SESSIONS.save_kyc_result(sid, {"status": "approved"})
    assert client.delete(f"/api/consent/{sid}").status_code == 200
    assert SESSIONS.get_kyc_result(sid) is None
    assert client.post("/api/kyc/upload", data={"session_id": sid}, files={"file": PNG}).status_code == 403
    assert client.delete(f"/api/consent/{sid}").status_code == 404


def test_cors_has_no_wildcard_origin():
    r = client.options("/api/health", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") != "*"
    assert "access-control-allow-origin" not in r.headers
