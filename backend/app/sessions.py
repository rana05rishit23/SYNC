"""Session manager, consent gating, and privacy-preserving audit logging.

Implements ADR-004:
- Sessions store advisor context and conversation state.
- Consent gate requires explicit agreement before document handling or PII intake.
- All logs and traces mask identifiers using mask_id().
"""
from __future__ import annotations
import datetime
import logging
from dataclasses import dataclass, field
from typing import Any

from .advisor import AdvisorContext
from .validators import mask_id

logger = logging.getLogger("udyam.audit")


@dataclass
class ConsentRecord:
    session_id: str
    owner_name: str
    masked_phone: str
    purpose: str
    consented_at: str
    ip_hash: str = "local"


class SessionStore:
    """In-memory session store. Not thread-safe and not persistent: state is lost on restart. Demo scope only."""

    def __init__(self):
        self._contexts: dict[str, AdvisorContext] = {}
        self._consents: dict[str, ConsentRecord] = {}
        self._kyc_results: dict[str, Any] = {}

    def get_or_create_context(self, session_id: str, language: str = "English") -> AdvisorContext:
        if session_id not in self._contexts:
            self._contexts[session_id] = AdvisorContext(
                session_id=session_id,
                preferred_language=language,
            )
        return self._contexts[session_id]

    def save_context(self, context: AdvisorContext) -> None:
        self._contexts[context.session_id] = context

    def record_consent(
        self, session_id: str, owner_name: str, phone: str, purpose: str = "KYC verification & Accel connection onboarding"
    ) -> ConsentRecord:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rec = ConsentRecord(
            session_id=session_id,
            owner_name=owner_name,
            masked_phone=mask_id(phone, keep_start=2, keep_end=2),
            purpose=purpose,
            consented_at=now,
        )
        self._consents[session_id] = rec
        logger.info(f"[AUDIT] Consent logged for session={session_id}, phone={rec.masked_phone}, purpose={purpose}")
        return rec

    def withdraw_consent(self, session_id: str) -> bool:
        """Remove the consent record and the personal data tied to it. Returns False if none existed."""
        existed = self._consents.pop(session_id, None) is not None
        self._kyc_results.pop(session_id, None)
        if existed:
            logger.info("[AUDIT] Consent withdrawn, data erased for session=%s", session_id)
        return existed

    def has_consent(self, session_id: str) -> bool:
        return session_id in self._consents

    def get_consent(self, session_id: str) -> ConsentRecord | None:
        return self._consents.get(session_id)

    def save_kyc_result(self, session_id: str, result: Any) -> None:
        self._kyc_results[session_id] = result

    def get_kyc_result(self, session_id: str) -> Any | None:
        return self._kyc_results.get(session_id)


# Global singleton store
SESSIONS = SessionStore()

