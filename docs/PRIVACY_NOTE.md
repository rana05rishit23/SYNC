# Data Privacy & Security Note

> **Compliance Architecture for Project Udyam Digital**  
> *Aligned with the Digital Personal Data Protection Act, 2023 (DPDP Act) and telecom regulatory guidelines.*

---

## 1. Regulatory Context & Objectives
Indian Micro, Small, and Medium Enterprises (MSMEs) handling identity and tax documents (GSTIN certificates, Udyam registration, Signatory Aadhaar/PAN) require bank-grade data protection during self-service telecom onboarding. Project Udyam Digital implements privacy by design, treating consent as an architectural gate rather than a cosmetic checkbox.

---

## 2. Core Privacy Principles

### A. Purpose Limitation & Notice (Section 5 & 6, DPDP Act 2023)
- **Bounded Usage:** All data collected (GSTIN, trade name, business premises, owner identity) is processed solely for telecom service onboarding, credit evaluation, and regulatory KYC compliance.
- **Affirmative Consent Gate:** Document ingestion endpoints (`POST /api/kyc/upload`) reject all requests (HTTP 403 Forbidden) unless explicit prior consent is logged in the session store (`POST /api/consent`).
- **Right to Withdraw (Section 6(4)):** Business owners can withdraw consent at any time via `DELETE /api/consent/{session_id}`. Withdrawal immediately removes the consent record and purges any verified KYC results.

### B. Data Minimization & Masking
- **Identifier Masking (`backend.app.validators.mask_id`):**
  - Full 15-character GSTINs and 10-digit phone numbers are never stored in plain text in logs or public UI.
  - GSTIN is displayed as: `32TT••••••••1ZW`
  - Phone is logged as: `98••••••10`
- **Zero Hallucination AI:** The Generative AI advisor never stores, remembers, or transmits unmasked tax identifiers or sensitive personal documents into LLM model weights.

### C. Storage Limitation & Ephemeral Processing
- **Ephemeral Uploads:** Uploaded document images are received into isolated system temporary directories (`udyam_kyc_uploads`) with randomly generated filenames (`uuid4`), preventing path-traversal attacks.
- **Immediate Cleanup:** Temporary image files are purged immediately following OCR feature extraction.
- **90-Day Retention Policy:** Activated accounts retain proof-of-verification markers; raw images are permanently purged 90 days after service activation.

### D. Access Control & Review Integrity
- **Admin-Gated Review Queue (`/api/kyc/review`):** Flagged or low-confidence documents (confidence 60–79%) are routed to an administrative queue requiring an `X-Admin-Key` header with constant-time HMAC comparison (`hmac.compare_digest`). Reviewer endpoints stay closed (HTTP 403) unless `ADMIN_API_KEY` is explicitly configured.
- **Strict CORS & Upload Guardrails:** Uploads are strictly capped at 8 MB with a content-type allowlist (PNG, JPEG, PDF) and session ID regex validation (`^[A-Za-z0-9_-]{1,64}$`).

---

## 3. Compliance Summary Table

| DPDP Act Requirement | Implementation in Udyam Digital | Code Reference |
|---|---|---|
| **Notice & Purpose Specification** | Contextual consent prompt before document upload | `POST /api/consent` |
| **Consent Withdrawal** | Dedicated withdrawal endpoint with instant KYC erasure | `DELETE /api/consent/{session_id}` |
| **Data Minimization** | PII masking for identifiers in logs and UI | `backend/app/validators.py:mask_id` |
| **Storage Security** | Ephemeral UUID-based temp files, unlinked after OCR | `backend/app/main.py:upload_kyc_document` |
| **Access Restriction** | Constant-time HMAC admin key header for manual reviews | `backend/app/main.py:require_admin` |
| **Audit Logging** | Masked phone and purpose logging for regulatory tracking | `backend/app/sessions.py` |
