# Engineering decision log (ADRs, newest last)

## ADR-001: Deterministic core, LLM at the edges
The advisor (LLM) only *converses* and *proposes*. Eligibility, dependencies and pricing are computed by
`backend/app/catalogue.py`. Every LLM-proposed bundle passes `validate_bundle()` before the customer sees it.
**Why:** the catalogue has hard rules (tier, size, GSTIN, dependencies). An LLM that hallucinates a leased line
for a kirana store is a compliance and trust failure. **Evidence:** see ADR-002.

## ADR-002: Audit data before designing (scripts/data_audit.py)
Findings that shape the design:
1. Registries are internally consistent: 100% of lead GSTINs/Udyam numbers exist in the mock registries; all GSTINs pass checksum.
2. 268 registry GSTINs are Cancelled (167) or Suspended (101). KYC must *block or re-route* these, not just check "exists".
3. **56 of 368 (15%) historically activated bundles break catalogue rules** (24 tier, 12 GSTIN missing, 10 GSTIN inactive,
   9 missing dependency, 8 below min employees). The manual process sells ineligible bundles -> business-case point:
   rules-in-code removes this class of failure.
4. All 368 historical bundle values sit within catalogue min-max price ranges -> our pricing policy should also stay in range.

## ADR-003: Smart KYC is layered, because OCR alone failed on GST certificates
Probe (scripts/ocr_probe.py): Tesseract reads Udyam cert, signatory ID and address proof cleanly, but the GST
certificate (diagonal watermark over monospace text) returned labels only; after thresholding, labels were fine
but the GSTIN itself came back garbled (`19E0GKH4} SyiZ,`).
Layers: (1) OCR + preprocessing -> (2) checksum + confusable-character repair (`repair_gstin`, tested) ->
(3) registry cross-check by legal name/address (fuzzy) -> (4) human-review queue.
*Amended in ADR-006:* the earlier "vision-LLM fallback" layer was dropped because Groq's production model list
has no vision model; we will not depend on one.
Never accept OCR text as truth; accept only what passes checksum AND registry.

## ADR-004: Privacy by construction
Identifiers are masked in logs/UI via `mask_id`. Consent screen precedes any document upload. Retention and purpose
limitation to be documented in the 1-page privacy note.

## ADR-005: Pricing = one monthly price shaped by contract length (chosen by the team)
`price = max - (max-min) * pct`, pct = 40% for 12 months, 80% for 24 months (`backend/app/pricing.py`).
Evidence (scripts/data_audit.py, scripts/pricing_backtest.py):
- Historical prices sit near the midpoint of each range (0.48 at 12m, 0.51 at 24m): the field force has **no term logic**,
  so this is a new policy, not something copied from the data.
- Setup fees are **per unit** (matched 100% of 368 sales); historical terms were 6/12/24 months only, so we offer 12 and 24.
- Backtest on 312 rule-valid historical bundles: our price = 97.8% of what was charged (12m: +3.5%, 24m: -8.8%).
  Roughly revenue-neutral. Micro segment is ~+3% at 12m: watch price sensitivity (A2 says "very high").
- 20% headroom above the floor is kept for partner or promo offers. Prices never leave the catalogue range (tested).

## ADR-006: Groq as the LLM provider
Model ids checked against console.groq.com/docs/models (Nov 2026): production models are `openai/gpt-oss-120b`,
`openai/gpt-oss-20b`, `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, plus `whisper-large-v3(-turbo)` for speech.
- Advisor: `openai/gpt-oss-120b` (tool use, 131K context). Alternate: `llama-3.3-70b-versatile`.
  **To evaluate on Hindi/regional-language quality before locking in.**
- Fast tasks (language detection, intent): `openai/gpt-oss-20b`.
- Voice stretch goal: `whisper-large-v3-turbo`.
- Models are env-configurable (`config.py`), so a deprecation is a one-line change. The LLM sits behind a small client
  interface and is injected, so the advisor is unit-tested with a fake client (no network, no key needed).
- No Groq vision model is relied on (see ADR-003).

## ADR-007: Segment bundle templates & adaptive hybrid onboarding flow
Based on analysis of the 368 historical activated sales across 4 primary segments:
1. Micro retail & services (`micro`): P01/P04 Broadband + P07 Postpaid + P19 Cloud POS, toggleable CCTV (P23) & Guest Wi-Fi (P22).
2. Professional offices (`prof`): P02 Fibre 300 + P08 Postpaid + P16 Cloud Email/Suite scaled to employees + P11 Static IP.
3. Small multi-site businesses (`multi`): P02 per site + P09 Failover per site + P10 SD-WAN (if >=3 sites) + P23 CCTV.
4. Small manufacturers (`mfg`): P05 Leased line (if eligible) or P03 1G Fibre + P21 Firewall + P08 + P16 + P24 IoT.
- Policy: Flow uses adaptive quick-choice chips (business type & location) on turn 1, auto-matches template, asks 1-2 focused questions, and immediately generates a transparent quote with 12m vs 24m term comparisons.
- Every template bundle is verified against `validate_bundle()` and auto-corrects any rule violations before output.

## ADR-008: Advisor tool calling & deterministic validation gate
The Advisor implements Groq OpenAI-compatible tool calling with 4 core tools:
1. `tool_detect_and_recommend_template`: Maps business profile to segment template and chips.
2. `tool_get_eligible_products`: Queries deterministic rules engine for available items.
3. `tool_check_gstin`: Verifies format, repair, and mock registry status (Active/Suspended/Cancelled).
4. `tool_propose_and_quote_bundle`: Computes verified pricing for 12m/24m and available add-ons.
- Sandbox & test compatibility: `MockAdvisorLLMClient` allows 100% offline unit and integration testing without Groq API keys or external network dependencies.
- Privacy & audit: Masked identifiers in logs (`mask_id`), explicit consent gate before document/KYC handling.

## ADR-009: Language-Agnostic Translation at the Edges
Reference: Engineering Note "Language-Agnostic Advisor: Supporting Any Language with an LLM" (8 Nov 2026).
- Zero-engineering scalability: Instead of maintaining fixed dictionaries for India's 22+ languages, canonical English + structured data is preserved in the core. The LLM translates only at the edges:
  1. Language Detection: Layer A (Unicode script check, free/instant) + Layer B (Fast model gpt-oss-20b for Devanagari/Latin disambiguation).
  2. NLU Understanding: Free text in any language -> canonical `BusinessProfileDraft` (English enums).
  3. Deterministic Core: Rules and pricing run on language-free codes.
  4. NLG Rendering: Generates message in owner's language from verified facts only.
  5. Number Faithfulness Check (`faithful()`): Strict regex/Unicode check ensuring no hallucinated prices or terms reach the owner. Unfaithful translations fall back to safe English templates.
  6. Interface i18n ("Translate once, cache forever"): Single English source (`locales/en.json`) translated in one batched call with placeholder preservation (`{setup}`, `{price}`) and cached in SQLite (`i18n_cache`).

## ADR-010: Smart KYC Multi-Layer Verification Architecture

- **Context:** Groq has no production vision model, and physical Indian business documents (GST certificates, Udyam certificates) frequently suffer from diagonal background watermarks, scan skew, and monospace character confusion (0↔O, 8↔B, 5↔S, 2↔Z). Furthermore, DPDP Act 2023 mandates strict user consent and purpose specification before document handling.
- **Decision:**
  - **Layer 0 (Consent Gate):** Verification is strictly blocked (`HTTP 403` / `ValueError`) unless explicit consent with purpose specification has been recorded for the session.
  - **Layer 1 (OCR Extraction):** Image preprocessing with adaptive thresholding to suppress watermarks, followed by Tesseract OCR with mean confidence scoring. Lenient regex and token-level heuristics extract GSTIN, Udyam, legal/trade name, and address.
  - **Layer 2 (Checksum & Confusable Repair):** Deterministic Mod-36 checksum check; if invalid, `repair_gstin()` systematically attempts up to 2 confusable character swaps.
  - **Layer 3 (Registry Cross-Check & Fuzzy Name Match):** Cross-checks candidates against mock GSTIN/Udyam registries. If found Cancelled or Suspended, the document is immediately rejected with an audit trail. If Active, `kyc_matcher.py` runs dual Jaccard token-set and sequence similarity (>=0.8 = match, 0.6-0.79 = review, <0.6 = mismatch).
  - **Layer 4 (Human Review Queue):** Any document with low OCR confidence (<60%), ambiguous name matching (0.6-0.79), or missing identifiers is placed in the review queue (`/api/kyc/review`).
- **Safety & Privacy:** All identifiers are masked (`mask_id`) in all logs and audit trails.

## ADR-011: API Hardening, Access Control, and DPDP Consent Lifecycle

- **Context:** Penetration analysis identified path-traversal vulnerabilities in file uploads via crafted session IDs, unbounded upload payloads, open reviewer endpoints, and absence of a programmatic consent withdrawal mechanism required under Section 6(4) of the DPDP Act 2023.
- **Decision:**
  - Restrict `session_id` strictly to regex `^[A-Za-z0-9_-]{1,64}$` using Pydantic `StringConstraints`.
  - Server generates ephemeral UUID-based filenames inside system temp directory (`udyam_kyc_uploads`); nothing user-controlled enters filesystem paths. Temp files are unlinked immediately after OCR.
  - Enforce upload limits: 8 MB ceiling (`HTTP 413`) and content-type allowlist (PNG, JPEG, PDF; `HTTP 415`).
  - Secure reviewer triage endpoints (`/api/kyc/review`) behind `X-Admin-Key` header with constant-time HMAC check; endpoints remain closed (`HTTP 403`) unless `ADMIN_API_KEY` is explicitly configured.
  - Implement consent withdrawal endpoint `DELETE /api/consent/{session_id}` which revokes consent and immediately erases associated KYC results.
  - Restrict CORS middleware to explicit origin list without credentials.

## ADR-012: Chat-First Mobile UI with Quick-Reply Suggestions

- **Context:** The standalone UI used static forms. The multilingual advisor backend returns rich structured responses (`quick_chips`, `detected_language`, `quote`, `bundle`).
- **Decision:** Wire `frontend/index.html` to backend REST APIs using a chat-first interface:
  - Dynamic conversational thread supporting free-text typing in 5+ Indic languages and quick-reply chips.
  - Real-time display of detected language and confidence.
  - Clear "Live AI (Groq)" vs "Demo Mode (Mock AI)" status badge derived from `/api/health`.
  - Seamless handoff from advisor bundle recommendation to interactive quote customization, live GSTIN verification, and multipart KYC file upload.

## Open decisions (next)
- Submission deliverables: 10-slide business case deck and 3-minute demo video script.


