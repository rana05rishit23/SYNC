"""Case-level KYC: decide an applicant from all four documents together (ADR-013).

A single document can look fine while the case is wrong: a cancelled GSTIN, a trade name that differs
between certificates, an ID that has expired, an electricity bill older than 90 days, a missing Udyam
certificate. Findings carry a severity and the case outcome is the most severe one:

    rejected    a hard failure the owner cannot fix by re-uploading (cancelled or suspended GSTIN, unknown business)
    reupload    a document-level problem the owner can fix (blurred scan, expired ID, stale bill)
    incomplete  a required document is missing
    review      the documents disagree in a way a person should look at
    approved    nothing above; 'info' findings may remain (e.g. ask the owner to confirm a recognised GSTIN)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from .docscan import ScannedDoc
from .kyc_matcher import token_set_similarity as sim
from .registries import GSTINRecord, UdyamRecord, load_gstin_registry, load_udyam_registry
from .validators import mask_id

TODAY = date(2026, 7, 15)           # the dataset's reference date for every date-based check
MAX_BILL_AGE_DAYS = 90
NAME_OK = 0.85
REQUIRED = ("gst_certificate", "udyam_certificate", "signatory_id", "address_proof")
_RANK = {"reject": 4, "reupload": 3, "incomplete": 2, "review": 1, "info": 0}
_STATUS = {"reject": "rejected", "reupload": "reupload", "incomplete": "incomplete", "review": "review"}


@dataclass
class Finding:
    code: str
    severity: str          # reject | reupload | incomplete | review | info
    message: str
    doc: str | None = None


@dataclass
class CaseDecision:
    status: str
    findings: list[Finding] = field(default_factory=list)
    gstin_masked: str | None = None
    udyam_masked: str | None = None
    confirm_gstin_with_owner: bool = False


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _pincode(addr: str | None) -> str | None:
    m = re.search(r"(\d{6})\s*$", (addr or "").strip())
    return m.group(1) if m else None


def _proprietor_parts(legal_name: str | None) -> tuple[str | None, str | None]:
    """'Neha Kaur (Prop. Vindhya Tours & Travels)' -> ('Neha Kaur', 'Vindhya Tours & Travels')."""
    m = re.match(r"^(.*?)\s*\(\s*Prop\.?\s*(.*?)\s*\)\s*$", legal_name or "", re.I)
    return (m.group(1), m.group(2)) if m else (None, None)


def _resolve_gstin(doc: ScannedDoc, reg: dict[str, GSTINRecord]) -> tuple[GSTINRecord | None, str]:
    """Returns (record, how) where how is 'exact', 'reconciled' or 'none'."""
    cands = [c for c in (doc.fields.get("gstin_candidates") or "").split(",") if c in reg]
    if len(cands) == 1:
        return reg[cands[0]], "exact"
    # Reconcile: the registry entry with the same business name AND the same printed pincode, and an OCR
    # read that is close to it. All identity attributes agree; only the monospace digits were noisy.
    name, pin = doc.fields.get("trade_name"), _pincode(doc.fields.get("address"))
    raw = re.sub(r"[^0-9A-Z]", "", (doc.fields.get("gstin_raw") or "").upper())
    if name and pin and raw:
        hits = [r for r in reg.values() if sim(r.trade_name, name) >= 0.9 and _pincode(r.principal_place_of_business) == pin]
        if len(hits) == 1 and _lev(raw, hits[0].gstin) <= 5:
            return hits[0], "reconciled"
    return None, "none"


def _find_udyam(doc: ScannedDoc, reg: dict[str, UdyamRecord]) -> UdyamRecord | None:
    """The specimen certificates do not print the Udyam number, so match on name + city + state."""
    name = doc.fields.get("enterprise_name")
    hits = [r for r in reg.values()
            if name and sim(r.enterprise_name, name) >= 0.9
            and sim(r.city, doc.fields.get("city")) >= 0.9 and sim(r.state, doc.fields.get("state")) >= 0.9]
    if len(hits) > 1:   # the normaliser drops 'LLP'/'Pvt Ltd', so near-twins can collide: prefer the exact name
        hits = [r for r in hits if r.enterprise_name.strip().lower() == (name or "").strip().lower()]
    return hits[0] if len(hits) == 1 else None


def verify_case(docs: list[ScannedDoc], today: date = TODAY,
                gst_reg: dict[str, GSTINRecord] | None = None,
                udyam_reg: dict[str, UdyamRecord] | None = None) -> CaseDecision:
    gst_reg = gst_reg if gst_reg is not None else load_gstin_registry()
    udyam_reg = udyam_reg if udyam_reg is not None else load_udyam_registry()
    by = {d.doc_type: d for d in docs}
    out: list[Finding] = []
    add = lambda code, sev, msg, doc=None: out.append(Finding(code, sev, msg, doc))

    for t in REQUIRED:
        if t not in by:
            add("MISSING_DOC", "incomplete", f"Missing required document: {t.replace('_', ' ')}.", t)
    for d in docs:
        if not d.readable:
            add("BLURRY_DOC", "reupload", f"The {d.doc_type.replace('_', ' ')} is too blurred to read. Please upload a sharper photo.", d.doc_type)

    gst, decision = by.get("gst_certificate"), CaseDecision("approved")
    trade = None
    if gst and gst.readable:
        trade = gst.fields.get("trade_name")
        rec, how = _resolve_gstin(gst, gst_reg)
        if rec is None:
            add("GSTIN_UNRESOLVED", "review", "The GSTIN could not be matched to the registry.", "gst_certificate")
        else:
            decision.gstin_masked = mask_id(rec.gstin)
            if how == "reconciled":
                decision.confirm_gstin_with_owner = True
                add("GSTIN_CONFIRM", "info", f"Recognised GSTIN {mask_id(rec.gstin)} from name and pincode; ask the owner to confirm.", "gst_certificate")
            if rec.status != "Active":
                add(f"GSTIN_{rec.status.upper()}", "reject", f"GSTIN {mask_id(rec.gstin)} is {rec.status.lower()} in the registry.", "gst_certificate")
            if sim(rec.trade_name, trade) < NAME_OK:
                add("GST_NAME_VS_REGISTRY", "review", "Trade name on the certificate differs from the registry.", "gst_certificate")
        person, prop_biz = _proprietor_parts(gst.fields.get("legal_name"))
        if prop_biz and sim(prop_biz, trade) < NAME_OK:
            add("GST_LEGAL_VS_TRADE", "review", f"Legal name says '{prop_biz}' but trade name is '{trade}'.", "gst_certificate")
    else:
        person = None

    udy = by.get("udyam_certificate")
    if udy and udy.readable:
        rec_u = _find_udyam(udy, udyam_reg)
        if rec_u is None:
            add("UDYAM_NOT_FOUND", "review", "No Udyam registry entry matches this enterprise, city and state.", "udyam_certificate")
        else:
            decision.udyam_masked = mask_id(rec_u.udyam_number)
        if trade and sim(udy.fields.get("enterprise_name"), trade) < NAME_OK:
            add("UDYAM_NAME_MISMATCH", "review", "Enterprise name on the Udyam certificate differs from the GST trade name.", "udyam_certificate")

    sid = by.get("signatory_id")
    if sid and sid.readable:
        vt = sid.fields.get("valid_till")
        if vt and date.fromisoformat(vt) < today:
            add("ID_EXPIRED", "reupload", f"Signatory ID expired on {vt}.", "signatory_id")
        if trade and sid.fields.get("business") and sim(sid.fields["business"], trade) < NAME_OK:
            add("ID_BUSINESS_MISMATCH", "review", "The ID names a different business than the GST certificate.", "signatory_id")
        if person and sid.fields.get("name") and sim(sid.fields["name"], person) < NAME_OK:
            add("ID_NAME_MISMATCH", "review", "The proprietor on the GST certificate is not the person on the ID.", "signatory_id")

    adr = by.get("address_proof")
    if adr and adr.readable:
        bd = adr.fields.get("bill_date")
        if bd:
            d0 = date.fromisoformat(bd)
            if d0 < today - timedelta(days=MAX_BILL_AGE_DAYS):
                add("ADDR_STALE", "reupload", f"Address proof is dated {bd}, older than {MAX_BILL_AGE_DAYS} days.", "address_proof")
            elif d0 > today:
                add("ADDR_FUTURE_DATED", "review", f"Address proof is dated in the future ({bd}).", "address_proof")
        if trade and adr.fields.get("billed_to") and sim(adr.fields["billed_to"], trade) < NAME_OK:
            add("ADDR_NAME_MISMATCH", "review", "The bill is in a different business name.", "address_proof")
        if gst and gst.readable and _pincode(adr.fields.get("address")) != _pincode(gst.fields.get("address")):
            add("ADDR_MISMATCH", "review", "Address on the bill differs from the GST principal place of business.", "address_proof")

    worst = max((f.severity for f in out), key=lambda s: _RANK[s], default="info")
    decision.findings = out
    decision.status = _STATUS.get(worst, "approved")
    return decision
