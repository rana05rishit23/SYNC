"""Mock registry services for GSTIN and Udyam lookups.

Provides deterministic registry verification, checksum repair resolution,
and status checks (Active vs Cancelled/Suspended).
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

from .validators import is_valid_gstin, is_valid_udyam, repair_gstin, repair_gstin_structured, mask_id

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class GSTINRecord:
    gstin: str
    legal_name: str
    trade_name: str
    constitution: str
    principal_place_of_business: str
    state: str
    registration_date: str
    status: str       # 'Active' | 'Cancelled' | 'Suspended'

    @property
    def is_active(self) -> bool:
        return self.status == "Active"

    @property
    def masked_gstin(self) -> str:
        return mask_id(self.gstin)


@dataclass(frozen=True)
class UdyamRecord:
    udyam_number: str
    enterprise_name: str
    enterprise_type: str   # 'Micro' | 'Small' | 'Medium'
    major_activity: str    # 'Services' | 'Trading' | 'Manufacturing'
    nic_description: str
    state: str
    city: str
    date_of_registration: str

    @property
    def masked_udyam(self) -> str:
        return mask_id(self.udyam_number)


@lru_cache(maxsize=1)
def load_gstin_registry(path: Path | None = None) -> dict[str, GSTINRecord]:
    p = path or DATA_DIR / "mock_gstin_registry.csv"
    out: dict[str, GSTINRecord] = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["gstin"]] = GSTINRecord(
                gstin=r["gstin"],
                legal_name=r["legal_name"],
                trade_name=r["trade_name"],
                constitution=r["constitution"],
                principal_place_of_business=r["principal_place_of_business"],
                state=r["state"],
                registration_date=r["registration_date"],
                status=r["status"],
            )
    return out


@lru_cache(maxsize=1)
def load_udyam_registry(path: Path | None = None) -> dict[str, UdyamRecord]:
    p = path or DATA_DIR / "mock_udyam_registry.csv"
    out: dict[str, UdyamRecord] = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["udyam_number"]] = UdyamRecord(
                udyam_number=r["udyam_number"],
                enterprise_name=r["enterprise_name"],
                enterprise_type=r["enterprise_type"],
                major_activity=r["major_activity"],
                nic_description=r["nic_description"],
                state=r["state"],
                city=r["city"],
                date_of_registration=r["date_of_registration"],
            )
    return out


def lookup_gstin(gstin: str, registry: dict[str, GSTINRecord] | None = None) -> GSTINRecord | None:
    """Look up a normalized GSTIN in the registry."""
    clean = (gstin or "").strip().upper()
    reg = registry or load_gstin_registry()
    return reg.get(clean)


def lookup_udyam(udyam: str, registry: dict[str, UdyamRecord] | None = None) -> UdyamRecord | None:
    """Look up a normalized Udyam registration number in the registry."""
    clean = (udyam or "").strip().upper()
    reg = registry or load_udyam_registry()
    return reg.get(clean)


def verify_gstin_with_repair(
    raw_gstin: str, registry: dict[str, GSTINRecord] | None = None
) -> tuple[GSTINRecord | None, list[str]]:
    """Two-layer verification:
    1. Direct format + checksum + registry lookup.
    2. If format fails or not found, attempt OCR confusable-character repair
       and cross-check candidates against the registry.
    Returns: (record or None, audit_notes).
    """
    clean = (raw_gstin or "").strip().upper()
    reg = registry or load_gstin_registry()
    notes: list[str] = []

    # 1. Direct check
    if is_valid_gstin(clean):
        rec = reg.get(clean)
        if rec:
            notes.append(f"Direct match in registry with status: {rec.status}")
            return rec, notes
        else:
            notes.append("Checksum valid, but not found in mock GSTIN registry")
            return None, notes

    # 2. Repair attempt for OCR misreads (character substitutions + structural position repair)
    candidates = repair_gstin(clean)
    if not candidates:
        candidates = repair_gstin_structured(clean)
    if candidates:
        notes.append(f"Repaired OCR confusions into candidate(s): {candidates}")
        for cand in candidates:
            rec = reg.get(cand)
            if rec:
                notes.append(f"Repaired GSTIN matched registry: {mask_id(cand)} ({rec.status})")
                return rec, notes
        notes.append("None of the repaired checksum-valid candidates exist in registry")
    else:
        notes.append("Could not repair GSTIN format/checksum")

    return None, notes
