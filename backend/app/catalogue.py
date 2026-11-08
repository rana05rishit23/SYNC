"""Product catalogue + eligibility rules engine.

This is the single source of truth for 'can this business buy this?'.
The GenAI advisor may *propose* bundles, but every proposal is passed through
validate_bundle() before the customer sees it. The LLM never overrides these rules.
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Product:
    product_id: str
    category: str
    name: str
    description: str
    unit: str
    price_min: int
    price_max: int
    setup_fee: int
    min_contract_months: int
    city_tiers: frozenset[int]
    min_employees: int
    min_sites: int
    requires_gstin: bool
    requires_one_of: frozenset[str]
    install_days: int
    target_segments: frozenset[str]


@dataclass(frozen=True)
class BusinessProfile:
    city_tier: int
    employees: int
    sites: int
    gstin_status: str | None = None   # None = no GSTIN; else "Active" | "Suspended" | "Cancelled"


@dataclass(frozen=True)
class Issue:
    product_id: str
    code: str      # TIER | MIN_EMPLOYEES | MIN_SITES | GSTIN_REQUIRED | GSTIN_INACTIVE | MISSING_DEPENDENCY | UNKNOWN_PRODUCT | BAD_QUANTITY
    message: str   # plain-language, safe to show to the owner / feed to the advisor


def _split(s: str, sep: str) -> list[str]:
    return [x for x in (s or "").split(sep) if x]


def load_catalogue(path: Path | None = None) -> dict[str, Product]:
    path = path or DATA_DIR / "product_catalogue.csv"
    out: dict[str, Product] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["product_id"]] = Product(
                product_id=r["product_id"], category=r["category"], name=r["product_name"],
                description=r["description"], unit=r["unit"],
                price_min=int(r["price_min_inr_per_month"]), price_max=int(r["price_max_inr_per_month"]),
                setup_fee=int(r["one_time_setup_inr"]), min_contract_months=int(r["min_contract_months"]),
                city_tiers=frozenset(int(t) for t in _split(r["available_city_tiers"], ";")),
                min_employees=int(r["min_employees"]), min_sites=int(r["min_sites"]),
                requires_gstin=r["requires_gstin"] == "Y",
                requires_one_of=frozenset(_split(r["requires_one_of"], "|")),
                install_days=int(r["typical_install_days"]),
                target_segments=frozenset(_split(r["target_segments"], ";")),
            )
    return out


def check_product(profile: BusinessProfile, p: Product) -> list[Issue]:
    """Rules that depend only on the business profile (not on the rest of the bundle)."""
    issues: list[Issue] = []
    if profile.city_tier not in p.city_tiers:
        issues.append(Issue(p.product_id, "TIER", f"{p.name} is not available in Tier {profile.city_tier} cities yet."))
    if profile.employees < p.min_employees:
        issues.append(Issue(p.product_id, "MIN_EMPLOYEES", f"{p.name} is for businesses with {p.min_employees}+ employees."))
    if profile.sites < p.min_sites:
        issues.append(Issue(p.product_id, "MIN_SITES", f"{p.name} is for businesses with {p.min_sites}+ locations."))
    if p.requires_gstin:
        if profile.gstin_status is None:
            issues.append(Issue(p.product_id, "GSTIN_REQUIRED", f"{p.name} needs a GST registration."))
        elif profile.gstin_status != "Active":
            issues.append(Issue(p.product_id, "GSTIN_INACTIVE",
                                f"{p.name} needs an active GSTIN; yours shows as {profile.gstin_status.lower()}."))
    return issues


def validate_bundle(profile: BusinessProfile, items: dict[str, int], catalogue: dict[str, Product]) -> list[Issue]:
    """Full validation of a proposed bundle {product_id: quantity}. Empty list == valid."""
    issues: list[Issue] = []
    for pid, qty in items.items():
        p = catalogue.get(pid)
        if p is None:
            issues.append(Issue(pid, "UNKNOWN_PRODUCT", f"Unknown product {pid}."))
            continue
        if qty < 1:
            issues.append(Issue(pid, "BAD_QUANTITY", f"Quantity for {p.name} must be at least 1."))
        issues.extend(check_product(profile, p))
        if p.requires_one_of and not (p.requires_one_of & items.keys()):
            needed = " or ".join(catalogue[x].name for x in sorted(p.requires_one_of) if x in catalogue)
            issues.append(Issue(pid, "MISSING_DEPENDENCY", f"{p.name} needs an internet plan in the same bundle ({needed})."))
    return issues


def eligible_products(profile: BusinessProfile, catalogue: dict[str, Product]) -> list[Product]:
    """Products the business could buy on its own profile (dependencies are resolved at bundle level)."""
    return [p for p in catalogue.values() if not check_product(profile, p)]
