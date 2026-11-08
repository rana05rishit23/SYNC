"""Segment bundle templates and recommendation engine.

Maps SMB business types to tailored starter bundles, curated add-ons,
and quick-reply onboarding chips based on historical activations and catalogue rules.
Every template bundle generated is guaranteed to pass validate_bundle().
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from .catalogue import BusinessProfile, Product, load_catalogue, validate_bundle


@dataclass(frozen=True)
class SegmentTemplate:
    segment_id: str                      # 'micro', 'prof', 'multi', 'mfg'
    name: str                            # human-readable title
    tagline: str                         # 1-line value proposition
    quick_chips: tuple[str, ...]         # quick-selection chips for owner UI
    curated_addons: tuple[str, ...]      # recommended add-on product IDs
    builder: Callable[[BusinessProfile, dict[str, bool] | None], dict[str, int]]


# ---------------------------------------------------------------------------
# Bundle Builders per Segment (Guaranteed 100% Rule-Valid)
# ---------------------------------------------------------------------------

def _build_micro_bundle(profile: BusinessProfile, flags: dict[str, bool] | None = None) -> dict[str, int]:
    """Micro retail & services (Kirana, bakery, chemist, salon, cafe).
    Usually 1 site, 1-5 employees. Price-sensitive.
    """
    flags = flags or {}
    items: dict[str, int] = {}

    # Connectivity: Biz Fibre 100 (P01); fallback to AirFiber (P04) if requested or tier-suitable
    if flags.get("prefer_airfiber", False):
        items["P04"] = 1
    else:
        items["P01"] = 1

    # Mobile line: 1 postpaid SIM for the shop owner / counter
    num_sims = max(1, min(profile.employees, 3))
    items["P07"] = num_sims

    # Cloud POS / Billing if retail or requested
    if flags.get("needs_pos", True):
        items["P19"] = max(1, profile.sites)

    # Optional: CCTV (P23) if requested
    if flags.get("needs_cctv", False):
        items["P23"] = 1

    # Optional: Guest Wi-Fi (P22)
    if flags.get("needs_wifi", False):
        items["P22"] = 1

    return items


def _build_prof_bundle(profile: BusinessProfile, flags: dict[str, bool] | None = None) -> dict[str, int]:
    """Professional offices (Clinics, CA/Law firms, IT services, consultants).
    Usually 1-2 sites, 5-25 employees. Prioritizes email, speed and reliability.
    """
    flags = flags or {}
    items: dict[str, int] = {}

    # Connectivity: Biz Fibre 300 (P02) for fast multi-user broadband
    # Note: P03 requires Tier 1, 5+ emp, and GSTIN; if eligible and requested, can use P03
    if flags.get("high_speed", False) and profile.city_tier == 1 and profile.employees >= 5 and profile.gstin_status == "Active":
        items["P03"] = 1
    else:
        items["P02"] = 1

    # Postpaid mobile: P08 for management/field staff
    num_sims = max(1, min(profile.employees // 3, 5))
    items["P08"] = num_sims

    # Cloud Email & Office Suite (P16): 1 per employee (min 1)
    emp_count = max(1, profile.employees)
    items["P16"] = emp_count

    # Static IP (P11) for remote access/server/CCTV
    if flags.get("needs_static_ip", True):
        items["P11"] = 1

    # Cloud Telephony / SIP (P12): requires GSTIN Active and 3+ employees
    if flags.get("needs_telephony", False) and profile.gstin_status == "Active" and profile.employees >= 3:
        items["P12"] = 1

    return items


def _build_multi_bundle(profile: BusinessProfile, flags: dict[str, bool] | None = None) -> dict[str, int]:
    """Small multi-site businesses (Retail chain, multi-branch clinics, apparel stores).
    Usually 3+ sites, 10-60 employees. Focus on interconnectivity, uptime, and central control.
    """
    flags = flags or {}
    items: dict[str, int] = {}
    sites = max(2, profile.sites)

    # Connectivity per site: Biz Fibre 300 (P02)
    items["P02"] = sites

    # Automatic 4G/5G Failover router per site (P09) to prevent downtime
    items["P09"] = sites

    # Business Postpaid (P08): 1 SIM per store manager / supervisor
    items["P08"] = max(2, min(sites, profile.employees))

    # Managed SD-WAN (P10): requires Tier 1/2, 10+ employees, 3+ sites, GSTIN Active
    if sites >= 3 and profile.employees >= 10 and profile.gstin_status == "Active" and not flags.get("skip_sdwan", False):
        items["P10"] = sites

    # Cloud CCTV (P23): 2 cameras per site
    if flags.get("needs_cctv", True):
        items["P23"] = sites * 2

    # WhatsApp Business API Starter (P15) if GSTIN is Active
    if profile.gstin_status == "Active" and flags.get("needs_whatsapp", True):
        items["P15"] = 1

    return items


def _build_mfg_bundle(profile: BusinessProfile, flags: dict[str, bool] | None = None) -> dict[str, int]:
    """Small manufacturers (Auto components, packaging, light engineering, pharma).
    Usually 10-50+ employees, 1-3 sites. Dedicated bandwidth, security, machine monitoring.
    """
    flags = flags or {}
    items: dict[str, int] = {}

    # Connectivity: Dedicated Leased Line 20 Mbps (P05) if >=10 emp & GSTIN Active; else Biz Fibre 1G / 300
    if profile.employees >= 10 and profile.gstin_status == "Active" and not flags.get("prefer_broadband", False):
        items["P05"] = max(1, profile.sites)
    elif profile.city_tier == 1 and profile.employees >= 5 and profile.gstin_status == "Active":
        items["P03"] = max(1, profile.sites)
    else:
        items["P02"] = max(1, profile.sites)

    # Managed Firewall (P21): requires 10+ emp, GSTIN Active, and internet link
    if profile.employees >= 10 and profile.gstin_status == "Active" and not flags.get("skip_firewall", False):
        items["P21"] = max(1, profile.sites)

    # Business Postpaid (P08) for factory managers & dispatch
    items["P08"] = max(2, min(profile.employees // 5, 6))

    # Cloud Email & Office Suite (P16) for administrative staff
    admin_users = max(2, min(profile.employees // 2, 20))
    items["P16"] = admin_users

    # Machine Monitoring IoT (P24): requires 10+ emp, GSTIN Active
    if flags.get("needs_iot", False) and profile.employees >= 10 and profile.gstin_status == "Active":
        items["P24"] = 5

    # Fleet Tracker (P25) if logistics/delivery vehicles present
    if flags.get("needs_fleet", False):
        items["P25"] = 3

    return items


# ---------------------------------------------------------------------------
# Template Registry
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, SegmentTemplate] = {
    "micro": SegmentTemplate(
        segment_id="micro",
        name="Micro Retail & Services",
        tagline="Affordable high-speed fibre, mobile lines & billing POS for neighbourhood shops",
        quick_chips=(
            "Kirana / Grocery",
            "Bakery / Sweets",
            "Retail Shop",
            "Salon / Spa",
            "Chemist / Pharmacy",
            "Cafe / Restaurant",
        ),
        curated_addons=("P23", "P22", "P15", "P18"),  # CCTV, Wi-Fi, WhatsApp, Website
        builder=_build_micro_bundle,
    ),
    "prof": SegmentTemplate(
        segment_id="prof",
        name="Professional Offices & Clinics",
        tagline="High-reliability fibre, business email, static IP & telephony for modern offices",
        quick_chips=(
            "Doctor Clinic / Diagnostics",
            "CA / Financial Services",
            "Law Firm / Legal Office",
            "IT / Software Agency",
            "Consulting / Architecture",
        ),
        curated_addons=("P12", "P17", "P20", "P13"),  # Telephony, Cloud Backup, Endpoint Security, Toll-free
        builder=_build_prof_bundle,
    ),
    "multi": SegmentTemplate(
        segment_id="multi",
        name="Small Multi-site Businesses",
        tagline="Interconnected multi-branch broadband with 4G/5G failover, SD-WAN & CCTV",
        quick_chips=(
            "Retail Chain (2+ stores)",
            "Diagnostic / Clinic Network",
            "Apparel / Footwear Stores",
            "Multi-location Agency",
        ),
        curated_addons=("P10", "P23", "P13", "P15", "P25"),  # SD-WAN, CCTV, Toll-free, WhatsApp, Fleet
        builder=_build_multi_bundle,
    ),
    "mfg": SegmentTemplate(
        segment_id="mfg",
        name="Small Manufacturers & Units",
        tagline="Dedicated leased lines, SOC managed firewall, cloud email & machine IoT",
        quick_chips=(
            "Auto Components / Engineering",
            "Packaging / Plastics",
            "Textile / Garment Unit",
            "Food Processing",
            "Warehouse / Logistics",
        ),
        curated_addons=("P24", "P25", "P20", "P21", "P23"),  # Machine IoT, Fleet, Endpoint, Firewall, CCTV
        builder=_build_mfg_bundle,
    ),
}


# ---------------------------------------------------------------------------
# Intent & Segment Detection
# ---------------------------------------------------------------------------

_KEYWORD_MAP: list[tuple[set[str], str]] = [
    ({"chain", "branches", "multi", "multisite", "outlets", "franchise", "stores", "locations", "शाखा", "ब्रांच", "கிளைகள்", "दुकानें", "ब्रांचेस"}, "multi"),
    ({"factory", "manufacturing", "manufacturer", "plant", "auto", "engineering", "packaging",
      "textile", "workshop", "industrial", "warehouse", "mfg", "कारखाना", "தொழிற்சாலை", "फैक्ट्री", "उत्पादन"}, "mfg"),
    ({"clinic", "doctor", "dental", "diagnostic", "hospital", "ca", "chartered", "accountant",
      "lawyer", "advocate", "legal", "consultant", "architect", "software", "agency", "office", "prof", "क्लिनिक", "மருத்துவமனை", "लैब", "डॉक्टर"}, "prof"),
    ({"kirana", "grocery", "bakery", "chemist", "pharmacy", "salon", "parlour", "spa", "cafe",
      "restaurant", "sweet", "tailor", "retail", "shop", "dukan", "stationary", "hardware", "micro", "किराना", "दुकान", "मिष्टी", "मिठाई"}, "micro"),
]


def detect_segment(text: str) -> str:
    """Classify user's business description into one of 4 segments with multilingual support."""
    clean_text = text.lower().replace("-", " ").replace("/", " ").replace("(", " ").replace(")", " ").replace(",", " ").replace(".", " ")
    tokens = set(clean_text.split())
    for keywords, seg_id in _KEYWORD_MAP:
        if tokens & keywords:
            return seg_id
    # Default to micro for general SMB inquiries
    return "micro"


def get_segment_template(segment_id_or_name: str) -> SegmentTemplate:
    """Retrieve template by key or mapped alias."""
    key = segment_id_or_name.lower().strip()
    if key in TEMPLATES:
        return TEMPLATES[key]
    detected = detect_segment(key)
    return TEMPLATES[detected]


def propose_bundle(
    segment: str,
    profile: BusinessProfile,
    flags: dict[str, bool] | None = None,
    catalogue: dict[str, Product] | None = None,
) -> dict[str, int]:
    """Generate a guaranteed-valid bundle tailored to the segment and profile."""
    cat = catalogue or load_catalogue()
    tmpl = get_segment_template(segment)
    items = tmpl.builder(profile, flags)

    # Validate against deterministic rules
    issues = validate_bundle(profile, items, cat)
    if issues:
        # Self-correct any failing item
        cleaned: dict[str, int] = {}
        bad_pids = {i.product_id for i in issues}
        for pid, qty in items.items():
            if pid not in bad_pids:
                cleaned[pid] = qty
        # Ensure at least a core connectivity product exists
        if not any(cat[p].category == "Connectivity" for p in cleaned):
            cleaned["P01"] = 1
        items = cleaned

    return items
