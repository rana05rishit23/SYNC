"""Pricing engine: one transparent monthly price per bundle, shaped by contract length.

POLICY (ADR-005): each product has a catalogue price range [min, max]. The unit price is
    price = max - (max - min) * discount_pct / 100
where discount_pct depends on the contract term the owner picks:
    12 months -> 40%   (still above the midpoint of the range)
    24 months -> 80%   (never the absolute floor: 20% of headroom is kept for partner/promo offers)
Prices therefore always stay inside the catalogue range. Setup fees are per unit (verified
against all 368 historical sales). Pure functions: no I/O, no LLM, trivially testable.
"""
from __future__ import annotations
from dataclasses import dataclass

from .catalogue import BusinessProfile, Issue, Product, validate_bundle

TERM_DISCOUNT_PCT: dict[int, int] = {12: 40, 24: 80}
BASELINE_TERM = 12


class InvalidBundle(Exception):
    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("; ".join(i.message for i in issues))


class InvalidTerm(Exception):
    pass


@dataclass(frozen=True)
class QuoteLine:
    product_id: str
    name: str
    unit: str
    quantity: int
    unit_price: int
    monthly: int
    setup_per_unit: int
    setup: int


@dataclass(frozen=True)
class Quote:
    lines: tuple[QuoteLine, ...]
    term_months: int
    monthly_total: int
    one_time_total: int
    first_payment: int              # first month + all setup fees
    contract_value: int             # monthly * term + setup
    saving_vs_baseline_monthly: int # vs the 12-month price of the same bundle
    install_days: int               # longest typical install -> drives the tracker ETA


def unit_price(p: Product, term_months: int) -> int:
    pct = TERM_DISCOUNT_PCT[term_months]
    return p.price_max - ((p.price_max - p.price_min) * pct + 50) // 100   # integer maths, round half up


def bundle_min_term(items: dict[str, int], catalogue: dict[str, Product]) -> int:
    return max(catalogue[pid].min_contract_months for pid in items)


def term_options(items: dict[str, int], catalogue: dict[str, Product]) -> list[int]:
    """Terms the owner may choose: standard terms that are >= the longest minimum contract in the bundle."""
    floor = bundle_min_term(items, catalogue)
    return [t for t in sorted(TERM_DISCOUNT_PCT) if t >= floor]


def build_quote(profile: BusinessProfile, items: dict[str, int], term_months: int,
                catalogue: dict[str, Product]) -> Quote:
    issues = validate_bundle(profile, items, catalogue)
    if issues:
        raise InvalidBundle(issues)
    if not items:
        raise InvalidBundle([Issue("", "EMPTY", "Choose at least one product.")])
    if term_months not in term_options(items, catalogue):
        raise InvalidTerm(f"{term_months}-month term not available; options: {term_options(items, catalogue)}")

    lines = []
    for pid, qty in items.items():
        p = catalogue[pid]
        up = unit_price(p, term_months)
        lines.append(QuoteLine(pid, p.name, p.unit, qty, up, up * qty, p.setup_fee, p.setup_fee * qty))
    monthly = sum(l.monthly for l in lines)
    setup = sum(l.setup for l in lines)
    base_monthly = sum(unit_price(catalogue[pid], BASELINE_TERM) * q for pid, q in items.items())
    return Quote(
        lines=tuple(lines), term_months=term_months, monthly_total=monthly, one_time_total=setup,
        first_payment=monthly + setup, contract_value=monthly * term_months + setup,
        saving_vs_baseline_monthly=max(0, base_monthly - monthly),
        install_days=max(catalogue[pid].install_days for pid in items),
    )


def available_addons(profile: BusinessProfile, items: dict[str, int], catalogue: dict[str, Product]) -> list[Product]:
    """Products the owner can toggle on right now (profile-eligible AND dependencies met by the current bundle)."""
    out = []
    for pid, p in catalogue.items():
        if pid in items:
            continue
        if not [i for i in validate_bundle(profile, {**items, pid: 1}, catalogue) if i.product_id == pid]:
            out.append(p)
    return out
