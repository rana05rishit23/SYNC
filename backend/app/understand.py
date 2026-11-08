"""Natural language understanding: transforms free text in ANY language into a structured profile.

Implements Section 3.2 of the Language-Agnostic Advisor spec:
- The LLM parses user messages in any Indian language / Hinglish into canonical English enums and numbers.
- Validated with Pydantic; invalid fields become null (never guessed).
- Downstream rules, pricing, and catalogue engines only consume the canonical English structure.
"""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator

from .llm import FakeLLM, LLM


class SegmentEnum(str, Enum):
    micro = "micro"
    prof = "prof"
    multi = "multi"
    mfg = "mfg"
    unknown = "unknown"


class BusinessProfileDraft(BaseModel):
    business_type: SegmentEnum = SegmentEnum.unknown
    employees: int | None = Field(default=None, description="Total staff/employee count")
    sites: int | None = Field(default=None, description="Number of locations/branches/stores")
    city: str | None = Field(default=None, description="City or district name")
    has_gstin: bool | None = Field(default=None, description="True if business has GST, False if not, None if unstated")
    products_of_interest: list[str] = Field(default_factory=list, description="Extracted catalogue IDs if mentioned")
    missing_fields: list[str] = Field(default_factory=list, description="Fields still needed before final quote")

    @field_validator("employees", mode="before")
    def validate_employees(cls, v: Any) -> int | None:
        if v is None:
            return None
        try:
            val = int(v)
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None

    @field_validator("sites", mode="before")
    def validate_sites(cls, v: Any) -> int | None:
        if v is None:
            return None
        try:
            val = int(v)
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None


UNDERSTAND_SYSTEM = (
    "You are an information extraction assistant for Indian SMB business onboarding. "
    "Read the user's message (which may be in English, Hindi, Marathi, Tamil, Bengali, or Hinglish) "
    "and extract their business profile into this JSON schema:\n"
    "{\n"
    '  "business_type": "micro" | "prof" | "multi" | "mfg" | "unknown",\n'
    '  "employees": integer or null,\n'
    '  "sites": integer or null,\n'
    '  "city": string or null,\n'
    '  "has_gstin": boolean or null,\n'
    '  "products_of_interest": ["P01", ...],\n'
    '  "missing_fields": ["employees", ...]\n'
    "}\n"
    "Rules:\n"
    "- 'micro': retail shop, kirana, bakery, pharmacy/chemist, salon, cafe, restaurant (single site)\n"
    "- 'prof': dental/doctor clinic, diagnostic lab, CA firm, law office, consulting, IT agency\n"
    "- 'multi': chain of stores, multiple branches, multi-location clinics/retail\n"
    "- 'mfg': factory, auto parts unit, textile mill, packaging plant, industrial workshop\n"
    "- Never invent numbers. If not explicitly mentioned, output null.\n"
    "- Identify which fields among ['business_type', 'employees', 'sites', 'city'] are still null and list them in 'missing_fields'."
)


def extract_profile(text: str, llm: LLM | None = None) -> BusinessProfileDraft:
    """Extract canonical business profile draft from free-text in any language."""
    if not text or not text.strip():
        return BusinessProfileDraft(
            business_type=SegmentEnum.unknown,
            missing_fields=["business_type", "employees", "sites", "city"],
        )

    client = llm or FakeLLM()
    try:
        raw_json = client.json(UNDERSTAND_SYSTEM, text)
        draft = BusinessProfileDraft.model_validate(raw_json)

        # Re-verify missing fields
        missing = []
        if draft.business_type == SegmentEnum.unknown:
            missing.append("business_type")
        if draft.employees is None:
            missing.append("employees")
        if draft.sites is None:
            missing.append("sites")
        if draft.city is None:
            missing.append("city")
        draft.missing_fields = missing

        return draft
    except Exception:
        # Fallback empty profile
        return BusinessProfileDraft(
            business_type=SegmentEnum.unknown,
            missing_fields=["business_type", "employees", "sites", "city"],
        )
