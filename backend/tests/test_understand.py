import pytest
from backend.app.understand import extract_profile, BusinessProfileDraft, SegmentEnum
from backend.app.llm import FakeLLM


def test_extract_profile_with_mock_llm():
    fake = FakeLLM(json_replies=[{
        "business_type": "prof",
        "employees": 12,
        "sites": 1,
        "city": "Lucknow",
        "has_gstin": True,
        "products_of_interest": ["P02", "P16"],
        "missing_fields": [],
    }])
    draft = extract_profile("Dental clinic in Lucknow with 12 staff members and GST", llm=fake)
    assert draft.business_type == SegmentEnum.prof
    assert draft.employees == 12
    assert draft.sites == 1
    assert draft.city == "Lucknow"
    assert draft.has_gstin is True
    assert "P02" in draft.products_of_interest
    assert draft.missing_fields == []


def test_extract_profile_detects_missing_fields():
    fake = FakeLLM(json_replies=[{
        "business_type": "micro",
        "employees": None,
        "sites": 1,
        "city": None,
        "has_gstin": None,
        "products_of_interest": ["P01"],
        "missing_fields": ["employees", "city"],
    }])
    draft = extract_profile("I run a bakery", llm=fake)
    assert draft.business_type == SegmentEnum.micro
    assert draft.employees is None
    assert "employees" in draft.missing_fields
    assert "city" in draft.missing_fields
