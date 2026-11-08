"""Fuzzy name matching for KYC Layer 3 of the Smart KYC pipeline.

Deterministic verification combining token-set (Jaccard) similarity and
SequenceMatcher to cross-verify OCR-extracted business names against
official registry records (GSTIN and Udyam).
"""
from __future__ import annotations

from dataclasses import dataclass
import difflib
import logging
import re
from typing import Any

from .registries import GSTINRecord, UdyamRecord

logger = logging.getLogger(__name__)

# Common business suffixes and honorifics to strip during normalization
_SUFFIXES = {
    "pvt",
    "ltd",
    "llp",
    "private",
    "limited",
    "partnership",
    "proprietorship",
    "m/s",
    "messrs",
    "shri",
    "smt",
}


@dataclass
class MatchResult:
    ocr_name: str
    registry_name: str  # the name it best matched against
    score: float        # 0.0-1.0
    verdict: str        # 'match' (>=0.8) | 'review' (0.6-0.79) | 'mismatch' (<0.6)


def _determine_verdict(score: float) -> str:
    """Return verdict category for a 0.0-1.0 similarity score."""
    if score >= 0.8:
        return "match"
    if score >= 0.6:
        return "review"
    return "mismatch"


def normalize_name(name: str | None) -> str:
    """Normalize a business or personal name for fuzzy comparison.

    Transformations:
    - Lowercase
    - Strip common prefixes/suffixes ('m/s', 'pvt', 'ltd', 'llp', etc.)
    - Strip punctuation (replacing with spaces)
    - Collapse multiple spaces into single spaces
    """
    if not name or not isinstance(name, str):
        return ""

    # 1. Lowercase
    text = name.lower()

    # 2. Strip 'm/s' which contains a slash before stripping punctuation
    text = re.sub(r"(?:\b|^)m\s*/\s*s(?:\b|$)", " ", text)

    # 3. Strip punctuation (keeping word characters and whitespace, replacing punctuation/symbols with spaces)
    text = re.sub(r"[^\w\s]|_", " ", text)

    # 4. Tokenize, strip common suffixes / noise words, and collapse multiple spaces
    tokens = [t for t in text.split() if t not in _SUFFIXES]

    return " ".join(tokens)


def token_set_similarity(a: str | None, b: str | None) -> float:
    """Compute fuzzy similarity between two names using Jaccard token similarity
    and SequenceMatcher, returning the higher of the two scores (0.0 - 1.0).
    """
    norm_a = normalize_name(a)
    norm_b = normalize_name(b)

    if not norm_a or not norm_b:
        return 0.0

    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())

    union = tokens_a | tokens_b
    if not union:
        return 0.0

    intersection = tokens_a & tokens_b
    jaccard = len(intersection) / len(union)

    # Secondary check: SequenceMatcher for partial/spelling matches
    seq_ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

    return max(jaccard, seq_ratio)


def match_name(
    ocr_name: str | None,
    registry_record: GSTINRecord | UdyamRecord | Any | None,
) -> MatchResult:
    """Compare OCR-extracted name against a registry record.

    - For GSTINRecord: compares against legal_name and trade_name, takes the higher score.
    - For UdyamRecord: compares against enterprise_name.
    - Returns MatchResult with OCR name, best-matching registry name, score, and verdict.
    """
    ocr_str = (ocr_name or "").strip()

    if not ocr_str or registry_record is None:
        reg_name = ""
        if isinstance(registry_record, GSTINRecord):
            reg_name = registry_record.trade_name or registry_record.legal_name or ""
        elif isinstance(registry_record, UdyamRecord):
            reg_name = registry_record.enterprise_name or ""
        elif hasattr(registry_record, "trade_name") or hasattr(registry_record, "legal_name"):
            reg_name = getattr(registry_record, "trade_name", "") or getattr(registry_record, "legal_name", "") or ""
        elif hasattr(registry_record, "enterprise_name"):
            reg_name = getattr(registry_record, "enterprise_name", "") or ""

        return MatchResult(
            ocr_name=ocr_str,
            registry_name=reg_name,
            score=0.0,
            verdict="mismatch",
        )

    best_name = ""
    best_score = 0.0

    if isinstance(registry_record, GSTINRecord):
        trade = registry_record.trade_name or ""
        legal = registry_record.legal_name or ""
        score_trade = token_set_similarity(ocr_str, trade)
        score_legal = token_set_similarity(ocr_str, legal)

        if score_trade > score_legal:
            best_score = score_trade
            best_name = trade
        elif score_legal > score_trade:
            best_score = score_legal
            best_name = legal
        else:
            best_score = score_trade
            best_name = trade or legal

    elif isinstance(registry_record, UdyamRecord):
        best_name = registry_record.enterprise_name or ""
        best_score = token_set_similarity(ocr_str, best_name)

    elif hasattr(registry_record, "trade_name") or hasattr(registry_record, "legal_name"):
        trade = getattr(registry_record, "trade_name", "") or ""
        legal = getattr(registry_record, "legal_name", "") or ""
        score_trade = token_set_similarity(ocr_str, trade)
        score_legal = token_set_similarity(ocr_str, legal)

        if score_trade > score_legal:
            best_score = score_trade
            best_name = trade
        elif score_legal > score_trade:
            best_score = score_legal
            best_name = legal
        else:
            best_score = score_trade
            best_name = trade or legal

    elif hasattr(registry_record, "enterprise_name"):
        best_name = getattr(registry_record, "enterprise_name", "") or ""
        best_score = token_set_similarity(ocr_str, best_name)

    else:
        return MatchResult(
            ocr_name=ocr_str,
            registry_name="",
            score=0.0,
            verdict="mismatch",
        )

    verdict = _determine_verdict(best_score)

    logger.debug(
        "Fuzzy matched OCR name '%s' with registry '%s': score=%.4f, verdict=%s",
        ocr_str,
        best_name,
        best_score,
        verdict,
    )

    return MatchResult(
        ocr_name=ocr_str,
        registry_name=best_name,
        score=best_score,
        verdict=verdict,
    )
